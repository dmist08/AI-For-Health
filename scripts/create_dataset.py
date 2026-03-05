import os
import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.parsers import parse_signal_file, parse_events_file
from src.config import config
from src.utils import setup_logger

logger = setup_logger("CreateDataset")

def apply_bandpass(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter to breathing signals."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

def apply_lowpass(signal: np.ndarray, fs: float, cutoff: float = 1.0, order: int = 2) -> np.ndarray:
    """Applies a zero-phase Butterworth lowpass filter to SpO2."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def extract_windows(series: pd.Series, fs: int, window_sec: int, step_sec: int) -> list:
    """
    Extracts continuous signal windows. 
    Assumes `series.index` is monotonic time in seconds relative to recording start.
    We iterate over time (not just purely by array slice) to handle potential gaps,
    but here we slice by array indices assuming no mid-stream sensor gaps as per dataset spec.
    """
    window_pts = window_sec * fs
    step_pts = step_sec * fs
    values = series.values
    times = series.index.values
    
    windows = []
    start_times = []
    
    for start_idx in range(0, len(values) - window_pts + 1, step_pts):
        windows.append(values[start_idx : start_idx + window_pts])
        start_times.append(times[start_idx])
        
    return windows, start_times

def get_window_label(win_start: float, win_duration: float, df_events: pd.DataFrame) -> str:
    """Assigns label string to a window if an event overlaps by >50% (>15s)."""
    if df_events.empty:
        return "Normal"
        
    win_end = win_start + win_duration
    required_overlap = win_duration * 0.5

    for _, event in df_events.iterrows():
        ev_start = event['start_sec']
        ev_duration = event['duration_sec']
        ev_end = ev_start + ev_duration
        label = event['label']
        
        # Calculate intersection
        overlap_start = max(win_start, ev_start)
        overlap_end = min(win_end, ev_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > required_overlap:
            return str(label).strip()
            
    return "Normal"

def map_label_to_int(label: str) -> int:
    """Map string classes to simple integers for ML."""
    label_map = {
        "Normal": 0,
        "Hypopnea": 1,
        "Obstructive Apnea": 2
    }
    return label_map.get(label, 0)

def process_participant(participant_dir: str):
    p_path = Path(participant_dir)
    p_id = p_path.name
    
    try:
        flow, start_time = parse_signal_file(str(p_path / "nasal_airflow.txt"))
        thorac, _ = parse_signal_file(str(p_path / "thoracic_movement.txt"))
        spo2, _ = parse_signal_file(str(p_path / "spo2.txt"))
        
        events_path = p_path / "flow_events.txt"
        if events_path.exists():
            df_events = parse_events_file(str(events_path), start_time)
        else:
            df_events = pd.DataFrame()
            logger.warning(f"No events file found for {p_id}. Defaulting to Normal windows.")
            
    except Exception as e:
        logger.error(f"Failed loading parsing for {p_id}: {e}")
        return None

    # Step 1: Filter
    flow.loc[:] = apply_bandpass(flow.values, fs=config.FS_FLOW, lowcut=config.LOWCUT, highcut=config.HIGHCUT, order=config.ORDER)
    thorac.loc[:] = apply_bandpass(thorac.values, fs=config.FS_FLOW, lowcut=config.LOWCUT, highcut=config.HIGHCUT, order=config.ORDER)
    spo2.loc[:] = apply_lowpass(spo2.values, fs=config.FS_SPO2, cutoff=1.0) # Smooth SpO2 jitter

    # Step 2: Windowing (no upsampling for SP02)
    flow_wins, flow_times = extract_windows(flow, config.FS_FLOW, config.WINDOW_SEC, config.STEP_SEC)
    thorac_wins, _ = extract_windows(thorac, config.FS_FLOW, config.WINDOW_SEC, config.STEP_SEC)
    spo2_wins, _ = extract_windows(spo2, config.FS_SPO2, config.WINDOW_SEC, config.STEP_SEC)

    # We must restrict to min logical chunks to avoid out-of-bounds at exactly the end
    min_len = min(len(flow_wins), len(thorac_wins), len(spo2_wins))
    
    X_32 = []
    X_4 = []
    labels = []
    flat_rows = [] # For CSV
    
    for i in range(min_len):
        # 32Hz [2, 960]
        x32 = np.vstack([flow_wins[i], thorac_wins[i]])
        # 4Hz [1, 120]
        x4 = np.array([spo2_wins[i]])
        
        win_start_sec = flow_times[i]
        label_str = get_window_label(win_start_sec, config.WINDOW_SEC, df_events)
        label_id = map_label_to_int(label_str)
        
        X_32.append(x32)
        X_4.append(x4)
        labels.append(label_id)
        
        # Flattening for CSV: Flow (960) + Thorac (960) + SpO2 (120) = 2040 features
        row = [p_id, win_start_sec, config.WINDOW_SEC, label_str, label_id]
        row.extend(x32[0].tolist()) # Flow
        row.extend(x32[1].tolist()) # Thorac
        row.extend(x4[0].tolist())  # SpO2
        flat_rows.append(row)

    return {
        "X_32": np.array(X_32, dtype=np.float32), 
        "X_4": np.array(X_4, dtype=np.float32), 
        "labels": np.array(labels, dtype=np.longlong),
        "flat_rows": flat_rows
    }

def main(in_dir: str):
    participants = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    dataset_dict = {}
    csv_rows = []
    
    for p in tqdm(participants, desc="Processing Participants"):
        p_path = os.path.join(in_dir, p)
        res = process_participant(p_path)
        if res:
            dataset_dict[p] = {
                "X_32": res["X_32"],
                "X_4":  res["X_4"],
                "y":    res["labels"]
            }
            csv_rows.extend(res["flat_rows"])
            
            logger.info(f"{p} Layout: X_32: {res['X_32'].shape}, X_4: {res['X_4'].shape}, y: {res['labels'].shape}")

    # Generate CSV columns
    cols = ['Participant', 'WinStartSec', 'WinDurSec', 'LabelStr', 'LabelID']
    cols += [f'Flow_{i}' for i in range(config.FS_FLOW * config.WINDOW_SEC)]
    cols += [f'Thorac_{i}' for i in range(config.FS_FLOW * config.WINDOW_SEC)]
    cols += [f'SpO2_{i}' for i in range(config.FS_SPO2 * config.WINDOW_SEC)]
    
    df_out = pd.DataFrame(csv_rows, columns=cols)
    
    logger.info("Saving Data...")
    
    # Save CSV
    csv_path = config.DATASET_DIR / "breathing_dataset.csv"
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path.stat().st_size / 1e6:.2f} MB")
    
    # Save Pickle
    pkl_path = config.DATASET_DIR / "breathing_dataset.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    logger.info(f"Saved PKL: {pkl_path.stat().st_size / 1e6:.2f} MB")

    val_counts = df_out['LabelStr'].value_counts()
    logger.info(f"Final Global Label Distribution:\n{val_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", type=str, default="Data")
    args = parser.parse_args()
    main(args.in_dir)
