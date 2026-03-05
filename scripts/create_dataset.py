"""
create_dataset.py — Signal preprocessing and windowing pipeline.

Usage:
    python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
"""
import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.parsers import parse_signal_file, parse_events_file
from src.config import config
from src.utils import setup_logger

logger = setup_logger("CreateDataset")


# ─── Signal Processing Helpers ───────────────────────────────────────────────

def apply_bandpass(signal: np.ndarray, fs: float, lowcut: float = 0.17,
                   highcut: float = 0.40, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for breathing signals (Flow, Thoracic)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
    return filtfilt(b, a, signal)

def apply_lowpass(signal: np.ndarray, fs: float, cutoff: float = 1.0,
                  order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter for SpO2 (removes sensor jitter)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

def upsample_spo2(spo2_values: np.ndarray, from_fs: int = 4,
                  to_fs: int = 32) -> np.ndarray:
    """
    Upsample SpO2 from 4Hz to 32Hz using polyphase resampling.
    This gives it the same length as the Flow/Thoracic arrays so all 3 channels
    can be stacked into a single (3, N) tensor.
    """
    return resample_poly(spo2_values, up=to_fs, down=from_fs)


# ─── Windowing & Labeling ─────────────────────────────────────────────────────

def label_to_int(label: str) -> int:
    """
    Binary classification:
        0 = Normal
        1 = Event (Hypopnea or any Apnea variant)
    """
    return 0 if label.strip() == "Normal" else 1

def get_window_label(win_start: float, win_sec: int,
                     df_events: pd.DataFrame) -> str:
    """
    Assigns a label to a 30-second window using the >50% overlap rule.
    If any event overlaps more than 15 seconds with this window, return that event.
    """
    if df_events.empty:
        return "Normal"

    win_end = win_start + win_sec
    threshold = win_sec * 0.5

    for _, ev in df_events.iterrows():
        ev_start = ev['start_sec']
        ev_end = ev_start + ev['duration_sec']
        overlap = max(0, min(win_end, ev_end) - max(win_start, ev_start))
        if overlap > threshold:
            return str(ev['label']).strip()

    return "Normal"

def extract_windows(signal: np.ndarray, fs: int, window_sec: int,
                    step_sec: int) -> tuple[list, list]:
    """Slides a fixed-length window over a signal, returning (windows, start_indices)."""
    win_pts = window_sec * fs
    step_pts = step_sec * fs
    windows, starts = [], []
    for i in range(0, len(signal) - win_pts + 1, step_pts):
        windows.append(signal[i:i + win_pts])
        starts.append(i / fs)
    return windows, starts


# ─── Per-Participant Processing ───────────────────────────────────────────────

def process_participant(p_dir: Path) -> dict | None:
    p_id = p_dir.name

    # 1. Load and parse signals
    try:
        flow, rec_start = parse_signal_file(str(p_dir / "nasal_airflow.txt"))
        thorac, _ = parse_signal_file(str(p_dir / "thoracic_movement.txt"))
        spo2, _ = parse_signal_file(str(p_dir / "spo2.txt"))
    except Exception as e:
        logger.error(f"{p_id}: Failed to load signals — {e}")
        return None

    # 2. Load events
    events_path = p_dir / "flow_events.txt"
    if events_path.exists():
        df_events = parse_events_file(str(events_path), rec_start)
    else:
        df_events = pd.DataFrame()
        logger.warning(f"{p_id}: No events file — all windows will be Normal.")

    # 3. Filter
    flow_filtered = apply_bandpass(flow.values, fs=config.FS_FLOW)
    thorac_filtered = apply_bandpass(thorac.values, fs=config.FS_FLOW)
    spo2_filtered = apply_lowpass(spo2.values, fs=config.FS_SPO2)

    # 4. Upsample SpO2 from 4Hz → 32Hz to match Flow/Thoracic length
    spo2_upsampled = upsample_spo2(spo2_filtered, from_fs=config.FS_SPO2,
                                   to_fs=config.FS_FLOW)

    # 5. Align all signals to same minimum length
    min_len = min(len(flow_filtered), len(thorac_filtered), len(spo2_upsampled))
    flow_filtered = flow_filtered[:min_len]
    thorac_filtered = thorac_filtered[:min_len]
    spo2_upsampled = spo2_upsampled[:min_len]

    # 6. Extract windows
    flow_wins, win_starts = extract_windows(flow_filtered, config.FS_FLOW,
                                            config.WINDOW_SEC, config.STEP_SEC)
    thorac_wins, _ = extract_windows(thorac_filtered, config.FS_FLOW,
                                     config.WINDOW_SEC, config.STEP_SEC)
    spo2_wins, _ = extract_windows(spo2_upsampled, config.FS_FLOW,
                                   config.WINDOW_SEC, config.STEP_SEC)

    n = min(len(flow_wins), len(thorac_wins), len(spo2_wins))

    X, y, label_strs = [], [], []
    for i in range(n):
        # Stack 3 channels → (3, 960)
        window = np.stack([flow_wins[i], thorac_wins[i], spo2_wins[i]], axis=0)
        label_str = get_window_label(win_starts[i], config.WINDOW_SEC, df_events)

        X.append(window)
        label_strs.append(label_str)
        y.append(label_to_int(label_str))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    label_counts = {s: label_strs.count(s) for s in set(label_strs)}
    logger.info(f"{p_id}: {len(X)} windows → {label_counts}")

    return {"X": X, "y": y, "label_strs": label_strs, "p_id": p_id,
            "win_starts": win_starts[:n]}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(in_dir: str, out_dir: str) -> None:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    participants = sorted([d for d in in_path.iterdir() if d.is_dir()])
    if not participants:
        logger.error(f"No participant folders found in {in_path}")
        return

    dataset_dict = {}
    csv_rows = []

    for p_dir in tqdm(participants, desc="Processing"):
        result = process_participant(p_dir)
        if result is None:
            continue

        p_id = result["p_id"]
        dataset_dict[p_id] = {"X": result["X"], "y": result["y"]}

        # Flatten for CSV: participant ID + label + 3*960 = 2882 columns
        for i, (x_win, label_str, label_int, t_start) in enumerate(
                zip(result["X"], result["label_strs"], result["y"],
                    result["win_starts"])):
            row = [p_id, round(t_start, 2), label_str, label_int]
            row.extend(x_win.flatten().tolist())
            csv_rows.append(row)

    # Build CSV column names
    n_pts = config.WINDOW_SEC * config.FS_FLOW  # 960
    cols = ["Participant", "WinStart_s", "LabelStr", "LabelInt"]
    cols += [f"Flow_{i}" for i in range(n_pts)]
    cols += [f"Thorac_{i}" for i in range(n_pts)]
    cols += [f"SpO2_{i}" for i in range(n_pts)]

    df = pd.DataFrame(csv_rows, columns=cols)

    pkl_path = out_path / "breathing_dataset.pkl"
    csv_path = out_path / "breathing_dataset.csv"

    with open(pkl_path, "wb") as f:
        pickle.dump(dataset_dict, f)
    logger.info(f"Saved PKL: {pkl_path} ({pkl_path.stat().st_size / 1e6:.1f} MB)")

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    logger.info("\n=== GLOBAL LABEL DISTRIBUTION ===")
    logger.info(str(df["LabelStr"].value_counts()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create windowed dataset from raw signals.")
    parser.add_argument("-in_dir", type=str, default="Data",
                        help="Root directory containing participant folders.")
    parser.add_argument("-out_dir", type=str, default="Dataset",
                        help="Output directory for CSV and PKL files.")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
