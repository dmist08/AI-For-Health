import argparse
import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.parsers import parse_signal_file, parse_events_file, parse_sleep_profile
from src.config import config
from src.utils import setup_logger

logger = setup_logger("Visualization")

def create_visualization(participant_dir: str):
    p_path = Path(participant_dir)
    p_id = p_path.name
    
    if not p_path.exists():
        logger.error(f"Participant directory {p_path} does not exist.")
        return

    logger.info(f"Loading data for {p_id}...")

    try:
        flow, rec_start = parse_signal_file(str(p_path / "nasal_airflow.txt"))
        thorac, _ = parse_signal_file(str(p_path / "thoracic_movement.txt"))
        spo2, _ = parse_signal_file(str(p_path / "spo2.txt"))
    except Exception as e:
        logger.error(f"Failed loading signals for {p_id}: {e}")
        return

    # Events and sleep profile
    events_path = p_path / "flow_events.txt"
    if events_path.exists():
        df_events = parse_events_file(str(events_path), rec_start)
    else:
        df_events = pd.DataFrame()
        logger.warning(f"{p_id} has no flow_events.txt")

    profile_path = p_path / "sleep_profile.txt"
    if profile_path.exists():
        df_profile = parse_sleep_profile(str(profile_path), rec_start)
    else:
        df_profile = pd.DataFrame()
        logger.warning(f"{p_id} has no sleep_profile.txt")

    # Plotting
    logger.info(f"Generating plot for {p_id}...")
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(f"Overnight Sleep Profile - {p_id}", fontsize=16, fontweight='bold')

    # Data to hours for x-axis
    flow.index = flow.index / 3600.0
    thorac.index = thorac.index / 3600.0
    spo2.index = spo2.index / 3600.0

    # 1. Nasal Airflow
    axes[0].plot(flow.index, flow.values, color='tab:blue', linewidth=0.5)
    axes[0].set_title("Nasal Airflow (32 Hz)")
    axes[0].set_ylabel("Amplitude")

    # 2. Thoracic Movement
    axes[1].plot(thorac.index, thorac.values, color='tab:orange', linewidth=0.5)
    axes[1].set_title("Thoracic Movement (32 Hz)")
    axes[1].set_ylabel("Amplitude")

    # 3. SpO2
    axes[2].plot(spo2.index, spo2.values, color='tab:red', linewidth=1)
    axes[2].set_title("SpO2 (4 Hz)")
    axes[2].set_ylabel("Oxygen Saturation (%)")
    axes[2].set_ylim(min(70, spo2.min() - 5), 105)

    # Function to overlay events on the first two axes
    def overlay_events(ax):
        if not df_events.empty:
            for _, ev in df_events.iterrows():
                start_h = ev['start_sec'] / 3600.0
                end_h = (ev['start_sec'] + ev['duration_sec']) / 3600.0
                color = 'red' if 'Apnea' in ev['label'] else 'yellow'
                ax.axvspan(start_h, end_h, color=color, alpha=0.3, label=ev['label'])

    overlay_events(axes[0])
    overlay_events(axes[1])

    # Deduplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')

    # 4. Sleep Profile
    if not df_profile.empty:
        df_profile['time_h'] = df_profile['time_sec'] / 3600.0
        # Map stages to y-values
        stage_map = {"Wake": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "N4": 1}
        y_vals = df_profile['stage'].map(stage_map)
        
        axes[3].step(df_profile['time_h'], y_vals, where='post', color='indigo', linewidth=2)
        axes[3].set_yticks(list(stage_map.values()))
        axes[3].set_yticklabels(list(stage_map.keys()))
    else:
        axes[3].text(0.5, 0.5, "Sleep Profile Not Available", ha='center', va='center', fontsize=18)
        axes[3].set_yticks([])

    axes[3].set_title("Sleep Stages")
    axes[3].set_xlabel("Time (Hours since recording start)")
    axes[3].set_xlim(left=0, right=max(flow.index[-1], thorac.index[-1], spo2.index[-1]))

    out_pdf = config.VIS_DIR / f"{p_id}_visualization.pdf"
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {out_pdf}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create sleep visualizations")
    parser.add_argument("-name", type=str, required=True, help="Path to participant Data directory (e.g. Data/AP01)")
    args = parser.parse_args()
    
    create_visualization(args.name)
