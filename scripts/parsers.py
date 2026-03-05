import re
import pandas as pd
from datetime import datetime
import logging

def parse_datetime(dt_str: str) -> datetime:
    """
    Parses datetime from the two known formats in the headers:
    A: DD-MM-YYYY HH:MM:SS (or DD.MM.YYYY)
    B: M/D/YYYY H:MM:SS [AM/PM]
    
    If it ends with AM/PM it's format B, else format A.
    """
    dt_str = dt_str.strip()
    if 'AM' in dt_str or 'PM' in dt_str:
        return datetime.strptime(dt_str, "%m/%d/%Y %I:%M:%S %p")
    else:
        # Sometimes separator is '-' or '.'
        dt_str = dt_str.replace('.', '-')
        return datetime.strptime(dt_str, "%d-%m-%Y %H:%M:%S")

def parse_row_datetime(dt_str: str) -> datetime:
    """
    Parses DD.MM.YYYY HH:MM:SS,mmm into datetime.
    """
    dt_str = dt_str.strip()
    return datetime.strptime(dt_str, "%d.%m.%Y %H:%M:%S,%f")

def parse_signal_file(path: str) -> pd.Series:
    """
    Reads signal file.
    Returns pd.Series with time in *seconds* (relative to recording start).
    """
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        
    start_time_str = None
    data_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith("Start Time:"):
            start_time_str = line.split("Start Time:", 1)[1].strip()
        if line.startswith("Data:"):
            data_idx = i + 1
            break
            
    if start_time_str is None or data_idx == -1:
        raise ValueError(f"Invalid format in signal file: {path}")
        
    recording_start = parse_datetime(start_time_str)
    
    times_sec = []
    values = []
    
    for line in lines[data_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            time_part, val_part = line.split(';')
            row_dt = parse_row_datetime(time_part)
            val = float(val_part.strip())
            
            # Seconds from start
            t_sec = (row_dt - recording_start).total_seconds()
            times_sec.append(t_sec)
            values.append(val)
        except ValueError:
            continue
            
    return pd.Series(values, index=times_sec), recording_start

def parse_events_file(path: str, recording_start: datetime) -> pd.DataFrame:
    """
    Parses events file.
    Merges 'Mixed Apnea' into 'Obstructive Apnea'.
    Discards 'Body event'.
    Returns DataFrame: [start_sec, duration_sec, label, stage]
    """
    events = []
    
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Signal ID:") or line.startswith("Start Time:") or line.startswith("Unit:") or line.startswith("Signal Type:"):
            continue
            
        # Example format: 30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1
        try:
            time_range, duration, label, stage = line.split(';')
            time_range = time_range.strip()
            label = label.strip()
            stage = stage.strip()
            duration = float(duration.strip())
            
            # Discard Body event
            if label == "Body event":
                continue
                
            # Merge Mixed Apnea into Obstructive Apnea
            if label == "Mixed Apnea":
                label = "Obstructive Apnea"
                
            # "30.05.2024 23:48:45,119-23:49:01,408"
            start_str, end_time_only = time_range.split('-')
            start_dt = parse_row_datetime(start_str)
            
            start_sec = (start_dt - recording_start).total_seconds()
            
            events.append({
                'start_sec': start_sec,
                'duration_sec': duration,
                'label': label,
                'stage': stage
            })
            
        except ValueError:
            continue
            
    return pd.DataFrame(events)

def parse_sleep_profile(path: str, recording_start: datetime) -> pd.DataFrame:
    """
    Parses sleep profile.
    Returns DataFrame: [time_sec, stage]
    """
    epochs = []
    
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Skip headers
        if not line or ';' not in line or line.startswith("Signal ID:") or line.startswith("Start Time:") or line.startswith("Events list:") or line.startswith("Rate:") or line.startswith("Unit:") or line.startswith("Signal Type:"):
            continue
            
        try:
            time_str, stage = line.split(';')
            time_str = time_str.strip()
            stage = stage.strip()
            
            dt = parse_row_datetime(time_str)
            t_sec = (dt - recording_start).total_seconds()
            
            if stage != "A": # Ignore "A" epoch artifact markers
                epochs.append({
                    'time_sec': t_sec,
                    'stage': stage
                })
        except ValueError:
            continue
            
    return pd.DataFrame(epochs)
