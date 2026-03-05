import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def standardize_data_structure(source_base: str, dest_base: str):
    """
    Copies and renames files from the messy source directory into a clean, 
    standardized structure required for the parsing scripts.
    """
    src_path = Path(source_base)
    dest_path = Path(dest_base)
    
    if not src_path.exists():
        logging.error(f"Source directory {src_path} does not exist.")
        return

    dest_path.mkdir(exist_ok=True, parents=True)
    
    # Expected standard names for our internal logic
    STANDARD_FILES = {
        'flow': 'nasal_airflow.txt',
        'thorac': 'thoracic_movement.txt',
        'spo2': 'spo2.txt',
        'flow events': 'flow_events.txt',
        'sleep profile': 'sleep_profile.txt'
    }

    participants = ['AP01', 'AP02', 'AP03', 'AP04', 'AP05']
    
    for participant in participants:
        src_p_dir = src_path / participant
        dest_p_dir = dest_path / participant
        
        if not src_p_dir.exists():
            logging.warning(f"Participant {participant} not found in {src_path}")
            continue
            
        dest_p_dir.mkdir(exist_ok=True, parents=True)
        
        # Match files loosely based on prefixes
        copied_files = set()
        for file_path in src_p_dir.iterdir():
            if not file_path.is_file():
                continue
                
            fname_lower = file_path.name.lower()
            
            # Identify which standard file this represents
            if 'flow events' in fname_lower:
                std_name = STANDARD_FILES['flow events']
            elif 'sleep profile' in fname_lower:
                std_name = STANDARD_FILES['sleep profile']
            elif 'flow' in fname_lower:
                std_name = STANDARD_FILES['flow']
            elif 'thorac' in fname_lower:
                std_name = STANDARD_FILES['thorac']
            elif 'spo2' in fname_lower:
                std_name = STANDARD_FILES['spo2']
            else:
                logging.info(f"Ignoring unrecognized file: {file_path.name}")
                continue
                
            shutil.copy2(file_path, dest_p_dir / std_name)
            copied_files.add(std_name)
            
        logging.info(f"{participant}: Copied {len(copied_files)} files -> {list(copied_files)}")

if __name__ == '__main__':
    source_dir = "internship/Data"
    dest_dir = "Data"
    logging.info(f"Setting up organized data from '{source_dir}' to '{dest_dir}'...")
    standardize_data_structure(source_dir, dest_dir)
    logging.info("Data organization complete.")
