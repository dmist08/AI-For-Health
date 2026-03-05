import os
import random
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

LOG_DIR = Path(__file__).parent.parent / "logs"

def set_seed(seed: int = 42) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Configures and returns a logger that writes to both console and a log file.
    Log file is saved to logs/{name}_{timestamp}.log
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{name}_{timestamp}.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_file}")

    return logger

def get_device() -> torch.device:
    """Returns the optimal available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
