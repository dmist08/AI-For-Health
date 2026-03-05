from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "Data"  # Note: using given Data structure
    DATASET_DIR: Path = BASE_DIR / "Dataset"
    VIS_DIR: Path = BASE_DIR / "Visualizations"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Preprocessing
    FS_FLOW: int = 32
    FS_SPO2: int = 4
    WINDOW_SEC: int = 30
    STEP_SEC: int = 15
    LOWCUT: float = 0.17
    HIGHCUT: float = 0.40
    ORDER: int = 4
    
    # Training
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3
    DATALOADER_WORKERS: int = 0
    SEED: int = 42
    
    def __post_init__(self):
        self.DATASET_DIR.mkdir(exist_ok=True, parents=True)
        self.VIS_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

config = Config()
