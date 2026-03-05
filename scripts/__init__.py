import sys
from pathlib import Path

# Add project root to python path to import config
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
