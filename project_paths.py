"""Local data and result locations for PAMED."""
from pathlib import Path
import os

REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PAMED_DATA_ROOT", REPOSITORY_ROOT / "data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("PAMED_RESULTS_ROOT", REPOSITORY_ROOT / "results")).expanduser().resolve()
