from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEFAULT_START_DATE = "2010-01-01"
DEFAULT_HORIZON = 10
ROLLING_WINDOWS = (5, 20, 60)


def ensure_project_dirs() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
