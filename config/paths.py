from pathlib import Path

# Project root = ml-test-synthesis/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Workspace
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
TARGET_REPOS_DIR = WORKSPACE_DIR / "target-repos"
VENVS_DIR = WORKSPACE_DIR / "venvs"

# Data
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"
VALIDATION_DATA_DIR = DATA_DIR / "validation"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
