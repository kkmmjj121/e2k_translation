"""Repository-relative resources, optionally overridden with E2K_* variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def configured(name, default):
    value = Path(os.path.expandvars(os.environ.get(name, str(default)))).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value

DATA_ROOT = configured("E2K_DATA_ROOT", PROJECT_ROOT / "data")
DATA_CSV = configured("E2K_DATA_CSV", PROJECT_ROOT / "en2ko.csv")
MODEL_ROOT = configured("E2K_MODEL_ROOT", PROJECT_ROOT / "model")
SELF_MODEL_DIR = MODEL_ROOT / "self"
TOKENIZER_ROOT = configured("E2K_TOKENIZER_ROOT", PROJECT_ROOT)
PRETRAINED_CHECKPOINT = configured("E2K_PRETRAINED_CHECKPOINT", MODEL_ROOT / "en2ko/checkpoint-64110")
CUSTOM_CHECKPOINT = configured("E2K_CUSTOM_CHECKPOINT", SELF_MODEL_DIR / "last_model.pt")

def resume_checkpoint(filename):
    return configured("E2K_RESUME_CHECKPOINT", SELF_MODEL_DIR / filename)
