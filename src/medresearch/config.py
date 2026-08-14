"""Every path in the project, in one place.

WHY THIS FILE EXISTS
--------------------
The flat layout hardcoded relative paths in 12 different modules -- "medical_text.txt"
in train.py, benchmark.py, gpt_app.py and tokenizer.py; "health_model.pkl" in three
more. Two consequences, both of which actually bit:

  1. Every script only ran if your shell happened to be cd'd into the repo root.
     Run `python scripts/train_gpt.py` from anywhere else and it died on a
     FileNotFoundError with no hint as to why.
  2. Moving or renaming a data file meant grepping the whole tree and hoping.

Now there is exactly one definition of each path, resolved from the project root
rather than the working directory, so scripts run from anywhere.

THE MLFLOW PATHS ARE NOT NEGOTIABLE
-----------------------------------
MLRUNS_DIR and MLFLOW_DB must stay at the repo root. MLflow stores the artifact
location of every run as an ABSOLUTE path inside mlflow.db -- 16 runs currently
point at <root>/mlruns/1/<run_id>/artifacts. Moving the directory silently breaks
every historical run's artifacts, and repairing it means an UPDATE across the
experiments and runs tables. Do not "tidy" these into data/ or models/.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the repo root, in priority order.

    1. $MEDRESEARCH_ROOT, so a container or CI job can point at a mounted volume.
    2. Walk up from this file looking for pyproject.toml. This is what makes an
       editable install (`pip install -e .`) work from any working directory.
    3. Fall back to three levels up (src/medresearch/config.py -> root), which is
       correct for a plain source checkout that was never installed.
    """
    if env := os.environ.get("MEDRESEARCH_ROOT"):
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


PROJECT_ROOT = _find_project_root()

# --- source-controlled inputs ------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Corpora. MEDICAL_TEXT is the 58MB / 60.7M-character pretraining corpus.
MEDICAL_TEXT = RAW_DIR / "medical_text.txt"
TRANSCRIPTIONS = RAW_DIR / "transcriptions.txt"
HEALTHCARE_CSV = RAW_DIR / "healthcare_dataset.csv"

NOTE_TRAIN = PROCESSED_DIR / "note_train.txt"
NOTE_VAL = PROCESSED_DIR / "note_val.txt"
CHAT_DATA = PROCESSED_DIR / "chat_data.txt"
CHAT_HOLDOUT = PROCESSED_DIR / "chat_holdout.txt"
TOKENIZER_JSON = PROCESSED_DIR / "tokenizer.json"

# --- generated artifacts -----------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
RF_MODEL = MODELS_DIR / "health_model.pkl"
LEGACY_GPT = MODELS_DIR / "gpt_medical.pt"      # the pre-versioning v1 checkpoint

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# --- model identity ----------------------------------------------------------
# The language model is called m1. Use MODEL_NAME for anything user-facing:
# checkpoint filenames, UI titles, generated docs.
MODEL_NAME = "m1"

# --- experiment tracking (see the docstring: do not move these) --------------
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

# These two stay "medresearch-gpt" ON PURPOSE, even though the model is now m1.
#
# The MLflow registry keys versions by name: medresearch-gpt already holds v1-v4
# plus the @champion alias, and the run training right now is registering into
# it. Renaming starts an EMPTY registry at version 1 and orphans the existing
# lineage -- "is this better than the last one?" stops being answerable, which
# is the entire reason the registry exists.
#
# To rename anyway, migrate rather than just editing this string:
#   client.rename_registered_model("medresearch-gpt", "m1")
# which carries the versions and aliases across. Then update both constants.
EXPERIMENT_NAME = "medresearch-gpt"
REGISTERED_MODEL_NAME = "medresearch-gpt"


def checkpoint_path(version: str) -> Path:
    """Checkpoint file for a training version, e.g. "v5-large" -> m1_v5-large.pt.

    Existing checkpoints on disk use the older gpt_medical_ prefix; find_checkpoints()
    matches both so nothing already trained becomes invisible.
    """
    return CHECKPOINTS_DIR / f"{MODEL_NAME}_{version}.pt"


def find_checkpoints() -> list[Path]:
    """Every checkpoint, newest first, under either naming scheme."""
    if not CHECKPOINTS_DIR.is_dir():
        return []
    found = list(CHECKPOINTS_DIR.glob("*.pt"))
    if LEGACY_GPT.exists():
        found.append(LEGACY_GPT)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def ensure_dirs() -> None:
    """Create the output directories. Inputs are deliberately not created --
    a missing corpus should raise, not silently train on nothing."""
    for d in (CHECKPOINTS_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require(path: Path, hint: str = "") -> Path:
    """Fail with an actionable message instead of a bare FileNotFoundError.

    The old code let a missing corpus surface as `[Errno 2] 'medical_text.txt'`
    from deep inside a training script, which says nothing about how to fix it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found."
            + (f" {hint}" if hint else "")
            + f"\n(project root resolved to {PROJECT_ROOT}; "
            "override with $MEDRESEARCH_ROOT)"
        )
    return path
