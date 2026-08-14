"""Paths resolve correctly, and the MLflow directories stay where MLflow expects."""
import os
import subprocess
import sys

from medresearch import config


def test_project_root_contains_pyproject():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_mlflow_paths_are_at_the_project_root():
    """mlflow.db stores artifact locations as ABSOLUTE paths. If these ever move
    under data/ or models/, all historical runs lose their artifacts and the
    experiments/runs tables need an UPDATE to repair. Pin them here so a future
    "tidy up" fails a test instead of silently breaking the run history."""
    assert config.MLRUNS_DIR.parent == config.PROJECT_ROOT
    assert config.MLFLOW_DB.parent == config.PROJECT_ROOT
    assert config.MLFLOW_TRACKING_URI.startswith("sqlite:///")


def test_all_paths_are_absolute():
    """Relative paths were the old bug: scripts only ran from the repo root."""
    paths = [v for k, v in vars(config).items()
             if k.isupper() and hasattr(v, "is_absolute")]
    assert paths, "no Path constants found in config"
    assert all(p.is_absolute() for p in paths)


def test_data_paths_live_under_data_dir():
    for p in (config.MEDICAL_TEXT, config.TRANSCRIPTIONS, config.HEALTHCARE_CSV):
        assert config.RAW_DIR in p.parents
    for p in (config.NOTE_TRAIN, config.NOTE_VAL, config.CHAT_DATA,
              config.CHAT_HOLDOUT, config.TOKENIZER_JSON):
        assert config.PROCESSED_DIR in p.parents


def test_checkpoint_path_uses_model_name():
    assert config.MODEL_NAME == "m1"
    assert config.checkpoint_path("v5-large").name == "m1_v5-large.pt"


def test_require_raises_with_an_actionable_message(tmp_path):
    missing = tmp_path / "nope.txt"
    try:
        config.require(missing, "Build it with: python tools/fetch_wiki_medical.py")
    except FileNotFoundError as e:
        assert "fetch_wiki_medical" in str(e)
        assert "MEDRESEARCH_ROOT" in str(e)
    else:
        raise AssertionError("require() should have raised")


def test_root_is_independent_of_working_directory():
    """The whole point of config.py: importing from / must find the same root."""
    code = "from medresearch import config; print(config.PROJECT_ROOT)"
    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(config.PROJECT_ROOT)
