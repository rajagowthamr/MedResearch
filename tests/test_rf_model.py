"""The Random Forest bundle, as pytest.

Was a top-level script of asserts that only ran if you were cd'd into the repo
root and had already trained the model. Now it skips cleanly when the artifact
is absent, so `pytest` passes on a fresh clone.
"""
import pytest

from medresearch import config
from medresearch.rf.pipeline import VALID_CLASSES, load_bundle, make_row

pytestmark = pytest.mark.requires_checkpoint

PATIENT = {
    "Age": 45, "Gender": "Male", "Blood Type": "A+",
    "Medical Condition": "Diabetes", "Insurance Provider": "Aetna",
    "Billing Amount": 25000, "Admission Type": "Urgent",
    "Medication": "Aspirin", "Length of Stay": 10,
}


@pytest.fixture(scope="module")
def bundle():
    if not config.RF_MODEL.exists():
        pytest.skip(f"{config.RF_MODEL} not built — run scripts/train_rf.py")
    return load_bundle(config.RF_MODEL)


def test_bundle_has_model_encoders_and_columns(bundle):
    assert bundle.model is not None
    assert bundle.encoders and bundle.columns


def test_prediction_is_a_valid_class(bundle):
    pred = bundle.model.predict(make_row(PATIENT, bundle))[0]
    assert pred in VALID_CLASSES


def test_same_input_gives_same_output(bundle):
    """A RandomForest is deterministic once fitted; if this fails, something is
    reseeding or the encoders are being refitted."""
    a = bundle.model.predict(make_row(PATIENT, bundle))[0]
    b = bundle.model.predict(make_row(PATIENT, bundle))[0]
    assert a == b


def test_probabilities_sum_to_one(bundle):
    probs = bundle.model.predict_proba(make_row(PATIENT, bundle))[0]
    assert abs(sum(probs) - 1.0) < 1e-3


def test_make_row_restores_training_column_order(bundle):
    """sklearn matches features POSITIONALLY. A reordered frame predicts on
    scrambled inputs without raising, so this is the test that catches it."""
    shuffled = dict(reversed(list(PATIENT.items())))
    assert list(make_row(shuffled, bundle).columns) == bundle.columns
