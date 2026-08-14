"""Random Forest baseline: feature prep, training, and the saved bundle.

WHY A BUNDLE AND NOT JUST THE MODEL
-----------------------------------
A fitted RandomForest is useless on its own here. Prediction needs three things
that must match training exactly:

  model     the estimator
  encoders  the LabelEncoders -- "Male" has to become the SAME integer it was
            during fit, and a fresh LabelEncoder would assign a different one
  columns   the column ORDER, because sklearn matches features positionally,
            not by name. Reordering silently predicts on scrambled inputs.

The old code rebuilt `make_row` separately in the Streamlit app and in the test
script; both had to stay in sync with training by hand. Now all three live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dropped because they identify a row rather than describe a patient (Name,
# Doctor, Hospital, Room Number), or because they are superseded by the
# engineered Length of Stay (the two dates).
DROP_COLUMNS = [
    "Name", "Doctor", "Hospital",
    "Date of Admission", "Discharge Date",
    "Room Number", "Test Results",
]
TARGET = "Test Results"
VALID_CLASSES = ["Normal", "Abnormal", "Inconclusive"]


@dataclass
class Bundle:
    """Everything prediction needs. Persisted as a single .pkl."""
    model: Any
    encoders: dict[str, LabelEncoder]
    columns: list[str]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean, engineer, encode. Returns (X, y, encoders)."""
    df = df.copy()
    df["Name"] = df["Name"].str.title()          # "bObbY jAcksOn" -> "Bobby Jackson"
    df = df.drop_duplicates()

    # Raw dates carry no signal; the span between them does.
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    return X, y, encoders


def train_model(X, y, encoders, *, n_estimators: int = 12, seed: int = 42):
    """Fit on 80%, report on the held-out 20%. Returns (Bundle, report_text)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=seed
    )
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    return Bundle(model=model, encoders=encoders, columns=list(X.columns)), report


def make_row(patient: dict, bundle: Bundle) -> pd.DataFrame:
    """One patient dict -> a single-row frame the model can score.

    Applies the training-time encoders and restores the training column order.
    """
    row = pd.DataFrame([patient])
    for col, le in bundle.encoders.items():
        row[col] = le.transform(row[col])
    return row[bundle.columns]


def save_bundle(bundle: Bundle, path: str | Path) -> None:
    # Saved as a plain dict, not the dataclass, so an old .pkl written before
    # this refactor still loads and so unpickling does not require this module.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": bundle.model, "encoders": bundle.encoders, "columns": bundle.columns},
        path,
    )


def load_bundle(path: str | Path) -> Bundle:
    raw = joblib.load(path)
    return Bundle(model=raw["model"], encoders=raw["encoders"], columns=raw["columns"])
