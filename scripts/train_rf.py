"""Train the Random Forest test-result baseline.

    python scripts/train_rf.py

The pipeline steps (clean -> engineer -> encode -> fit) live in
medresearch.rf.pipeline so the Streamlit app and the tests apply the exact same
encoders and column order. Getting that wrong does not raise -- it silently
predicts on scrambled features -- which is why it is shared code and not
repeated here.
"""
import pandas as pd

from medresearch import config
from medresearch.rf import build_features, save_bundle, train_model

print(f"Reading {config.HEALTHCARE_CSV}")
df = pd.read_csv(config.require(
    config.HEALTHCARE_CSV,
    "Download it with: python tools/download_kaggle_dataset.py"))

X, y, encoders = build_features(df)
print(f"{len(X):,} rows | {len(X.columns)} features -> {list(X.columns)}")

bundle, report = train_model(X, y, encoders)
print("\n=== How accurate is the model? ===")
print(report)

save_bundle(bundle, config.RF_MODEL)
print(f"Model saved to {config.RF_MODEL}")
