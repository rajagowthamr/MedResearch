import joblib
import pandas as pd

print("Running tests...\n")

# ------------------------------------------------------------------
# TEST 1: Does the saved model file load correctly?
# ------------------------------------------------------------------
bundle = joblib.load("health_model.pkl")
assert "model" in bundle,    "FAIL: 'model' missing from bundle"
assert "encoders" in bundle, "FAIL: 'encoders' missing from bundle"
assert "columns" in bundle,  "FAIL: 'columns' missing from bundle"
print("TEST 1 PASSED: model file loads and has model + encoders + columns")

model = bundle["model"]
encoders = bundle["encoders"]
columns = bundle["columns"]


# helper: turn a patient dict into a ready-to-predict row
def make_row(patient):
    row = pd.DataFrame([patient])
    for col, le in encoders.items():
        row[col] = le.transform(row[col])
    return row[columns]


# a sample patient
patient = {
    "Age": 45, "Gender": "Male", "Blood Type": "A+",
    "Medical Condition": "Diabetes", "Insurance Provider": "Aetna",
    "Billing Amount": 25000, "Admission Type": "Urgent",
    "Medication": "Aspirin", "Length of Stay": 10,
}

# ------------------------------------------------------------------
# TEST 2: Can the model make a prediction without crashing?
# ------------------------------------------------------------------
pred = model.predict(make_row(patient))[0]
print("TEST 2 PASSED: model predicted ->", pred)

# ------------------------------------------------------------------
# TEST 3: Is the prediction one of the valid classes?
# ------------------------------------------------------------------
valid = ["Normal", "Abnormal", "Inconclusive"]
assert pred in valid, f"FAIL: '{pred}' is not a valid class"
print("TEST 3 PASSED: prediction is a valid class")

# ------------------------------------------------------------------
# TEST 4: Is the SAME input always giving the SAME output? (consistency)
# ------------------------------------------------------------------
pred_again = model.predict(make_row(patient))[0]
assert pred == pred_again, "FAIL: same input gave different outputs"
print("TEST 4 PASSED: same input gives same output (consistent)")

# ------------------------------------------------------------------
# TEST 5: Do the prediction probabilities add up to ~100%?
# ------------------------------------------------------------------
probs = model.predict_proba(make_row(patient))[0]
assert abs(sum(probs) - 1.0) < 0.001, "FAIL: probabilities don't sum to 1"
print("TEST 5 PASSED: probabilities sum to 100%")
print("   probabilities:", dict(zip(model.classes_, probs.round(2))))

print("\nAll tests passed. The model is working correctly.")
