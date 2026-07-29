import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------------------------------------------------------------
# STEP 1: LOAD  -  read the CSV file into a table (DataFrame)
# ------------------------------------------------------------------
df = pd.read_csv("healthcare_dataset.csv")

# ------------------------------------------------------------------
# STEP 2: CLEAN  -  fix messy text and remove duplicate rows
# ------------------------------------------------------------------
df["Name"] = df["Name"].str.title()   # "bObbY jAcksOn" -> "Bobby Jackson"
df = df.drop_duplicates()

# ------------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING  -  create a useful new column.
# Raw dates aren't useful, but "how many days the patient stayed" is.
# ------------------------------------------------------------------
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# ------------------------------------------------------------------
# STEP 4: DEFINE INPUTS (X) AND TARGET (y)
#   y = what we want to predict (Test Results)
#   X = the info the model learns from (everything useful except the answer)
# ------------------------------------------------------------------
drop_cols = ["Name", "Doctor", "Hospital", "Date of Admission",
             "Discharge Date", "Room Number", "Test Results"]
X = df.drop(columns=drop_cols)
y = df["Test Results"]

# ------------------------------------------------------------------
# STEP 5: ENCODE  -  models need numbers, so turn text into numbers.
#   e.g. Gender: Male->1, Female->0
#   We KEEP each encoder in a dictionary so the UI can reuse them later.
# ------------------------------------------------------------------
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ------------------------------------------------------------------
# STEP 6: SPLIT  -  train on 80% of the data, test on the unseen 20%.
# This checks whether the model actually learned (not just memorised).
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------
# STEP 7: TRAIN  -  the model finds patterns between X and y.
# ------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=12, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# STEP 8: EVALUATE  -  see how well it predicts on the unseen test set.
# ------------------------------------------------------------------
preds = model.predict(X_test)
print("=== How accurate is the model? ===")
print(classification_report(y_test, preds))

# ------------------------------------------------------------------
# STEP 9: SAVE  -  store the trained model so you can reuse it later
# without retraining.
# ------------------------------------------------------------------
# We save a "bundle": the model + the encoders + the exact column order.
# The UI needs all three to make a correct prediction.
bundle = {
    "model": model,
    "encoders": encoders,
    "columns": list(X.columns),
}
joblib.dump(bundle, "health_model.pkl")
print("Model saved to health_model.pkl")
