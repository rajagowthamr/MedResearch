import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load the saved bundle (model + encoders + column order)
# ------------------------------------------------------------------
bundle = joblib.load("health_model.pkl")
model = bundle["model"]
encoders = bundle["encoders"]
columns = bundle["columns"]

st.title("🏥 Healthcare Test-Result Predictor")
st.write("Enter patient details and the model will predict the Test Result.")

# ------------------------------------------------------------------
# Build one input box per feature.
#   - If the column was text (has an encoder) -> show a dropdown
#   - If it was a number -> show a number box
# ------------------------------------------------------------------
user_input = {}
for col in columns:
    if col in encoders:
        # text column -> dropdown of the original text options
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(col, options)
    else:
        # numeric column -> number input
        user_input[col] = st.number_input(col, value=30)

# ------------------------------------------------------------------
# When the button is clicked, make a prediction
# ------------------------------------------------------------------
if st.button("Predict"):
    # 1. put the inputs into a one-row table
    row = pd.DataFrame([user_input])

    # 2. encode the text columns the SAME way as during training
    for col in encoders:
        row[col] = encoders[col].transform(row[col])

    # 3. keep the exact column order the model expects
    row = row[columns]

    # 4. predict
    prediction = model.predict(row)[0]
    st.success(f"Predicted Test Result: **{prediction}**")

    # 5. show HOW SURE the model is for each class (the vote share)
    probs = model.predict_proba(row)[0]
    st.write("### How confident is the model?")
    for cls, p in zip(model.classes_, probs):
        st.write(f"- {cls}: {p*100:.1f}%")
        st.progress(float(p))
