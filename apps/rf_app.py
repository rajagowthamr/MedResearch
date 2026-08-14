"""Streamlit UI for the Random Forest baseline.

    streamlit run apps/rf_app.py
"""
import streamlit as st

from medresearch import config
from medresearch.rf import load_bundle, make_row

st.title("🏥 Healthcare Test-Result Predictor")
st.write("Enter patient details and the model will predict the Test Result.")

if not config.RF_MODEL.exists():
    st.error(f"No model at {config.RF_MODEL}. Run `python scripts/train_rf.py` first.")
    st.stop()

bundle = load_bundle(config.RF_MODEL)

# One input per feature: a dropdown where training used a LabelEncoder (so the
# options are exactly the categories the model was fitted on, and an unseen
# value cannot be entered), a number box otherwise.
user_input = {}
for col in bundle.columns:
    if col in bundle.encoders:
        user_input[col] = st.selectbox(col, list(bundle.encoders[col].classes_))
    else:
        user_input[col] = st.number_input(col, value=30)

if st.button("Predict"):
    # make_row applies the training encoders and restores the training column
    # ORDER -- sklearn matches features positionally, so a reordered frame
    # predicts on scrambled inputs without raising.
    row = make_row(user_input, bundle)

    prediction = bundle.model.predict(row)[0]
    st.success(f"Predicted Test Result: **{prediction}**")

    st.write("### How confident is the model?")
    for cls, p in zip(bundle.model.classes_, bundle.model.predict_proba(row)[0]):
        st.write(f"- {cls}: {p*100:.1f}%")
        st.progress(float(p))
