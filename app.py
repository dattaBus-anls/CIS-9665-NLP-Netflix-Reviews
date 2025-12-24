import streamlit as st
import joblib

# ------------------------------------------------------------
# Load trained NLP model (TF-IDF + Logistic Regression pipeline)
# ------------------------------------------------------------
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Netflix Review Rating Predictor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Netflix Review Rating Predictor")
st.write("Type a Netflix app review and get a predicted star rating (1–5 ⭐).")

review_text = st.text_area(
    "Enter your review text:",
    placeholder="Example: The app keeps crashing after the update. Very frustrating...",
    height=180
)

if st.button("Predict Rating ⭐", use_container_width=True):
    if not review_text.strip():
        st.warning("Please type a review first.")
    else:
        prediction = model.predict([review_text])[0]
        st.success(f"⭐ Predicted Rating: {prediction} / 5")
