import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Model
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Mental Health Predictor")
st.title("🧠 Mental Health Risk Prediction")

# -------------------------------
# Inputs
# -------------------------------
age = st.slider("Age", 13, 19)
gender = st.selectbox("Gender", ["Male", "Female"])
social_media = st.slider("Daily Social Media Hours", 0, 12)
platform = st.selectbox("Platform", ["Instagram", "TikTok", "Both", "Other"])
sleep = st.slider("Sleep Hours", 0, 12)
screen_time = st.selectbox("Screen Before Sleep", ["Yes", "No"])
academic = st.slider("Academic Performance", 2.0, 4.0)
physical = st.slider("Physical Activity", 0, 5)
social = st.selectbox("Social Interaction", ["Low", "Medium", "High"])
stress = st.slider("Stress Level", 1, 10)
anxiety = st.slider("Anxiety Level", 1, 10)

# -------------------------------
# Manual Encoding (IMPORTANT)
# -------------------------------
gender_map = {"Male": 1, "Female": 0}
platform_map = {"Instagram": 0, "TikTok": 1, "Both": 2, "Other": 3}
social_map = {"Low": 0, "Medium": 1, "High": 2}

gender = gender_map[gender]
platform = platform_map[platform]
social = social_map[social]
screen_time = 1 if screen_time == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[age, gender, social_media, platform,
                            sleep, screen_time, academic,
                            physical, social, stress, anxiety]])

    prediction = model.predict(input_data)[0]

    st.subheader("Result:")

    # If model gives numbers
    if prediction == 0:
        st.success("🟢 Low Risk")
    elif prediction == 1:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚠️ This is not a medical diagnosis. For educational use only.")
