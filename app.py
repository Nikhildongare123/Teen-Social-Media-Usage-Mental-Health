import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Model + Encoders
# -------------------------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_gender = data["le_gender"]
le_platform = data["le_platform"]
le_social = data["le_social"]
le_target = data["le_target"]

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health Risk Prediction System")
st.write("Predict depression risk based on lifestyle and behavior")

# -------------------------------
# User Inputs
# -------------------------------
age = st.slider("Age", 13, 19)

gender = st.selectbox("Gender", ["Male", "Female"])

social_media = st.slider("Daily Social Media Hours", 0, 12)

platform = st.selectbox("Platform Usage", ["Instagram", "TikTok", "Both", "Other"])

sleep = st.slider("Sleep Hours", 0, 12)

screen_time = st.selectbox("Screen Time Before Sleep", ["Yes", "No"])

academic = st.slider("Academic Performance (GPA)", 2.0, 4.0)

physical = st.slider("Physical Activity (hours)", 0, 5)

social = st.selectbox("Social Interaction Level", ["Low", "Medium", "High"])

stress = st.slider("Stress Level (1-10)", 1, 10)

anxiety = st.slider("Anxiety Level (1-10)", 1, 10)

# -------------------------------
# Encode Inputs
# -------------------------------
gender = le_gender.transform([gender])[0]
platform = le_platform.transform([platform])[0]
social = le_social.transform([social])[0]

screen_time = 1 if screen_time == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):

    input_data = np.array([[age, gender, social_media, platform,
                            sleep, screen_time, academic,
                            physical, social, stress, anxiety]])

    prediction = model.predict(input_data)[0]
    result = le_target.inverse_transform([prediction])[0]

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("Prediction Result:")

    if result == "Low":
        st.success("🟢 Low Risk")
    elif result == "Medium":
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # -------------------------------
    # Suggestions
    # -------------------------------
    st.subheader("Suggestions:")

    if result == "High":
        st.write("- Reduce screen time 📵")
        st.write("- Improve sleep 🛌")
        st.write("- Do physical activity 🏃")
        st.write("- Talk to someone 🤝")

    elif result == "Medium":
        st.write("- Maintain healthy routine")
        st.write("- Manage stress")
        st.write("- Stay socially active")

    else:
        st.write("- Keep up the good lifestyle 👍")

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown("---")
st.caption("⚠️ This is not a medical diagnosis. For educational purposes only.")
