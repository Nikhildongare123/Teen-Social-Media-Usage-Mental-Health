import streamlit as st
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Model with Error Handling
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model (2).pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'model (2).pkl' not found. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# -------------------------------
# App Title and Description
# -------------------------------
st.title("🧠 Mental Health Risk Prediction Tool")
st.markdown("""
This tool helps assess potential mental health risks based on lifestyle factors and habits.
**Note:** This is for educational purposes only and not a substitute for professional medical advice.
""")

# -------------------------------
# Sidebar Information
# -------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This prediction model considers:
    - Demographics
    - Social media usage
    - Sleep patterns
    - Academic performance
    - Physical activity
    - Stress and anxiety levels
    
    Fill in all fields for an accurate assessment.
    """)
    
    st.header("📊 Risk Levels")
    st.markdown("""
    - 🟢 **Low Risk**: Healthy patterns
    - 🟡 **Medium Risk**: Some concerns present
    - 🔴 **High Risk**: Multiple factors indicate need for attention
    """)

# -------------------------------
# Main Form
# -------------------------------
with st.form("prediction_form"):
    st.subheader("📝 Personal Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider(
            "Age", 
            min_value=13, 
            max_value=19, 
            value=16,
            help="Age range: 13-19 years"
        )
        
        gender = st.radio(
            "Gender",
            options=["Male", "Female"],
            horizontal=True
        )
        
        social_media = st.slider(
            "📱 Daily Social Media Hours",
            min_value=0,
            max_value=12,
            value=4,
            help="Hours spent on social media per day"
        )
    
    with col2:
        platform = st.selectbox(
            "🎮 Primary Platform",
            options=["Instagram", "TikTok", "Both", "Other"],
            help="Main social media platform used"
        )
        
        sleep = st.slider(
            "😴 Sleep Hours",
            min_value=0,
            max_value=12,
            value=7,
            help="Average hours of sleep per night"
        )
        
        screen_time = st.selectbox(
            "📱 Screen Before Sleep",
            options=["Yes", "No"],
            help="Do you use screens right before sleeping?"
        )
    
    with col3:
        academic = st.slider(
            "📚 Academic Performance (GPA)",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.1,
            format="%.1f",
            help="Current Grade Point Average (2.0-4.0)"
        )
        
        physical = st.slider(
            "🏃 Physical Activity (hours/week)",
            min_value=0,
            max_value=5,
            value=2,
            help="Hours of physical activity per week"
        )
        
        social = st.selectbox(
            "👥 Social Interaction Level",
            options=["Low", "Medium", "High"],
            help="Frequency of in-person social interactions"
        )
    
    # Separate section for stress and anxiety
    st.subheader("🎯 Mental Well-being Indicators")
    
    col4, col5 = st.columns(2)
    
    with col4:
        stress = st.slider(
            "😰 Stress Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Very low stress, 10 = Extremely high stress"
        )
        
        # Add stress level description
        if stress <= 3:
            st.caption("✅ Low stress level")
        elif stress <= 7:
            st.caption("⚠️ Moderate stress level")
        else:
            st.caption("🔴 High stress level - consider stress management techniques")
    
    with col5:
        anxiety = st.slider(
            "😥 Anxiety Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Very low anxiety, 10 = Extremely high anxiety"
        )
        
        # Add anxiety level description
        if anxiety <= 3:
            st.caption("✅ Low anxiety level")
        elif anxiety <= 7:
            st.caption("⚠️ Moderate anxiety level")
        else:
            st.caption("🔴 High anxiety level - consider speaking with a professional")
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Mental Health Risk", type="primary", use_container_width=True)

# -------------------------------
# Manual Encoding (IMPORTANT)
# -------------------------------
gender_map = {"Male": 1, "Female": 0}
platform_map = {"Instagram": 0, "TikTok": 1, "Both": 2, "Other": 3}
social_map = {"Low": 0, "Medium": 1, "High": 2}

# -------------------------------
# Prediction Logic
# -------------------------------
if submitted and model is not None:
    # Encode categorical variables
    gender_encoded = gender_map[gender]
    platform_encoded = platform_map[platform]
    social_encoded = social_map[social]
    screen_time_encoded = 1 if screen_time == "Yes" else 0
    
    # Create input array
    input_data = np.array([[age, gender_encoded, social_media, platform_encoded,
                            sleep, screen_time_encoded, academic,
                            physical, social_encoded, stress, anxiety]])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Display results in a nice container
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create columns for results display
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            # Show risk level with appropriate styling
            if prediction == 0:
                st.success("### 🟢 LOW RISK")
                st.markdown("""
                **Your lifestyle patterns appear healthy.**
                
                ✅ Maintain regular sleep schedule
                ✅ Continue physical activity
                ✅ Keep healthy social connections
                """)
                
            elif prediction == 1:
                st.warning("### 🟡 MEDIUM RISK")
                st.markdown("""
                **Some concerns were detected in your lifestyle patterns.**
                
                💡 Consider these improvements:
                - Monitor your social media usage
                - Improve sleep quality and duration
                - Increase physical activity
                - Practice stress management techniques
                - Talk to friends or family about your feelings
                """)
                
            else:
                st.error("### 🔴 HIGH RISK")
                st.markdown("""
                **Your responses indicate potential mental health concerns.**
                
                🚨 We strongly recommend:
                - Speaking with a school counselor or mental health professional
                - Contacting a mental health helpline
                - Sharing your concerns with a trusted adult
                - Prioritizing self-care and rest
                
                **Immediate Resources:**
                - National Suicide Prevention Lifeline: 988
                - Crisis Text Line: Text HOME to 741741
                """)
        
        # Show additional insights
        with st.expander("📈 View Detailed Analysis"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Daily Social Media", f"{social_media} hours", 
                         delta="High" if social_media > 5 else "Moderate" if social_media > 2 else "Low")
                st.metric("Sleep Duration", f"{sleep} hours",
                         delta="Good" if sleep >= 7 else "Low" if sleep < 6 else "Fair")
                st.metric("Physical Activity", f"{physical} hours/week",
                         delta="Good" if physical >= 3 else "Low" if physical < 1 else "Fair")
            
            with col_b:
                st.metric("Stress Level", f"{stress}/10",
                         delta="High" if stress > 7 else "Moderate" if stress > 4 else "Low")
                st.metric("Anxiety Level", f"{anxiety}/10",
                         delta="High" if anxiety > 7 else "Moderate" if anxiety > 4 else "Low")
                st.metric("Academic Performance", f"{academic}/4.0",
                         delta="Good" if academic >= 3.0 else "Needs Improvement")
        
        # Disclaimer
        st.info("💡 **Remember:** This is a screening tool only. For proper diagnosis and treatment, please consult a qualified mental health professional.")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

elif submitted and model is None:
    st.error("❌ Cannot make prediction. Model not loaded properly.")

# -------------------------------
# Helpful Tips Section
# -------------------------------
with st.expander("💡 Mental Health Tips & Resources"):
    st.markdown("""
    ### Daily Wellness Tips:
    - 🛌 Maintain consistent sleep schedule (7-9 hours)
    - 📱 Take regular breaks from social media
    - 🏃 Exercise for at least 30 minutes daily
    - 🥗 Eat balanced, nutritious meals
    - 💧 Stay hydrated
    - 🧘 Practice mindfulness or meditation
    
    ### When to Seek Help:
    - Feeling sad or withdrawn for more than 2 weeks
    - Severe mood swings affecting daily life
    - Changes in eating or sleeping patterns
    - Difficulty concentrating
    - Withdrawing from social activities
    - Thoughts of self-harm
    
    ### Professional Resources:
    - School counselor
    - Primary care physician
    - Licensed therapist or psychologist
    - Mental health hotlines
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f2:
    st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only and not a substitute for professional medical advice, diagnosis, or treatment.")
    st.caption(f"📍 Last updated: {datetime.now().strftime('%Y-%m-%d')}")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
