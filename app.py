import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Student Marks Prediction & Result",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for modern vertical card layout, colored boxes, and shadows
st.markdown("""
    <style>
    .main {
        background-color: #f1f5f9;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .pass-box {
        background-color: #d1fae5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 10px;
        color: #065f46;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .fail-box {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 10px;
        color: #991b1b;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
    }
    h1 {
        color: #1e293b;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model safely
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Title Section
st.title("🎓 Student Performance & Result Portal")
st.markdown("Enter individual subject marks below. **Total marks** are automatically calculated and passed to the model.")

# Main Form Card Container
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Enter Subject Marks (Out of 100)")

# 6 Subject Inputs
hindi = st.number_input("Hindi Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
english = st.number_input("English Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
science = st.number_input("Science Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
maths = st.number_input("Maths Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
history = st.number_input("History Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
geography = st.number_input("Geography Marks", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

# Automatically calculate total marks (the 7th feature required by the model)
total_marks = hindi + english + science + maths + history + geography
percentage = total_marks / 6.0

st.markdown("---")
st.markdown(f"### 📊 Calculated Total Marks: **{total_marks} / 600** &nbsp;&nbsp;|&nbsp;&nbsp; Percentage: **{percentage:.2f}%**")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Button & Dynamic Pass/Fail Layout
if st.button("🚀 Predict Result"):
    # The model expects exactly 7 features: Hindi, English, Science, Maths, History, Geography, Total[cite: 1]
    input_data = np.array([[hindi, english, science, maths, history, geography, total_marks]])
    
    try:
        with st.spinner("Analyzing performance..."):
            prediction = model.predict(input_data)
            predicted_label = str(prediction[0]).strip().lower()
            
            # Trigger celebration effects
            st.balloons()
            
        # Determine if result is pass or fail based on model output and percentage check
        is_pass = "fail" not in predicted_label and percentage >= 40.0
        
        # Display Result with Custom Colored Box Layout and Shadow
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if is_pass:
            st.markdown(f"""
                <div class="pass-box">
                    <h2>🎉 Congratulations! Status: PASS</h2>
                    <p><b>Model Category Output:</b> {prediction[0]}</p>
                    <p>Your performance meets the passing criteria successfully with <b>{percentage:.2f}%</b>.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="fail-box">
                    <h2>⚠️ Status: FAIL</h2>
                    <p><b>Model Category Output:</b> {prediction[0]}</p>
                    <p>Your overall performance is below the passing threshold. Keep working hard!</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
