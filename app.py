import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="ML Model Predictor",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for modern vertical layout, colors, and shadows
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.6);
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

# Determine number of features expected by the model to prevent mismatch errors
n_features = getattr(model, "n_features_in_", 5)
feature_names = getattr(model, "feature_names_in_", [f"Feature {i+1}" for i in range(n_features)])

# Title Section
st.title("🚀 Model Prediction Portal")
st.markdown("Provide the required inputs below to generate a prediction from your trained classifier.")

# Vertical Form Container with Shadow Effect
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Input Parameters")

input_values = []
# Create dynamic vertical inputs based on what the model expects
for i in range(n_features):
    name = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
    val = st.number_input(f"Enter value for **{name}**", value=0.0, format="%.2f")
    input_values.append(val)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Button & Logic
if st.button("✨ Run Prediction"):
    # Reshape input to match model expectation (1 sample, n features)
    input_data = np.array([input_values])
    
    try:
        with st.spinner("Analyzing data..."):
            prediction = model.predict(input_data)
            
            # Trigger celebration effect
            st.balloons()
            
        # Display Result Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success(f"### 🎉 Predicted Result: **{prediction[0]}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
