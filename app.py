import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Model Prediction Dashboard",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for an attractive layout
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
        color: white;
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

# Header Section
st.title("⚡ Machine Learning Predictor")
st.markdown("Welcome! Use the sidebar controls to input your feature values and instantly get predictions from your trained KNN model.")

# Sidebar Inputs
st.sidebar.header("🎛️ Input Features")
st.sidebar.markdown("Adjust the values below:")

# Assuming 5 features based on your model configuration
feature_1 = st.sidebar.slider("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature_2 = st.sidebar.slider("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature_3 = st.sidebar.slider("Feature 3", min_value=0.0, max_value=100.0, value=50.0)
feature_4 = st.sidebar.slider("Feature 4", min_value=0.0, max_value=100.0, value=50.0)
feature_5 = st.sidebar.slider("Feature 5", min_value=0.0, max_value=100.0, value=50.0)

# Layout Columns for Information
col1, col2 = st.columns(2)

with col1:
    st.info("### 📌 Model Overview\n* **Algorithm:** K-Nearest Neighbors\n* **Target Classes:** Hindi, English, Science, Maths, History, Geography, Total")

with col2:
    st.warning("### 💡 Instructions\n1. Modify features on the left.\n2. Click **Run Prediction**.\n3. View your result below!")

st.markdown("---")

# Prediction Trigger Button
if st.button("🚀 Run Prediction"):
    # Combine features into an array format expected by scikit-learn
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])
    
    with st.spinner("Processing prediction..."):
        # Predict using the loaded model
        prediction = model.predict(input_data)
        
        # Trigger visual effect
        st.balloons()
        
    # Display Result
    st.success(f"🎉 **Predicted Category:** `{prediction[0]}`")
