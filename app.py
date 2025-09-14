import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set page title and description with emojis
st.set_page_config(page_title="🕵‍♂ Deepfake Classifier", page_icon="🖼", layout="centered")
st.title("🕵‍♂ Deepfake Image Classifier")
st.write("📤 Upload an image (jpg, jpeg, or png) to predict if it's *Real* or *Fake* 👀")

# Load the saved model with caching for better performance
@st.cache_resource
def load_model_cached():
    return load_model("my_model.keras")

model = load_model_cached()

# Function to preprocess and predict
def predict_image(img, model, img_size=(224, 224)):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prob = model.predict(img_array)[0][0]
        label = "Real ✅" if prob > 0.5 else "Fake ❌"
        confidence = prob if prob > 0.5 else 1 - prob
        return label, confidence
    except Exception as e:
        st.error(f"⚠ Error processing image: {str(e)}")
        return None, None

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    display_width = 300   
    
    col1, col2 = st.columns([1.2, 1])  
    
    with col1:
        st.image(img, caption="🖼 Uploaded Image", width=display_width)
    
    with col2:
        label, confidence = predict_image(img, model)
        if label is not None:
            bg_color = "#d4edda" if "Real" in label else "#f8d7da"
            text_color = "#155724" if "Real" in label else "#721c24"
            border_color = "#c3e6cb" if "Real" in label else "#f5c6cb"
            
            st.markdown(
                f"""
                <div style='border-radius: 15px; padding: 20px; margin-top: 30px;
                            text-align: left; font-size: 20px; font-weight: bold;
                            background-color: {bg_color};
                            color: {text_color};
                            border: 2px solid {border_color};
                            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);'>
                    🔎 Prediction: {label}<br><br>
                    <span style='font-size:16px; font-weight:normal;'>Confidence: {confidence:.2f} 🔹</span>
                </div>
                """,
                unsafe_allow_html=True
            )
