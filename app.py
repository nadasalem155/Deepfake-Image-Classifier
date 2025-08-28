# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Deepfake Detector 🎯",
    page_icon="🕵",
    layout="centered"
)

st.title("🔍 Deepfake Detector")
st.write("Upload an image and let the model decide if it's *Real* or *Fake*!")

# -----------------------------
# Load model and class indices
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("my_model.keras")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Reverse mapping (0 -> Fake, 1 -> Real)
    class_labels = {v: k for k, v in class_indices.items()}
    return model, class_labels

model, class_labels = load_resources()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # Ensure RGB
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Preprocess image
    # -----------------------------
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # -----------------------------
    # Make prediction
    # -----------------------------
    prob_real = float(model.predict(img_array)[0][0])  # probability of being Real (label=1)
    prob_fake = 1 - prob_real                           # probability of being Fake (label=0)

    if prob_real >= 0.5:
        label = "Real ✅"
        emoji = "😇"
        confidence = prob_real * 100
    else:
        label = "Fake ❌"
        emoji = "👹"
        confidence = prob_fake * 100

    # -----------------------------
    # Display result
    # -----------------------------
    st.markdown("---")
    st.markdown(f"## {label} {emoji}")
    st.markdown(f"*Real Probability:* {prob_real*100:.2f}%")
    st.markdown(f"*Fake Probability:* {prob_fake*100:.2f}%")
    st.markdown(f"*Confidence in Prediction:* {confidence:.2f}%")