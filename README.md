# 🕵️‍♂️ Deepfake Image Detection using ResNet50

## 📌 Project Overview
This project aims to classify images as **Real** or **Fake (Deepfake/AI-generated)** using a **Convolutional Neural Network (CNN)** based on **ResNet50**.  
The model was trained on a custom dataset of real and fake images with **data augmentation** for better generalization,and a **Streamlit web app** was developed to allow users to upload an image and instantly receive predictions with confidence scores.  

---

🔗 **Live Demo (Streamlit App):** [Click Here](https://deepfake-image-classifier.streamlit.app/) 🚀  

---

## 🏗️ Model Architecture
- **Base Model**: ResNet50 (frozen pretrained layers)  
- **Pooling Layer**: Global Average Pooling  
- **Dense Layer**: 256 units, ReLU activation  
- **Dropout Layer**: 0.5 (to reduce overfitting)  
- **Output Layer**: 1 unit, Sigmoid activation  

**Total parameters:** ~24M  
**Trainable parameters:** ~0.5M  

---

## 📊 Model Performance (Validation/Test Set)
- **Accuracy**: **85%**  
- **Precision**: **0.85**  
- **Recall**: **0.84**  
- **F1-score**: **0.85**  

### 🔎 Per Class Results
| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Real ✅  | 0.86      | 0.88   | 0.87     |
| Fake ❌  | 0.83      | 0.80   | 0.82     |

### 🔎 Confusion Matrix
[[105 14]
[ 17 70]]


---

## 🖥️ Streamlit App
The Streamlit interface allows real-time prediction:  

1. Upload an image (`jpg`, `jpeg`, `png`)  
2. The model classifies it as **Real** ✅ or **Fake** ❌  
3. Displays a **confidence score**  

---

## 🔮 Future Improvements
- Fine-tune more layers of ResNet50 for better performance

- Try other architectures (EfficientNet, Vision Transformers)

- Use larger and more diverse datasets for better generalization

- Deploy as a web service (Flask/Django + Docker
