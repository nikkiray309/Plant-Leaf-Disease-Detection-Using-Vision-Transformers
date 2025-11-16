import streamlit as st
from PIL import Image
import pickle
import torch
from torchvision import transforms
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import mahotas as mt
import warnings
import os
import tempfile
import logging
import joblib
import json


# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Streamlit page config
st.set_page_config(page_title="Plant Disease Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌿 Plant Leaf Disease Detection</h1>", unsafe_allow_html=True)

# Model selector
model_option = st.sidebar.selectbox("Select Model Type", ["Visual Transformer (ViT)", "SVM"])

@st.cache_resource
def load_disease_info_svm():
    try:
        with open("C:/Users/nikhi/Downloads/SDAI-GRP12/svm_disease_info.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load disease info: {str(e)}")
        return {}

disease_info_svm = load_disease_info_svm()

@st.cache_resource
def load_disease_info():
    try:
        with open("C:/Users/nikhi/Downloads/SDAI-GRP12/full_disease_info.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load disease information: {e}")
        return {}

disease_info = load_disease_info()

# Load ViT model
@st.cache_resource
def load_vit_model():
    try:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=38,
            ignore_mismatched_sizes=True
        )
        with open("C:/Users/nikhi/Downloads/SDAI-GRP12/vit_plantvillage.pkl", 'rb') as f:
            model.load_state_dict(pickle.load(f))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load ViT model: {str(e)}")
        raise

vit_model = load_vit_model()
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted([
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry(including_sour)__Powdery_mildew", "Cherry(including_sour)___healthy",
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)__Common_rust",
    "Corn_(maize)__Northern_Leaf_Blight", "Corn(maize)__healthy", "Grape__Black_rot",
    "Grape__Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange__Huanglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell_healthy", "Potato__Early_blight",
    "Potato__Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean__healthy",
    "Squash__Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato__healthy"
]))}

@st.cache_resource
def load_svm_components():
    try:
        with open('C:/Users/nikhi/Downloads/SDAI-GRP12/svm_model.pkl', "rb") as f:
            model = joblib.load(f)
        with open('C:/Users/nikhi/Downloads/SDAI-GRP12/scaler.pkl', "rb") as f:
            scaler = joblib.load(f)
        with open('C:/Users/nikhi/Downloads/SDAI-GRP12/lable_encoder.pkl', "rb") as f:
            label_encoder = joblib.load(f)
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Failed to load SVM components: {str(e)}")
        raise

def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from path: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
        red, green, blue = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        red[red == 255], green[green == 255], blue[blue == 255] = 0, 0, 0
        color_features = [np.mean(red), np.mean(green), np.mean(blue),
                          np.std(red), np.std(green), np.std(blue)]
        haralick = mt.features.haralick(gray)
        haralick_mean = haralick.mean(axis=0)
        texture_features = [haralick_mean[1], haralick_mean[2], haralick_mean[4], haralick_mean[8]]
        return color_features + texture_features, img_rgb
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

# Image upload
left_col, right_col = st.columns(2)
with left_col:
    st.markdown("### Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image")
    temp_file_path = None
    if uploaded_file is not None:
        original_filename = uploaded_file.name
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            st.error("Invalid file extension. Please upload a .jpg, .jpeg, or .png file.")
            st.stop()
        temp_ext = '.jpg' if file_ext in ['.jpg', '.jpeg'] else '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_ext) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        image = Image.open(temp_file_path)
    st.markdown("### ")
    predict = st.button("🔍 Predict Disease", use_container_width=True)

# Prediction and display
with right_col:
    if uploaded_file:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)
    if predict and uploaded_file:
        with st.spinner("Predicting disease..."):
            try:
                predicted_disease = None
                if model_option == "Visual Transformer (ViT)":
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    image_tensor = vit_transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = vit_model(image_tensor).logits
                        _, pred = torch.max(outputs, 1)
                    idx_to_class = {v: k for k, v in class_to_idx.items()}
                    predicted_disease = idx_to_class[pred.item()]
                elif model_option == "SVM":
                    svm_model, svm_scaler, svm_label_encoder = load_svm_components()
                    features, _ = extract_features(temp_file_path)
                    features_scaled = svm_scaler.transform([features])
                    prediction = svm_model.predict(features_scaled)
                    predicted_disease = svm_label_encoder.inverse_transform(prediction)[0]
                    if predicted_disease not in class_to_idx:
                        st.warning(f"SVM predicted class '{predicted_disease}' not in ViT class list. Please check label encoder compatibility.")
                if predicted_disease:
                    st.markdown(f"### 🧪 Predicted Disease: {predicted_disease}")
                else:
                    st.error("Prediction failed. Please try again.")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    elif predict and not uploaded_file:
        st.warning("Please upload an image before prediction.")


# Description, Suggestions & Precautions
st.markdown("---")
with st.container():
    st.subheader("📝 Description, Suggestions & Precautions")
    if predict and uploaded_file and 'predicted_disease' in locals() and predicted_disease:
        if model_option == "SVM":
            info = disease_info_svm.get(predicted_disease)
        else:
            info = disease_info.get(predicted_disease)
        if info:
            st.markdown(f"**Disease:** {info['disease']}")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown("**Suggestions:**")
            for s in info['suggestions']:
                st.markdown(f"- {s}")
            st.markdown("**Precautions:**")
            for p in info['precautions']:
                st.markdown(f"- {p}")
        else:
            st.info("No detailed information found for this disease.")
    else:
        st.info("Prediction details will appear here after analysis.")

