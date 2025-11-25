# Plant-Leaf-Disease-Detection-Using-Vision-Transformers

# Plant Leaf Disease Detection Using Vision Transformers

## 📘 Project Overview

This project implements a plant leaf disease detection system leveraging **Vision Transformers (ViT)**. The goal is to classify and identify diseases on plant leaves from images, enabling early diagnosis and more effective disease management in agriculture.

The repository includes:

- A trained (or fine-tuned) Vision Transformer model for leaf disease classification  
- A simple application (via `app.py`) for inference on uploaded images  
- Supporting files such as label encoders, scalers, and a disease info JSON  

---

## 🧪 Features

- **Vision Transformer-based Model**: Uses ViT to capture global context in leaf images, which may improve accuracy over conventional CNNs.  
- **Scalable Inference App**: A Python application (`app.py`) to run predictions on new leaf images.  
- **Disease Metadata**: Includes `full_disease_info.json` for detailed disease descriptions.  
- **Model Serialization**: Uses pickled SVM model (`svm_model.pkl`), label encoder, scaler, etc., allowing reproducible inference without retraining. The SVM model was used as a baseline comparision against Vision Transformers.

---

## 📁 Project Structure

Here’s a breakdown of the key files:├── app.py
├── full_disease_info.json
├── label_encoder.pkl
├── scaler.pkl
├── svm_model.pkl
└── README.md


- `app.py`: Main application script for performing predictions.  
- `full_disease_info.json`: Metadata about different diseases (names, symptoms, description etc.).  
- `label_encoder.pkl`: Encodes disease labels from numeric to string.  
- `scaler.pkl`: Data scaler used in model preprocessing.  
- `svm_model.pkl`: The SVM trained model acting as a baseline for comparision.

---

## 🧰 Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/nikkiray309/Plant-Leaf-Disease-Detection-Using-Vision-Transformers.git
   cd Plant-Leaf-Disease-Detection-Using-Vision-Transformers


