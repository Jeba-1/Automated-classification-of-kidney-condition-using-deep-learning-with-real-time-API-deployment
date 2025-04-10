# Automated-classification-of-kidney-condition-using-deep-learning-with-real-time-API-deployment
This project focuses on classifying kidney conditions—Normal, Cyst, Stone, and Tumor—using CT scan images and state-of-the-art deep learning techniques. Designed for real-time deployment, the system supports medical diagnostics by enabling automated and accurate disease detection through a web-based interface powered by Streamlit and a FastAPI backend.

🚀 Project Highlights
✅ Multi-class classification: Normal, Cyst, Stone, Tumor

🧠 Deep learning models (CNN, VGG16, Densenet201, InceptionV3, CNN-ViT Hybrid)

📊 Real-time prediction with a clean UI (Streamlit)

🔁 Feedback mechanism for continuous improvement

🧪 Evaluation using accuracy, confusion matrix, and more

📦 Deployed using FastAPI for robust backend API

# 📁 Dataset
Source: CT Kidney Dataset - Kaggle
# Classes:
1.Normal: 5,077 images

2.Cyst: 3,709 images

3.Tumor: 2,283 images

4.Stone: 1,377 images

Total: 12,446 CT scan images

# 🧪 Model Pipeline
Data Preprocessing

Image labeling & augmentation

Normalization & resizing

Train-validation-test split

Model Development

Basic CNN

Transfer Learning (VGG16, DenseNet201, InceptionV3)

Hybrid model (CNN + ViT) inspired by research literature

Training & Evaluation

Accuracy, loss tracking

Confusion matrix, precision, recall, F1

Comparison of model performance

# Deployment
Backend: FastAPI

Frontend: Streamlit Web App

Real-time CT image classification


# 🛠️ Technologies Used
Python 3.11

TensorFlow / Keras

PyTorch (for ViT)

OpenCV, NumPy, Matplotlib

FastAPI (backend REST API)

Streamlit (frontend web app)

Kaggle API, Google Colab (experiments & training)

# 🌐 Web App Features
Upload CT scan image

Predict kidney condition

View prediction confidence

Doctor/user feedback option

Model summary panel

# 📸 Sample Visualizations
Class distribution plot
![image](https://github.com/user-attachments/assets/4fa34507-78c1-460c-b20c-028e8f4a3a74)

Confusion matrices
![image](https://github.com/user-attachments/assets/6f4b93f1-01a4-41dc-8fea-369f203d5f86)
Training plot
![image](https://github.com/user-attachments/assets/94a4a6ff-beea-43ee-a2a4-ad17a9a1c62d)


# 📦 Installation & Usage
# Clone the repository
git clone https://github.com/yourusername/Automated-classification-of-kidney-condition-using-deep-learning-with-real-time-API-deployment.git
cd Automated-classification-of-kidney-condition...
# Install requirements
pip install -r requirements.txt
# Run Streamlit app
streamlit run app.py
# (Optional) Start API Server
uvicorn api.main:app --reload

# 📈 Results
![image](https://github.com/user-attachments/assets/3e1034f4-6616-4235-b9ed-7b73e01f2244)
The app : https://automated-classification-of-kidney-condition-jp.streamlit.app/



