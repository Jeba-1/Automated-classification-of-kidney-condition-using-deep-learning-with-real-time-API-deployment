import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# FastAPI Endpoint URL (Change this to your Render API URL)
API_URL = "https://automated-classification-of-kidney.onrender.com/docs#/default/predict_predict__post"

st.title("Kidney Condition Classification")

uploaded_file = st.file_uploader("Upload a Kidney CT Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to bytes for API request
    _, img_encoded = cv2.imencode('.jpg', img_array)
    files = {"file": img_encoded.tobytes()}
    
    if st.button("Classify"):
        with st.spinner("Getting Predictions..."):
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Condition: {result['class']}")
            else:
                st.error("Error in API request. Please try again.")
