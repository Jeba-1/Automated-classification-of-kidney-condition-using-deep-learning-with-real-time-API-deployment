import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# FastAPI Endpoint URL
API_URL = "https://automated-classification-of-kidney.onrender.com/predict/"

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
    files = [("files", ("image.jpg", img_encoded.tobytes(), "image/jpeg"))]  # <-- FIXED

    if st.button("Classify"):
        with st.spinner("Getting Predictions..."):
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()[0]  # Extract first item from the response list
                
                st.success(f"Predicted Condition: {result['prediction']}")
                st.write(f"**Description:** {result['description']}")
                
                st.subheader("Symptoms")
                st.write(", ".join(result["symptoms"]))
                
                st.subheader("Diagnosis Methods")
                st.write(", ".join(result["diagnosis"]))
                
                st.subheader("Treatment Options")
                st.write(", ".join(result["treatment"]))
            
            else:
                st.error("Error in API request. Please try again.")
