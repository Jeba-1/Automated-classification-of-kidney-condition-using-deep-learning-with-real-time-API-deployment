import streamlit as st
import requests
import io
from PIL import Image

# Title
st.title("Kidney Condition Classification")

# File uploader
uploaded_file = st.file_uploader("Upload a kidney CT scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    if st.button("Predict"):
        files = {"files": ("image.png", img_bytes, "image/png")}
        response = requests.post("https://your-fastapi-url.onrender.com/predict/", files=files)  # Replace with your API URL
        
        if response.status_code == 200:
            st.success(f"Prediction: {response.json()}")
        else:
            st.error("Prediction failed. Please try again.")
