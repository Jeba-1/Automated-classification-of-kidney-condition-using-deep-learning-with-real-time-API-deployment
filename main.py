
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# FastAPI Endpoint URL
API_URL = "https://automated-classification-of-kidney.onrender.com/predict/"

st.title("🔬 Automated Classification of Kidney Condition")

uploaded_files = st.file_uploader("📤 Upload a Kidney CT Scan Image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
    # Convert uploaded image to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
    
    # Display the uploaded image
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    
    # Convert image to bytes for API request
    for idx, img_array in enumerate(images):
        _, img_encoded = cv2.imencode('.jpg', img_array)
        files = [("files", ("image.jpg", img_encoded.tobytes(), "image/jpeg"))]  

        # "Classify" button without file name
        if st.button(f"🔍 Classify {idx+1}", key=f"classify_{idx}"):
            with st.spinner(f"⏳ Getting Predictions..."):
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()[0]  # Extract first item from the response list
                    
                    st.success(f"✅ **Predicted Condition:** {result['prediction']}")

                    # Expandable sections for details
                    with st.expander("📌 Description"):
                        st.info(result["description"])
                        
                    with st.expander("🩺 Symptoms"):
                        st.markdown("\n".join([f"- {symptom}" for symptom in result["symptoms"]]))

                    with st.expander("🔬 Diagnosis Methods"):
                        st.markdown("\n".join([f"- {diagnosis}" for diagnosis in result["diagnosis"]]))

                    with st.expander("💊 Treatment Options"):
                        st.markdown("\n".join([f"- {treatment}" for treatment in result["treatment"]]))

                else:
                    st.error("❌ Error in API request. Please try again.")



