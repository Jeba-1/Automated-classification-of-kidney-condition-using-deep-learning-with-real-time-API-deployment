
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
    for idx, uploaded_file in enumerate(uploaded_files):
        # Convert uploaded image to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Display the uploaded image
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        # Convert image to bytes for API request
        _, img_encoded = cv2.imencode('.jpg', img_array)
        files = [("files", ("image.jpg", img_encoded.tobytes(), "image/jpeg"))]  

        # Store prediction results in session state to persist after button clicks
        if f"prediction_{idx}" not in st.session_state:
            st.session_state[f"prediction_{idx}"] = None

        if st.button(f"🔍 Classify {idx+1}", key=f"classify_{idx}"):
            with st.spinner(f"⏳ Getting Predictions..."):
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()[0]  # Extract first item from the response list
                    st.session_state[f"prediction_{idx}"] = result  # Store result in session state
                else:
                    st.error("❌ Error in API request. Please try again.")

        # If prediction exists, display it
        if st.session_state[f"prediction_{idx}"]:
            result = st.session_state[f"prediction_{idx}"]
            st.success(f"✅ **Predicted Condition:** {result['prediction']}")

            # Buttons for additional details
            if st.button("📌 Description", key=f"desc_{idx}"):
                st.info(result["description"])
            
            if st.button("🩺 Symptoms", key=f"symptoms_{idx}"):
                st.markdown("\n".join([f"- {symptom}" for symptom in result["symptoms"]]))

            if st.button("🔬 Diagnosis Methods", key=f"diag_{idx}"):
                st.markdown("\n".join([f"- {diagnosis}" for diagnosis in result["diagnosis"]]))

            if st.button("💊 Treatment Options", key=f"treat_{idx}"):
                st.markdown("\n".join([f"- {treatment}" for treatment in result["treatment"]]))



