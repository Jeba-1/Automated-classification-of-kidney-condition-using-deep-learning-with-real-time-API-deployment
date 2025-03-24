
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import io

# FastAPI Endpoint URL
API_URL = "https://automated-classification-of-kidney.onrender.com/predict/"

st.title("🔬 Automated Classification of Kidney Condition")

uploaded_files = st.file_uploader("📤 Upload a Kidney CT Scan Image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

class PDF(FPDF):
    def header(self):
        self.set_font("Times", style='B', size=16)
        self.cell(200, 10, "Kidney Condition Classification Report", ln=True, align='C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", size=10)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')
    
    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        self.rect(5.0, 5.0, 200.0, 287.0)  # Border for all pages
    
    def add_section(self, title, content):
        self.set_font("Times", style='B', size=12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)
        self.set_font("Times", size=12)
        self.multi_cell(0, 10, content)
        self.ln(2)
        self.cell(0, 0, "", border='B')  # Horizontal line
        self.ln(5)

def generate_pdf(result, image):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Save and add image to PDF after title
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    pdf.image(image_path, x=40, y=30, w=130, h=100)
    pdf.ln(110)
    
    pdf.add_section("Predicted Condition:", result['prediction'])
    pdf.add_section("Description:", result['description'])
    pdf.add_section("Symptoms:", "\n".join([f"- {symptom}" for symptom in result["symptoms"]]))
    pdf.add_section("Diagnosis Measures:", "\n".join([f"- {diagnosis}" for diagnosis in result["diagnosis"]]))
    pdf.add_section("Treatment Options:", "\n".join([f"- {treatment}" for treatment in result["treatment"]]))
    
    return pdf

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

            if st.button("🔬 Diagnosis Measures", key=f"diag_{idx}"):
                st.markdown("\n".join([f"- {diagnosis}" for diagnosis in result["diagnosis"]]))

            if st.button("💊 Treatment Options", key=f"treat_{idx}"):
                st.markdown("\n".join([f"- {treatment}" for treatment in result["treatment"]]))
            
            # Download Report Button
            if st.button("📥 Download Report", key=f"download_{idx}"):
                pdf = generate_pdf(result, image)
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="📄 Download Prediction Report as PDF",
                    data=pdf_output,
                    file_name="Kidney_Condition_Report.pdf",
                    mime="application/pdf"
                )

