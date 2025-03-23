
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import io
import os
import uvicorn
import nest_asyncio
from google.colab import output
import logging

# ✅ Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Fix runtime issues in Colab
nest_asyncio.apply()

# ✅ Define Google Drive model link & local path
MODEL_URL = "https://drive.google.com/uc?id=1c9Rsky1DmCUsHO-rTRfzejvZQ5nV8Ukh"
MODEL_PATH = "Custom_CNN (1).h5"

# ✅ Download model if not available
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Load the trained model
print("🧠 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ✅ Define class labels and details
CLASS_INFO = {
    "Cyst": {
        "description": "Cystic kidney disease involves fluid-filled sacs in the kidney that may require monitoring or treatment.",
        "symptoms": [
            "Pain in the back or side",
            "High blood pressure",
            "Frequent urination",
            "Blood in urine"
        ],
        "diagnosis": [
            "Ultrasound",
            "CT scan",
            "MRI",
            "Kidney function tests"
        ],
        "treatment": [
            "Regular monitoring with imaging tests",
            "Medications for pain relief and blood pressure control",
            "Drainage procedures for large cysts",
            "Surgery in severe cases"
        ]
    },
    "Normal": {
        "description": "The kidney appears normal with no visible abnormalities.",
        "symptoms": ["No symptoms (healthy kidney function)"],
        "diagnosis": ["Routine medical checkup"],
        "treatment": ["Maintain a healthy lifestyle", "Drink plenty of water", "Regular medical checkups"]
    },
    "Stone": {
        "description": "Kidney stones are mineral deposits that may cause pain and require treatment.",
        "symptoms": [
            "Severe lower back or abdominal pain",
            "Blood in urine",
            "Frequent urge to urinate",
            "Nausea and vomiting"
        ],
        "diagnosis": [
            "CT scan",
            "X-ray",
            "Urine tests",
            "Ultrasound"
        ],
        "treatment": [
            "Increased water intake to help flush out small stones",
            "Pain relievers",
            "Medications to break down or pass stones",
            "Shock wave therapy (ESWL) for larger stones",
            "Surgical removal in severe cases"
        ]
    },
    "Tumor": {
        "description": "A kidney tumor might indicate malignancy or benign growth. Further testing is needed to determine the severity.",
        "symptoms": [
            "Blood in urine",
            "Abdominal pain",
            "Unexplained weight loss",
            "Fatigue",
            "Fever"
        ],
        "diagnosis": [
            "CT scan",
            "MRI",
            "Biopsy",
            "Blood tests"
        ],
        "treatment": [
            "Surgical removal (nephrectomy for malignant tumors)",
            "Targeted therapy or immunotherapy for cancerous tumors",
            "Radiation therapy in some cases",
            "Regular follow-up imaging"
        ]
    }
}

# ✅ Allowed file types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ✅ Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# ✅ Home Route
@app.get("/")
def home():
    return {"message": "Kidney Condition Classification API is running!"}

# ✅ Multiple Image Prediction Endpoint
@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            logging.info(f"📂 Received file: {file.filename}, Content-Type: {file.content_type}")

            # ✅ Validate file extension
            filename = file.filename.lower()
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                raise HTTPException(status_code=400, detail=f"Invalid file type for {filename}. Please upload JPG, JPEG, or PNG.")

            # ✅ Read and process image
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail=f"Uploaded file {filename} is empty.")

            # ✅ Try opening the image
            img = Image.open(io.BytesIO(contents))
            img.verify()  # Checks if it's a valid image
            img = Image.open(io.BytesIO(contents))  # Reload image after verification
            img_array = preprocess_image(img)

            # ✅ Make prediction
            prediction = model.predict(img_array)
            predicted_class = max(CLASS_INFO.keys(), key=lambda c: prediction[0][list(CLASS_INFO.keys()).index(c)])
            confidence = np.max(prediction) * 100  # Convert to percentage

            # ✅ Append results
            results.append({
                "filename": filename,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "description": CLASS_INFO[predicted_class]["description"],
                "symptoms": CLASS_INFO[predicted_class]["symptoms"],
                "diagnosis": CLASS_INFO[predicted_class]["diagnosis"],
                "treatment": CLASS_INFO[predicted_class]["treatment"]
            })

        except Exception as e:
            logging.error(f"Error processing {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})

    return results

# ✅ Start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
    