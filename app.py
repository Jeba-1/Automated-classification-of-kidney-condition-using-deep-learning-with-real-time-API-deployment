
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import io
import os
import uvicorn
import logging

# ✅ Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Define Google Drive model link & local path
MODEL_URL = "https://drive.google.com/uc?id=1H4L5fIq0AOXZpy66dq0zxhql4rR3iccm"
MODEL_PATH = "Custom_CNN.tflite"

# ✅ Download model if not available
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Load TensorFlow Lite model
print("🧠 Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✅ Model loaded successfully!")

# ✅ Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Define class labels (ensure correct order)
CLASS_LABELS = ["Cyst", "Normal", "Stone", "Tumor"]

# ✅ Define class information (description, symptoms, etc.)
CLASS_INFO = {
    "Cyst": {
        "description": "Cystic kidney disease involves fluid-filled sacs in the kidney that may require monitoring or treatment.",
        "symptoms": ["Pain in the back or side", "High blood pressure", "Frequent urination", "Blood in urine"],
        "diagnosis": ["Ultrasound", "CT scan", "MRI", "Kidney function tests"],
        "treatment": ["Regular monitoring", "Medications", "Drainage procedures", "Surgery in severe cases"]
    },
    "Normal": {
        "description": "The kidney appears normal with no visible abnormalities.",
        "symptoms": ["No symptoms (healthy kidney function)"],
        "diagnosis": ["Routine medical checkup"],
        "treatment": ["Maintain a healthy lifestyle", "Drink plenty of water", "Regular medical checkups"]
    },
    "Stone": {
        "description": "Kidney stones are mineral deposits that may cause pain and require treatment.",
        "symptoms": ["Severe back pain", "Blood in urine", "Frequent urge to urinate", "Nausea and vomiting"],
        "diagnosis": ["CT scan", "X-ray", "Urine tests", "Ultrasound"],
        "treatment": ["Increased water intake", "Pain relievers", "Shock wave therapy", "Surgical removal"]
    },
    "Tumor": {
        "description": "A kidney tumor might indicate malignancy or benign growth. Further testing is needed to determine the severity.",
        "symptoms": ["Blood in urine", "Abdominal pain", "Unexplained weight loss", "Fatigue", "Fever"],
        "diagnosis": ["CT scan", "MRI", "Biopsy", "Blood tests"],
        "treatment": ["Surgical removal", "Targeted therapy", "Radiation therapy", "Regular follow-ups"]
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
    return img.astype(np.float32)  # Ensure correct dtype
    
# ✅ Home Route
# ✅ Home Route (supports GET and HEAD requests)
@app.get("/")
@app.head("/")  # ✅ Add this line to support HEAD requests
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
            img.verify()  # Check if it's a valid image
            img = Image.open(io.BytesIO(contents))  # Reload image after verification
            img_array = preprocess_image(img)

            # ✅ Ensure correct input shape for model
            img_array = img_array.reshape(input_details[0]['shape']).astype(np.float32)

            # ✅ Make prediction with TFLite model
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            # ✅ Ensure valid prediction output
            if prediction is None or len(prediction) == 0:
                raise HTTPException(status_code=500, detail="Model returned an empty prediction.")

            # ✅ Convert np.int64 to Python int before using it
            predicted_index = int(np.argmax(prediction))  # Fix int64 issue

            # ✅ Ensure predicted index is within range
            if predicted_index >= len(CLASS_LABELS):
                raise HTTPException(status_code=500, detail="Model returned an invalid class index.")

            # ✅ Convert index to class label
            predicted_class = CLASS_LABELS[predicted_index]

            # ✅ Get additional class info
            class_details = CLASS_INFO[predicted_class]

            # ✅ Append results (WITHOUT confidence score)
            results.append({
                "filename": filename,
                "prediction": predicted_class,
                "description": class_details["description"],
                "symptoms": class_details["symptoms"],
                "diagnosis": class_details["diagnosis"],
                "treatment": class_details["treatment"]
            })

        except Exception as e:
            logging.error(f"Error processing {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})

    return results

# ✅ Start the FastAPI server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)

    
