from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from mangum import Mangum
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import tempfile

app = FastAPI()

    # title="Audio Alert Detection API", version="1.0.0"

# Load models (these will be loaded once when the function starts)
svm_model = joblib.load("svm_model.pkl")
mlp = load_model("mlp_model.h5")

def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=60)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

def get_alert_level(audio_path):
    features = extract_mfcc(audio_path)
    svm_pred = svm_model.predict(features)[0]
    mlp_pred = np.argmax(mlp.predict(features), axis=1)[0]
    
    if svm_pred == 1 and mlp_pred == 1:
        return "High Alert"
    elif svm_pred == 1 or mlp_pred == 1:
        return "Moderate Alert"
    else:
        return "Normal"

@app.get("/")
async def root():
    return {"message": "Audio Alert Detection API is running"}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    # Use temporary file with proper cleanup
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = temp_file.name
        temp_file.write(await audio.read())
    
    try:
        alert_level = get_alert_level(temp_path)
        return JSONResponse(content={
            "alert_level": alert_level,
            "filename": audio.filename,
            "status": "success"
        })
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "status": "error"}, 
            status_code=500
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Required for Vercel
handler = Mangum(app)