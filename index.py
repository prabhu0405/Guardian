from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from mangum import Mangum  # ✅ required for Vercel
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI()

# Load models
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

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    temp_path = "temp.wav"
    with open(temp_path, "wb") as buffer:
        buffer.write(await audio.read())

    try:
        alert_level = get_alert_level(temp_path)
        return JSONResponse(content={"alert_level": alert_level})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ✅ Required for Vercel
handler = Mangum(app)
