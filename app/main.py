from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io

from app.auth import verify_api_key
from app.config import SUPPORTED_LANGUAGES
from app.features import extract_features
from app.predictor import predict

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#here some middleware i have added
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def health():
    return {"status": "AI Voice Detection API Running"}


@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, api_key: str = Depends(verify_api_key)):

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_stream = io.BytesIO(audio_bytes)

        # Convert to mono 16kHz
        audio, sr = librosa.load(audio_stream, sr=16000, mono=True)

        features = extract_features(audio, sr)

        prediction, confidence = predict(features)

        classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

        explanation = generate_explanation(classification, confidence)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Malformed request")


def generate_explanation(classification, confidence):

    if classification == "AI_GENERATED":
        if confidence > 0.85:
            return "High spectral consistency and reduced pitch variation detected"
        else:
            return "Patterns resemble synthetic speech characteristics"
    else:
        if confidence > 0.85:
            return "Natural pitch variability and human speech dynamics detected"
        else:
            return "Acoustic features resemble human speech patterns"
