import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voice_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train the model first.")

model = joblib.load(MODEL_PATH)

def predict(features):
    probs = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    confidence = float(np.max(probs))

    return prediction, confidence
