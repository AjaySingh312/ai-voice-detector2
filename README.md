AI Voice Detector API

A FastAPI-based machine learning API that detects whether an uploaded audio file is Human voice or AI-generated voice using MFCC feature extraction and a Random Forest classifier.

Features-

  Accepts audio input (MP3)
  Extracts MFCC acoustic features
  Uses Random Forest classifier
  Secured with API Key authentication
  Deployed using FastAPI
  Swagger UI support for testing

Requirements-

Python 3.10
pip
Git

Installation & Setup (Local Machine)
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/ai-voice-detector.git
cd ai-voice-detector

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv

Activate:
Windows-
venv\Scripts\activate

Mac/Linux-
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set API Key

Inside config.py:
API_KEY = "your-secret-key"

Or set environment variable:

Windows (PowerShell)
setx API_KEY "your-secret-key"

Linux/Mac
export API_KEY="your-secret-key"

5️⃣ Run the Server
uvicorn app.main:app --reload


Server will start at:

http://127.0.0.1:8000
