import logging
import shutil
import os
import numpy as np
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
audio = FastAPI()

# CORS Middleware
audio.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# MODEL YÜKLEME
MODEL_DIR = "models/deepfake-audio"
hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
hf_model.eval()

# ENDPOINT
@audio.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), selected_model: str = Form(...)):
    logger.info(f"Received file: {file.filename}, selected_model: {selected_model}")
    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        if selected_model != "huggingface":
            raise HTTPException(status_code=400, detail="Only 'huggingface' model is available.")

        waveform, sample_rate = torchaudio.load(temp_file_path)

        # stereo ise mono'ya çevir
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # yeniden örnekle
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Tensor yerine numpy ver (shape: [time])
        input_np = waveform.squeeze().numpy()

        # feature extractor (zaten [1, time] haline getirir)
        inputs = hf_feature_extractor(input_np, sampling_rate=sample_rate, return_tensors="pt")

        with torch.no_grad():
            outputs = hf_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        return JSONResponse(content={
            "model": "huggingface",
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} deleted.")
