import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
import numpy as np
import torch
import tensorflow as tf
import cv2
import dlib
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Test için herkese izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Sabitler ====================
IMG_SIZE_VIDEO_1 = (299, 299)
IMG_SIZE_VIDEO_2 = (128, 128)
IMG_SIZE_VIDEO_3 = (128, 128)
IMG_SIZE_PHOTO = (256, 256)
FRAME_SKIP = 2
NO_OF_FRAMES = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = dlib.get_frontal_face_detector()

# ==================== Model Sınıfı (PyTorch) ====================
class MyModel(nn.Module):
    """
    InceptionResnetV1'i bir sınıf içerisinde wrap eden örnek model.
    """
    def __init__(self, pretrained="vggface2", classify=True, num_classes=1, device=DEVICE):
        super(MyModel, self).__init__()
        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            classify=classify,
            num_classes=num_classes
        )
        self.device = device

    def forward(self, x):
        return self.backbone(x)



# ==================== Model Yolları ====================
MODEL_PATHS = {
    "aaronespasa-deepfake detection": "models/resnetinceptionv1_epoch_32.pth",   # PyTorch
    "vjdevane-deepfake detection": "models/deepfake_detection_model.h5",         # TF/Keras
    "MaanVad3r-DeepFake Detector": "models/cnn_model.h5",                       # TF/Keras
    "DontFakeMe - Deepfake Photo Detection": "models/model_epoch_6.keras",    # TF/Keras
    "DontFakeMe - Deepfake Video Detection": "models/model_epoch_6.keras"     # TF/Keras
}

# ==================== TensorFlow Custom Objects ====================
@tf.keras.utils.register_keras_serializable()
def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@tf.keras.utils.register_keras_serializable()
def f_score(y_true, y_pred):
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    precision_metric.update_state(y_true, y_pred)
    recall_metric.update_state(y_true, y_pred)

    precision = precision_metric.result()
    recall = recall_metric.result()
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

# ==================== Model Yükleme Fonksiyonları ====================
def load_keras_model(model_path: str):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"perceptual_loss": perceptual_loss, "f_score": f_score}
        )
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading TensorFlow model: {str(e)}")
def load_pytorch_model(model_path: str):
    try:
        map_location = torch.device(DEVICE)
        checkpoint = torch.load(model_path, map_location=map_location)

        # State dict'i kontrol et
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Modeli oluştur
        model = MyModel(
            pretrained="vggface2",
            classify=True,
            num_classes=1,
            device=DEVICE
        )

        # Anahtarları eşleştirmek için düzenleme
        state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

        # Ağırlıkları yükle
        model.load_state_dict(state_dict, strict=False)

        # GPU/CPU'ya taşı ve eval moduna al
        model.to(DEVICE)
        model.eval()

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading PyTorch model: {str(e)}")

def load_model(model_name: str):
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

    if model_path.endswith(".keras") or model_path.endswith(".h5"):
        return load_keras_model(model_path)
    elif model_path.endswith(".pth") or model_path.endswith(".pt"):
        return load_pytorch_model(model_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model format.")

# ==================== Prediction Fonksiyonları ====================
# 1) PyTorch fotoğraf analizi
def predict_photo(model, image):
    mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()
    face = mtcnn(image)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    face = face.unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        score = output.item()
        label = "fake" if score >= 0.5 else "real"
        return label, round(score * 100, 2)  # Yüzdelik formatta skor


# 2) TF/Keras fotoğraf analizi
def predict_photo_model_2(model, image):
    """
    TF/Keras tabanlı bir CNN modelini tek kare (fotoğraf) üzerinde çalıştırma
    """
    try:
        # Görüntüyü uygun boyutlara getirme (örnek: 128x128)
        image = cv2.resize(np.array(image), IMG_SIZE_VIDEO_3)
        image = image / 255.0  # Normalizasyon
        image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)

        prediction = model.predict(image, verbose=0)[0][0]
        label = "FAKE" if prediction >= 0.5 else "REAL"
        return label, prediction * 100  # Yüzde olarak döndür
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fotoğraf analizi sırasında hata: {e}")

# 3) TF/Keras video analizi (örnek 1 - inception_v3 pre-processing)
def process_video_model_1(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_SKIP == 0:
            resized_face = cv2.resize(frame, IMG_SIZE_VIDEO_1)
            frames.append(resized_face)
        frame_count += 1
    cap.release()

    # Frame sayısı NO_OF_FRAMES'den azsa doldur
    while len(frames) < NO_OF_FRAMES:
        frames.append(np.zeros((*IMG_SIZE_VIDEO_1, 3), dtype=np.float32))

    frames = np.array(frames[:NO_OF_FRAMES])
    frames = np.expand_dims(frames, axis=0)
    # (1, NO_OF_FRAMES, 299, 299, 3)
    frames = tf.keras.applications.inception_v3.preprocess_input(frames)

    prediction = model.predict(frames)
    avg_prediction = np.mean(prediction)
    label = "FAKE" if avg_prediction >= 0.6 else "REAL"
    return label, avg_prediction * 100

# 4) TF/Keras video analizi (örnek 2 - yine frame bazlı ama 128x128)
def process_video_model_2(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_SKIP == 0:
            resized_face = cv2.resize(frame, IMG_SIZE_VIDEO_2)
            frames.append(resized_face)
        frame_count += 1
    cap.release()

    while len(frames) < NO_OF_FRAMES:
        frames.append(np.zeros((*IMG_SIZE_VIDEO_2, 3), dtype=np.float32))

    predictions = []
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    for frame in frames[:NO_OF_FRAMES]:
        frame = np.expand_dims(frame, axis=0)  # (1, 128, 128, 3)
        frame = preprocess_input(frame)
        pred = model.predict(frame)[0][0]
        predictions.append(pred)

    avg_prediction = np.mean(predictions)
    label = "FAKE" if avg_prediction >= 0.6 else "REAL"
    return label, avg_prediction * 100

# 5) TF/Keras video analizi (örnek 3 - frame_skip=10)
def process_video_model_3(model, video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, IMG_SIZE_VIDEO_3)
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            pred = model.predict(frame, verbose=0)[0][0]
            predictions.append(pred)
        frame_count += 1
    cap.release()

    if not predictions:
        # Video hiç frame okumadıysa
        raise HTTPException(status_code=400, detail="No frames processed from the video.")

    fake_percentage = np.mean(predictions) * 100
    label = "FAKE" if fake_percentage > 50 else "REAL"
    return label, fake_percentage

# ==================== FastAPI Router ====================
@app.get("/")
def read_root():
    return {"message": "Deepfake Detection API is running!"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    selected_model: str = Form(...)
):
    if selected_model not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    # Dosya uzantısını modele göre ayırmak için .mp4 veya .jpg gibi geçici isim
    is_video = "Video" in selected_model  # model ismindeki "Video"ya göre ayırıyoruz
    suffix = ".mp4" if is_video else ".jpg"

    # Geçici dosya oluşturma
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file_path = temp_file.name
        with open(temp_file_path, 'wb') as dest:
            shutil.copyfileobj(file.file, dest)

    try:
        # Modeli yükle
        model = load_model(selected_model)

        # Tahmin için kullanılacak değişkenler
        label = "UNKNOWN"
        confidence = 0.0

        # Fotoğraf veya Video'ya göre hangi fonksiyonun çağrılacağı:
        if selected_model == "aaronespasa-deepfake detection":
            # PyTorch + MTCNN fotoğraf modeli
            image = Image.open(temp_file_path).convert("RGB")
            label, confidence = predict_photo(model, image)

        elif selected_model == "DontFakeMe - Deepfake Photo Detection":
            # Keras fotoğraf modeli
            image = Image.open(temp_file_path).convert("RGB")
            label, confidence = predict_photo_model_2(model, image)

        elif selected_model == "vjdevane-deepfake detection":
            # Keras video modeli (örnek: InceptionV3 tabanlı)
            label, confidence = process_video_model_1(model, temp_file_path)

        elif selected_model == "MaanVad3r-DeepFake Detector":
            # Keras video modeli
            label, confidence = process_video_model_2(model, temp_file_path)

        elif selected_model == "DontFakeMe - Deepfake Video Detection":
            # Keras video modeli (frame_skip=10)
            label, confidence = process_video_model_3(model, temp_file_path)

        # Örnek olarak; eğer "FAKE" diyorsak confidence da fake olma oranı olsun.
        # "REAL" ise 100 - confidence şeklinde bir mantık yürütebilirsiniz.
        if label.lower() == "fake":
            ai_fake_percentage = confidence
            real_percentage = 100 - confidence
        else:
            ai_fake_percentage = 100 - confidence
            real_percentage = confidence

        # Geçici dosyayı siliyoruz
        os.remove(temp_file_path)

        # API döndürülecek JSON - sayısal değerleri float olarak gönderiyoruz
        return JSONResponse(content={
            "model": selected_model,
            "label": label,
            "confidence": round(confidence, 2),
            "aiFakePercentage": round(ai_fake_percentage, 2),
            "realPercentage": round(real_percentage, 2)
        })

    except Exception as e:
        # Hata oluşursa dosyayı yine de silelim
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))