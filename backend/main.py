import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import threading

# ============================================================
# CLOUDVISION - AI FIRE DETECTION AND ALERT SYSTEM
# ============================================================

APP_VERSION = "1.3.2"

app = FastAPI(
    title="CloudVision Fire Detection API",
    description="Cloud-based AI fire and smoke detection system",
    version=APP_VERSION
)

# ============================================================
# CORS - FRONTEND CONNECTION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "ai-model" / "models"

FIRE_MODEL_PATH = MODEL_DIR / "best.pt"
GENERAL_MODEL_PATH = BASE_DIR / "yolo11n.pt"

MAX_IMAGE_SIZE = 5 * 1024 * 1024

ALLOWED_TYPES = ["image/jpeg", "image/png"]

# Experimental detection settings
CONFIDENCE_THRESHOLD = 0.10
IMAGE_SIZE = 960
HISTORY_FILE = BASE_DIR / "backend" / "alert_history.json"
history_lock = threading.Lock()

prediction_lock = threading.Lock()

FIRE_CLASS_NAMES = {"fire", "flame", "flames"}
SMOKE_CLASS_NAMES = {"smoke"}

# ============================================================
# LOAD YOLO MODEL
# ============================================================

fire_model_available = FIRE_MODEL_PATH.is_file()

if fire_model_available:
    print("Loading fire detection model...")

    model = YOLO(str(FIRE_MODEL_PATH))
    model_type = "Fire and Smoke Detection YOLO"

else:
    print("Fire-trained model not found.")
    print("Loading general YOLO model instead.")

    if GENERAL_MODEL_PATH.is_file():
        model = YOLO(str(GENERAL_MODEL_PATH))
    else:
        model = YOLO("yolo11n.pt")

    model_type = "General YOLO Object Detection"

print("Model loaded successfully.")
print("Model type:", model_type)

model_classes = {
    str(name).strip().lower()
    for name in model.names.values()
}

supports_fire = bool(model_classes & FIRE_CLASS_NAMES)
supports_smoke = bool(model_classes & SMOKE_CLASS_NAMES)

fire_detection_enabled = supports_fire or supports_smoke

print("Fire detection supported:", supports_fire)
print("Smoke detection supported:", supports_smoke)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Image size:", IMAGE_SIZE)
print("Backend version:", APP_VERSION)

# ============================================================
# IMAGE VALIDATION
# ============================================================

def read_and_validate_image(file: UploadFile):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed."
        )

    content = file.file.read(MAX_IMAGE_SIZE + 1)

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 5 MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        image = Image.open(BytesIO(content))
        image.load()

        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Unsupported image format")

        image = image.convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    return image, len(content)


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "project": "CloudVision",
        "description": "AI Fire Detection and Alert System",
        "version": APP_VERSION,
        "status": "running",
        "model_type": model_type,
        "fire_detection_enabled": fire_detection_enabled
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "CloudVision Backend",
        "model_loaded": True,
        "model_type": model_type,
        "fire_detection_enabled": fire_detection_enabled,
        "fire_classes_supported": supports_fire,
        "smoke_classes_supported": supports_smoke
    }


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/upload")
def upload_image(file: UploadFile = File(...)):

    image, size_bytes = read_and_validate_image(file)

    return {
        "project": "CloudVision",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": size_bytes,
        "image_width": image.width,
        "image_height": image.height,
        "status": "Image uploaded successfully",
        "prediction": "Not analyzed yet"
    }


# ============================================================
# YOLO FIRE AND SMOKE DETECTION
# ============================================================

@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):

    image, size_bytes = read_and_validate_image(file)

    if not fire_detection_enabled:
        raise HTTPException(
            status_code=503,
            detail="Fire and smoke detection model is not available."
        )

    try:
        with prediction_lock:
            results = model.predict(
                source=image,
                device="cpu",
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

    except Exception as error:
        print("YOLO prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail="AI prediction failed."
        )

    detections = []

    fire_detected = False
    smoke_detected = False

    fire_confidence = 0.0
    smoke_confidence = 0.0

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = str(model.names[class_id])
        normalized_name = class_name.strip().lower()

        coordinates = box.xyxy[0].tolist()

        detection = {
            "object": class_name,
            "confidence": round(confidence, 3),
            "bounding_box": {
                "x1": round(coordinates[0], 2),
                "y1": round(coordinates[1], 2),
                "x2": round(coordinates[2], 2),
                "y2": round(coordinates[3], 2)
            }
        }

        detections.append(detection)

        if normalized_name in FIRE_CLASS_NAMES:
            fire_detected = True
            fire_confidence = max(
                fire_confidence,
                confidence
            )

        if normalized_name in SMOKE_CLASS_NAMES:
            smoke_detected = True
            smoke_confidence = max(
                smoke_confidence,
                confidence
            )

    # ========================================================
    # ALERT LOGIC
    # ========================================================

    if fire_detected:
        alert_status = "FIRE_ALERT"
        alert_message = "WARNING: Possible fire detected!"

    elif smoke_detected:
        alert_status = "SMOKE_ALERT"
        alert_message = "WARNING: Possible smoke detected!"

    else:
        alert_status = "NO_FIRE_DETECTED"
        alert_message = (
            "No fire or smoke detected by the model. "
            "This does not guarantee safety."
        )

       # ========================================================
    # API RESPONSE
    # ========================================================

    result = {
        "project": "CloudVision",
        "filename": file.filename,
        "size_bytes": size_bytes,
        "model_type": model_type,
        "fire_detection_enabled": fire_detection_enabled,
        "total_detections": len(detections),
        "detections": detections,
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "fire_confidence": round(fire_confidence, 3),
        "smoke_confidence": round(smoke_confidence, 3),
        "alert": {
            "status": alert_status,
            "message": alert_message
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AI analysis completed"
    }

    save_alert_history(result)
    return result


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "project": "CloudVision",
        "version": APP_VERSION,
        "model_type": model_type,
        "model_path": str(
            FIRE_MODEL_PATH
            if fire_model_available
            else GENERAL_MODEL_PATH
        ),
        "fire_detection_enabled": fire_detection_enabled,
        "supported_classes": model.names,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "device": "cpu"
    }
    # ============================================================
# ALERT HISTORY
# ============================================================

def save_alert_history(result):
    with history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        history.append(result)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


@app.get("/alerts")
def get_alert_history():
    with history_lock:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

    return {
        "project": "CloudVision",
        "total_records": len(history),
        "alerts": history
    }
