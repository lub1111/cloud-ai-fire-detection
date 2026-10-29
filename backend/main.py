
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import threading

# ============================================================
# CLOUDVISION - AI FIRE DETECTION AND ALERT SYSTEM
# ============================================================

app = FastAPI(
    title="CloudVision Fire Detection API",
    description="Cloud-based AI fire and smoke detection system",
    version="1.1.0"
)

# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "ai-model" / "models"

GENERAL_MODEL_PATH = BASE_DIR / "yolo11n.pt"

FIRE_MODEL_PATH = MODEL_DIR / "best.pt"

MAX_IMAGE_SIZE = 5 * 1024 * 1024

ALLOWED_TYPES = ["image/jpeg", "image/png"]

CONFIDENCE_THRESHOLD = 0.50

# Lock prevents simultaneous model predictions
prediction_lock = threading.Lock()

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
    print("Using general YOLO model for testing.")
    model = YOLO(str(GENERAL_MODEL_PATH) if GENERAL_MODEL_PATH.is_file()
                 else "yolo11n.pt")
    model_type = "General YOLO Object Detection"

print("Model loaded successfully.")
print("Model type:", model_type)

# Check model class names
model_classes = {
    str(name).strip().lower()
    for name in model.names.values()
}

FIRE_CLASS_NAMES = {"fire", "flame", "flames"}
SMOKE_CLASS_NAMES = {"smoke"}

supports_fire = bool(model_classes & FIRE_CLASS_NAMES)
supports_smoke = bool(model_classes & SMOKE_CLASS_NAMES)

fire_detection_enabled = supports_fire or supports_smoke

# ============================================================
# HELPER FUNCTION - VALIDATE UPLOADED IMAGE
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
        "version": "1.1.0",
        "status": "running",
        "model_type": model_type,
        "fire_detection_enabled": fire_detection_enabled
    }


# ============================================================
# HEALTH CHECK ENDPOINT
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
# IMAGE UPLOAD ENDPOINT
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
# YOLO AI DETECTION ENDPOINT
# ============================================================

@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):

    image, size_bytes = read_and_validate_image(file)

    try:
        with prediction_lock:
            results = model.predict(
                source=image,
                device="cpu",
                conf=0.25,
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

        # Fire detection
        if normalized_name in FIRE_CLASS_NAMES:
            if confidence >= CONFIDENCE_THRESHOLD:
                fire_detected = True
                fire_confidence = max(
                    fire_confidence,
                    confidence
                )

        # Smoke detection
        if normalized_name in SMOKE_CLASS_NAMES:
            if confidence >= CONFIDENCE_THRESHOLD:
                smoke_detected = True
                smoke_confidence = max(
                    smoke_confidence,
                    confidence
                )

    # ========================================================
    # ALERT LOGIC
    # ========================================================

    if not fire_detection_enabled:
        alert_status = "MODEL_NOT_READY"
        alert_message = (
            "Fire-trained model is not installed. "
            "General object detection only."
        )

    elif fire_detected:
        alert_status = "FIRE_ALERT"
        alert_message = "WARNING: Fire detected!"

    elif smoke_detected:
        alert_status = "SMOKE_ALERT"
        alert_message = "WARNING: Smoke detected!"

    else:
        alert_status = "NO_FIRE_DETECTED"
        alert_message = (
            "No fire or smoke detected in this image."
        )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
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
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "AI analysis completed"
    }


# ============================================================
# MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "project": "CloudVision",
        "model_type": model_type,
        "model_path": str(
            FIRE_MODEL_PATH
            if fire_model_available
            else GENERAL_MODEL_PATH
        ),
        "fire_detection_enabled": fire_detection_enabled,
        "supported_classes": model.names,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "device": "cpu"
    }
