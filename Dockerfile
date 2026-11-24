
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# Install system libraries used by YOLO and Pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and YOLO model
COPY backend/ ./backend/
COPY ai-model/models/ ./ai-model/models/

# API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
