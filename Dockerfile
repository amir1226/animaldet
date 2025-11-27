# Build stage for frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# Model conversion stage
FROM python:3.12-slim AS model-converter

WORKDIR /app

# Install dependencies for model conversion
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    onnx \
    onnxscript \
    git+https://github.com/roboflow/rf-detr

# Copy conversion script and models
COPY tools/convert_to_onnx.py ./tools/convert_to_onnx.py
COPY modelos/rf-detr-nano-animaldet.pth ./modelos/rf-detr-nano-animaldet.pth
COPY modelos/rf-detr-small-animaldet.pth ./modelos/rf-detr-small-animaldet.pth

# Convert PyTorch models to ONNX (both nano and small)
RUN python tools/convert_to_onnx.py --models all && \
    ls -lh modelos/*.onnx modelos/*.json && \
    echo "✓ Model conversion completed successfully"

# Production stage - lightweight runtime
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Copy application code
COPY animaldet/ ./animaldet/

# Copy built frontend files
COPY --from=frontend-builder /frontend/dist ./static

# Copy experiment CSVs for demo and ground truth
COPY experiments/RF-DETR/results/rfdetr_detections_phase2.csv ./static/experiments/rfdetr_detections_phase2.csv
COPY experiments/HerdNet/results/test_big_size_A_B_E_K_WH_WB.csv ./static/experiments/test_big_size_A_B_E_K_WH_WB.csv
COPY experiments/HerdNet/results/test_big_size_A_B_E_K_WH_WB.csv ./experiments/HerdNet/results/test_big_size_A_B_E_K_WH_WB.csv

# Copy demo images to app root (API endpoint looks for them here)
COPY demo_images/ ./demo_images/

# Create modelos directory and copy ONNX models there (matching registry paths)
RUN mkdir -p modelos
COPY --from=model-converter /app/modelos/rf-detr-nano-animaldet.onnx ./modelos/rf-detr-nano-animaldet.onnx
COPY --from=model-converter /app/modelos/rf-detr-nano-animaldet.json ./modelos/rf-detr-nano-animaldet.json
COPY --from=model-converter /app/modelos/rf-detr-small-animaldet.onnx ./modelos/rf-detr-small-animaldet.onnx
COPY --from=model-converter /app/modelos/rf-detr-small-animaldet.json ./modelos/rf-detr-small-animaldet.json

# Install minimal production dependencies
COPY infra/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_NAME=small

EXPOSE 8000

CMD ["uvicorn", "animaldet.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
