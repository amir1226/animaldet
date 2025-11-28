"""FastAPI application for RF-DETR inference."""

import csv
import io
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from animaldet.inference.rfdetr_onnx import RFDETRONNXInference
from animaldet.inference.registry import MODELS
from animaldet.app.class_names import get_class_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    """Bounding box in xywh format."""

    x: float
    y: float
    w: float
    h: float
    confidence: float


class Detection(BaseModel):
    """Single detection result."""

    class_id: int
    class_name: str
    bbox: BoundingBox


class Metadata(BaseModel):
    """Inference metadata."""

    model: str
    input_shape: List[int]
    num_detections: int
    latency_ms: float
    stitch_steps: Optional[int] = None
    gpu_memory_mb: Optional[float] = None


class InferenceResponse(BaseModel):
    """Inference API response."""

    data: dict


class RFDETRService:
    """RF-DETR inference service."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        resolution: int = 512,
        device: str = "cuda",
        use_stitcher: bool = True,
        runtime: str = "onnx",
        class_offset: int = 0,
    ):
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.runtime = runtime
        self.class_offset = class_offset

        # Only ONNX runtime supported in production
        if runtime != "onnx":
            raise ValueError(f"Only ONNX runtime is supported in production. Got: {runtime}")

        self.model = RFDETRONNXInference(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            resolution=resolution,
            use_stitcher=use_stitcher,
        )

    def predict(self, image_bytes: bytes, confidence_threshold: Optional[float] = None) -> dict:
        """Run inference on image bytes.

        Args:
            image_bytes: Raw image bytes
            confidence_threshold: Optional confidence threshold (overrides default)

        Returns:
            Dictionary with detections and metadata
        """
        # Load image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
            logger.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            logger.exception(f"Failed to load image: {str(e)}")
            raise ValueError(f"Failed to load image: {str(e)}. Received {len(image_bytes)} bytes.")

        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]
        logger.info(f"Image shape: {img_h}x{img_w}")

        # Use provided confidence threshold or default
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        logger.info(f"Using confidence threshold: {conf_thresh}")

        # Temporarily set confidence threshold for this inference
        original_threshold = self.model.confidence_threshold
        self.model.confidence_threshold = conf_thresh

        # Run inference
        start_time = time.time()
        logger.info("Starting model prediction...")
        detections = self.model.predict(image)
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Model prediction completed in {latency_ms:.2f}ms")

        # Restore original threshold
        self.model.confidence_threshold = original_threshold

        # Convert detections to response format
        detection_list = []
        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["labels"]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Convert from xyxy to xywh
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)

            # Apply class offset to correct for model-specific class numbering
            corrected_class_id = int(labels[i]) + self.class_offset

            detection_list.append(
                Detection(
                    class_id=corrected_class_id,
                    class_name=get_class_name(corrected_class_id),
                    bbox=BoundingBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        confidence=float(scores[i]),
                    ),
                )
            )

        # Extract metrics if available
        metrics = detections.get("metrics", {})

        metadata = Metadata(
            model=self.model_name,
            input_shape=[img_w, img_h],
            num_detections=len(detection_list),
            latency_ms=round(latency_ms, 2),
            stitch_steps=metrics.get("stitch_steps"),
            gpu_memory_mb=metrics.get("gpu_memory_mb"),
        )

        return {
            "detections": [det.model_dump() for det in detection_list],
            "metadata": metadata.model_dump(),
        }


# Global model instances (support multiple models)
_model_services: dict[str, RFDETRService] = {}
_current_model: Optional[str] = None


def get_model_service(model_name: Optional[str] = None) -> RFDETRService:
    """Get or load a model service on-demand.

    Args:
        model_name: Name of the model to use, or None for current/default model
    """
    global _current_model

    # Use current model if no specific model requested
    if model_name is None:
        model_name = _current_model

    if model_name is None:
        raise HTTPException(
            status_code=500,
            detail="No model initialized.",
        )

    # Load model on-demand if not already loaded
    if model_name not in _model_services:
        try:
            model_config = MODELS.get(model_name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Get default settings from env
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        use_stitcher = os.getenv("USE_STITCHER", "true").lower() == "true"

        service = RFDETRService(
            model_path=model_config.model_path,
            confidence_threshold=confidence_threshold,
            resolution=model_config.resolution,
            device="cpu",
            use_stitcher=use_stitcher,
            runtime="onnx",
            class_offset=model_config.class_offset,
        )
        _model_services[model_name] = service
        logger.info(f"Loaded model '{model_name}' on-demand")

    return _model_services[model_name]


# FastAPI app
app = FastAPI(
    title="RF-DETR Inference API",
    description="Animal detection inference using RF-DETR",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
async def list_models():
    """List all available models."""
    models = MODELS.list_models()
    default = MODELS.get_default()
    return {
        "models": models,
        "default": default,
        "loaded": list(_model_services.keys()),
        "current": _current_model,
    }


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the default model on startup."""
    global _current_model

    try:
        # Get model name from env or use default (nano)
        model_name = os.getenv("MODEL_NAME", MODELS.get_default())
        _current_model = model_name

        # Preload the default model
        get_model_service(model_name)

        print(f"✓ Model '{model_name}' loaded successfully on startup")
        print(f"  Available models: {list(MODELS.list_models().keys())}")
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {str(e)}")
        print("Models will be loaded on-demand when requested")


@app.post("/api/inference", response_model=InferenceResponse)
async def inference(
    request: Request,
    confidence_threshold: Optional[float] = None,
    model: Optional[str] = None,
):
    """Run inference on an image.

    Args:
        request: Raw image bytes in request body (application/octet-stream)
        confidence_threshold: Optional confidence threshold (overrides default)
        model: Optional model name to use ("nano" or "small", default: current model)

    Returns:
        Inference results with detections and metadata
    """
    logger.info(f"Received inference request (model={model or _current_model})")
    service = get_model_service(model)

    try:
        # Read raw body bytes
        image_bytes = await request.body()
        logger.info(f"Image data size: {len(image_bytes)} bytes")

        if not image_bytes:
            logger.error("No image data received")
            raise HTTPException(status_code=400, detail="No image data received")

        if len(image_bytes) < 100:
            logger.error(f"Image data too small: {len(image_bytes)} bytes")
            raise HTTPException(status_code=400, detail="Image data too small, likely corrupted")

        logger.info("Running inference...")
        result = service.predict(image_bytes, confidence_threshold=confidence_threshold)
        logger.info(f"Inference successful: {len(result['detections'])} detections found")
        return {"data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Inference failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/sample-images")
async def get_sample_images():
    """Get a list of sample images for testing."""
    import random

    demo_images_dir = project_root / "demo_images"

    if not demo_images_dir.exists():
        return JSONResponse(
            content={"images": []},
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    # Get all JPG files
    all_images = list(demo_images_dir.glob("*.JPG")) + list(demo_images_dir.glob("*.jpg"))

    # Select 5 random images
    sample_count = min(5, len(all_images))
    sampled_images = random.sample(all_images, sample_count) if all_images else []

    # Return image names with no-cache headers
    return JSONResponse(
        content={"images": [img.name for img in sampled_images]},
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/ground-truth/{image_name}")
async def get_ground_truth(image_name: str):
    """Get ground truth detections for a specific image from HerdNet ground truth CSV.

    Args:
        image_name: Name of the image file

    Returns:
        Ground truth detections in the same format as inference results
    """
    # Use the actual ground truth CSV with bounding boxes
    ground_truth_csv = project_root / "experiments" / "HerdNet" / "results" / "test_big_size_A_B_E_K_WH_WB.csv"

    if not ground_truth_csv.exists():
        return JSONResponse(
            content={"detections": [], "available": False},
            status_code=404
        )

    try:
        detections = []
        with open(ground_truth_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Image'] == image_name:
                    # Parse the detection from ground truth CSV
                    # CSV format: Image,x1,y1,x2,y2,Label
                    x1 = float(row['x1'])
                    y1 = float(row['y1'])
                    x2 = float(row['x2'])
                    y2 = float(row['y2'])
                    class_id = int(row['Label'])

                    # Convert from x1,y1,x2,y2 to x,y,w,h format
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1

                    detections.append({
                        "class_id": class_id,
                        "class_name": get_class_name(class_id),
                        "bbox": {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "confidence": 1.0  # Ground truth has 100% confidence
                        }
                    })

        return JSONResponse(
            content={
                "detections": detections,
                "available": True,
                "source": "HerdNet Ground Truth"
            }
        )
    except Exception as e:
        logger.exception(f"Failed to load ground truth: {str(e)}")
        return JSONResponse(
            content={"detections": [], "available": False, "error": str(e)},
            status_code=500
        )


# Mount static file directories at the end (after all API routes)
# Get project root (assuming this file is in animaldet/app/main.py)
project_root = Path(__file__).parent.parent.parent

data_dir = project_root / "data"
outputs_dir = project_root / "outputs"
demo_images_dir = project_root / "demo_images"
static_dir = project_root / "static"

# Mount directories if they exist
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
if outputs_dir.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")
if demo_images_dir.exists():
    app.mount("/demo_images", StaticFiles(directory=str(demo_images_dir)), name="demo_images")

# Mount frontend static files (must be last to allow SPA routing)
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
