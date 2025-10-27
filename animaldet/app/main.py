"""FastAPI application for RF-DETR inference."""

import io
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

from animaldet.inference.rfdetr import RFDETRInference
from animaldet.inference.rfdetr_onnx import RFDETRONNXInference
from animaldet.app.class_names import get_class_name


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


class InferenceResponse(BaseModel):
    """Inference API response."""

    data: dict


class RFDETRService:
    """RF-DETR inference service."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        resolution: int = 512,
        device: str = "cuda",
        use_stitcher: bool = True,
        runtime: str = "onnx",
    ):
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.runtime = runtime

        # Load model based on runtime
        if runtime == "onnx":
            self.model = RFDETRONNXInference(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                resolution=resolution,
                use_stitcher=use_stitcher,
            )
        elif runtime == "torchscript":
            self.model = RFDETRInference(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                resolution=resolution,
                device=device,
                use_stitcher=use_stitcher,
            )
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Choose 'onnx' or 'torchscript'")

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on image bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary with detections and metadata
        """
        # Load image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}. Received {len(image_bytes)} bytes.")

        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]

        # Run inference
        start_time = time.time()
        detections = self.model.predict(image)
        latency_ms = (time.time() - start_time) * 1000

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

            detection_list.append(
                Detection(
                    class_id=int(labels[i]),
                    class_name=get_class_name(int(labels[i])),
                    bbox=BoundingBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        confidence=float(scores[i]),
                    ),
                )
            )

        metadata = Metadata(
            model=self.model_name,
            input_shape=[img_w, img_h],
            num_detections=len(detection_list),
            latency_ms=round(latency_ms, 2),
        )

        return {
            "detections": [det.model_dump() for det in detection_list],
            "metadata": metadata.model_dump(),
        }


# Global model instance
_model_service: Optional[RFDETRService] = None


def get_model_service() -> RFDETRService:
    """Get or initialize the model service."""
    global _model_service
    if _model_service is None:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized. Call /initialize endpoint first.",
        )
    return _model_service


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

# Mount static file directories
# Get project root (assuming this file is in animaldet/app/main.py)
project_root = Path(__file__).parent.parent.parent

data_dir = project_root / "data"
outputs_dir = project_root / "outputs"

# Mount directories if they exist
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
if outputs_dir.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.post("/api/initialize")
async def initialize_model(
    model_path: str = "model.onnx",
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.45,
    resolution: int = 512,
    device: str = "cuda",
    use_stitcher: bool = True,
    runtime: str = "onnx",
):
    """Initialize the model service.

    Args:
        model_path: Path to model file (.pt for torchscript or .onnx for onnx)
        confidence_threshold: Minimum confidence for detections
        nms_threshold: IoU threshold for NMS
        resolution: Model input resolution
        device: Device for inference ('cuda' or 'cpu', only for torchscript)
        use_stitcher: Use stitching for large images
        runtime: Runtime to use ('onnx' or 'torchscript', default: 'onnx')
    """
    global _model_service
    try:
        _model_service = RFDETRService(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            resolution=resolution,
            device=device,
            use_stitcher=use_stitcher,
            runtime=runtime,
        )
        return {"status": "success", "message": f"Model loaded from {model_path} using {runtime} runtime"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: Request):
    """Run inference on an image.

    Args:
        request: Raw image bytes in request body (application/octet-stream)

    Returns:
        Inference results with detections and metadata
    """
    service = get_model_service()

    try:
        # Read raw body bytes
        image_bytes = await request.body()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image data received")

        if len(image_bytes) < 100:
            raise HTTPException(status_code=400, detail="Image data too small, likely corrupted")

        result = service.predict(image_bytes)
        return {"data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RF-DETR Inference API",
        "endpoints": {
            "initialize": "POST /api/initialize - Initialize model",
            "inference": "POST /api/inference - Run inference",
            "health": "GET /health - Health check",
            "data": "GET /data/* - Static files from data directory",
            "outputs": "GET /outputs/* - Static files from outputs directory",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
