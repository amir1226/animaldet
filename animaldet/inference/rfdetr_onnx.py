"""RF-DETR ONNX inference with stitching support for large images.

This module provides efficient inference for RF-DETR models using ONNX Runtime,
with automatic stitching for images larger than the model's resolution.

Usage:
    from animaldet.inference.rfdetr_onnx import RFDETRONNXInference

    # Load model
    model = RFDETRONNXInference("model.onnx", confidence_threshold=0.5)

    # Run inference
    detections = model.predict("image.jpg")
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image


class RFDETRONNXInference:
    """RF-DETR inference using ONNX Runtime.

    Supports automatic stitching for images larger than model resolution.

    Args:
        model_path: Path to ONNX model file (.onnx)
        confidence_threshold: Minimum confidence score for detections
        nms_threshold: IoU threshold for NMS
        resolution: Model input resolution (default: 512)
        providers: ONNX Runtime execution providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        use_stitcher: Use stitcher for large images
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        resolution: Optional[int] = None,
        providers: Optional[List[str]] = None,
        use_stitcher: bool = True,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_stitcher = use_stitcher
        self.resolution = resolution or 512
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Load ONNX model
        if self.model_path.suffix != ".onnx":
            raise ValueError(f"Unsupported format: {self.model_path.suffix}. Use .onnx for ONNX models")

        self._load_onnx()

        # Preprocessing parameters (ImageNet normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_onnx(self):
        """Load ONNX model."""
        import onnxruntime as ort

        print(f"Loading ONNX model from {self.model_path}")
        print(f"Available providers: {ort.get_available_providers()}")

        # Filter providers to only those available
        available_providers = ort.get_available_providers()
        self.providers = [p for p in self.providers if p in available_providers]

        if not self.providers:
            self.providers = ['CPUExecutionProvider']

        print(f"Using providers: {self.providers}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Model loaded - Input: {self.input_name}, Outputs: {self.output_names}")

    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, np.ndarray]:
        """Run inference on an image.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Dictionary with 'boxes' [N,4], 'scores' [N], 'labels' [N]
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Use stitcher for large images
        if self.use_stitcher and (h > self.resolution or w > self.resolution):
            return self._predict_with_stitcher(img_array)

        return self._predict_single(img_array)

    def _predict_single(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single image."""
        h, w = image.shape[:2]

        # Resize to model resolution
        image_resized = Image.fromarray(image).resize((self.resolution, self.resolution))
        image_array = np.array(image_resized).astype(np.float32) / 255.0

        # Normalize (ImageNet mean/std)
        image_normalized = (image_array - self.mean) / self.std

        # Convert to CHW format (channels first) and add batch dimension
        image_tensor = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: image_tensor})

        # ONNX model returns tuple: (pred_logits, pred_boxes)
        pred_logits = outputs[0][0]  # Remove batch dimension
        pred_boxes = outputs[1][0]

        # Process outputs
        pred_scores = self._sigmoid(pred_logits)
        scores = pred_scores.max(axis=-1)
        labels = pred_scores.argmax(axis=-1) + 1

        # Filter by confidence
        keep = scores > self.confidence_threshold
        boxes = pred_boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes = self._box_cxcywh_to_xyxy(boxes) * self.resolution

        # Scale to original size
        boxes[:, [0, 2]] *= w / self.resolution
        boxes[:, [1, 3]] *= h / self.resolution

        # Apply NMS
        keep_nms = self._nms(boxes, scores, self.nms_threshold)
        return {"boxes": boxes[keep_nms], "scores": scores[keep_nms], "labels": labels[keep_nms]}

    def _predict_with_stitcher(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference with stitching for large images using numpy-only implementation."""
        from animaldet.inference.onnx_stitcher import ONNXStitcher

        # Preprocess image: convert to float and normalize
        image_normalized = image.astype(np.float32) / 255.0
        image_normalized = (image_normalized - self.mean) / self.std

        # Convert HWC to CHW format
        image_array = np.transpose(image_normalized, (2, 0, 1))  # Shape: [C, H, W]

        # Create stitcher
        stitcher = ONNXStitcher(
            onnx_session=self.session,
            input_name=self.input_name,
            output_names=self.output_names,
            size=(self.resolution, self.resolution),
            overlap=0,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

        # Run stitched inference
        detections = stitcher(image_array)
        return detections

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)


def predict_batch(
    model: RFDETRONNXInference,
    image_paths: List[Union[str, Path]],
) -> List[Dict[str, np.ndarray]]:
    """Run inference on a batch of images."""
    return [model.predict(img_path) for img_path in image_paths]
