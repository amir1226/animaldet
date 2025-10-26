"""RF-DETR ONNX/TorchScript inference with stitching support for large images.

This module provides efficient inference for RF-DETR models using ONNX or TorchScript,
with automatic stitching for images larger than the model's resolution.

Usage:
    from animaldet.inference.rfdetr import RFDETRInference

    # Load model (ONNX or TorchScript)
    model = RFDETRInference("model.onnx", confidence_threshold=0.5)
    # or
    model = RFDETRInference("model.pt", confidence_threshold=0.5)

    # Run inference
    detections = model.predict("image.jpg")
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RFDETRInference:
    """RF-DETR inference using ONNX or TorchScript.

    Supports automatic stitching for images larger than model resolution.

    Args:
        model_path: Path to model file (.onnx or .pt)
        confidence_threshold: Minimum confidence score for detections
        nms_threshold: IoU threshold for NMS
        resolution: Model input resolution (auto-detect if not provided)
        device: Device for inference ('cuda' or 'cpu')
        use_stitcher: Use stitcher for large images
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        resolution: Optional[int] = None,
        device: str = "cuda",
        use_stitcher: bool = True,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_stitcher = use_stitcher
        self.device = device
        self.resolution = resolution or 512

        # Load model based on file extension
        if self.model_path.suffix == ".pt":
            self._load_torchscript()
        elif self.model_path.suffix == ".onnx":
            self._load_onnx()
            if resolution:
                self.resolution = resolution
        else:
            raise ValueError(f"Unsupported format: {self.model_path.suffix}. Use .pt or .onnx")

        # Preprocessing
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _load_onnx(self):
        """Load ONNX model."""
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.model_type = "onnx"
        self.resolution = self.session.get_inputs()[0].shape[2]

    def _load_torchscript(self):
        """Load TorchScript model."""
        # TorchScript models should be used on CPU for best compatibility
        print(f"Loading TorchScript model from {self.model_path}")
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()
        self.model_type = "torchscript"
        self.device = "cpu"  # Force CPU for TorchScript

        # Check if model returns tuple or dict by running a test inference
        dummy_input = torch.randn(1, 3, self.resolution, self.resolution)
        with torch.no_grad():
            test_output = self.model(dummy_input)
        self._returns_tuple = isinstance(test_output, tuple)
        print(f"Model returns {'tuple' if self._returns_tuple else 'dict'} output")

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
        image_array = np.array(image_resized)

        # Transform
        transformed = self.transform(image=image_array)
        image_tensor = transformed["image"]

        if self.model_type == "onnx":
            image_tensor = image_tensor.numpy()
            image_tensor = np.expand_dims(image_tensor, 0)
            outputs = self.session.run(None, {"images": image_tensor})
            pred_logits, pred_boxes = outputs[0], outputs[1]
            pred_logits = pred_logits[0]
            pred_boxes = pred_boxes[0]
        else:  # torchscript
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)

            # Handle both tuple and dict outputs
            if self._returns_tuple:
                pred_logits, pred_boxes = outputs
                pred_logits = pred_logits[0].cpu().numpy()
                pred_boxes = pred_boxes[0].cpu().numpy()
            else:
                pred_logits = outputs["pred_logits"][0].cpu().numpy()
                pred_boxes = outputs["pred_boxes"][0].cpu().numpy()

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
        """Run inference with stitching for large images."""
        from animaldet.experiments.rfdetr.stitcher import RFDETRStitcher

        # Wrapper for model compatibility
        class ModelWrapper(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, x):
                if self.parent.model_type == "onnx":
                    outputs = self.parent.session.run(None, {"images": x.numpy()})
                    return {"pred_logits": torch.from_numpy(outputs[0]), "pred_boxes": torch.from_numpy(outputs[1])}
                else:
                    with torch.no_grad():
                        outputs = self.parent.model(x)
                    # Convert tuple to dict for stitcher compatibility
                    if self.parent._returns_tuple:
                        return {"pred_logits": outputs[0], "pred_boxes": outputs[1]}
                    return outputs

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        wrapper = ModelWrapper(self)
        stitcher = RFDETRStitcher(
            model=wrapper,
            size=(self.resolution, self.resolution),
            overlap=160,
            batch_size=1,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            device_name="cpu" if self.model_type == "onnx" else self.device,
            voting_threshold=0.5,
        )

        detections = stitcher(image_tensor)
        return {
            "boxes": detections["boxes"].numpy(),
            "scores": detections["scores"].numpy(),
            "labels": detections["labels"].numpy(),
        }

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
    model: RFDETRInference,
    image_paths: List[Union[str, Path]],
) -> List[Dict[str, np.ndarray]]:
    """Run inference on a batch of images."""
    return [model.predict(img_path) for img_path in image_paths]
