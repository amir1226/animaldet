"""Numpy-based image stitcher for ONNX inference without torch dependencies."""

import numpy as np
from typing import Dict, List, Tuple


class ONNXStitcher:
    """Stitcher for ONNX inference to handle large images using only numpy.

    This class divides large images into patches at the model's expected resolution,
    runs inference on each patch, and rescales the bounding box predictions back to
    the original image coordinates. No overlap or NMS is applied (for E2E models).

    Args:
        onnx_session: ONNX runtime session
        input_name: Name of the input tensor
        output_names: Names of the output tensors
        size: Patch size (height, width), typically (512, 512)
        confidence_threshold: Minimum confidence score for detections (default: 0.5)
    """

    def __init__(
        self,
        onnx_session,
        input_name: str,
        output_names: List[str],
        size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.5,
    ):
        self.session = onnx_session
        self.input_name = input_name
        self.output_names = output_names
        self.size = size
        self.confidence_threshold = confidence_threshold

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply stitching algorithm to the image.

        Args:
            image: Input image array of shape [C, H, W] (normalized)

        Returns:
            Dictionary containing:
                - 'boxes': Array of shape [N, 4] with boxes in (x1, y1, x2, y2) format
                - 'scores': Array of shape [N] with confidence scores
                - 'labels': Array of shape [N] with class labels
        """
        # Get image dimensions
        c, h, w = image.shape

        # Calculate patch grid (no overlap)
        patch_h, patch_w = self.size

        # Calculate number of patches
        n_patches_h = max(1, int(np.ceil(h / patch_h)))
        n_patches_w = max(1, int(np.ceil(w / patch_w)))

        all_boxes = []
        all_scores = []
        all_labels = []

        # Process each patch
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates (no overlap)
                y1 = i * patch_h
                x1 = j * patch_w
                y2 = min(y1 + patch_h, h)
                x2 = min(x1 + patch_w, w)

                # Extract patch
                patch = image[:, y1:y2, x1:x2]

                # Pad if needed
                if patch.shape[1] != patch_h or patch.shape[2] != patch_w:
                    padded = np.zeros((c, patch_h, patch_w), dtype=np.float32)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded

                # Run inference on patch
                patch_batch = patch[np.newaxis, :, :, :]  # Add batch dimension
                outputs = self.session.run(self.output_names, {self.input_name: patch_batch})

                pred_logits = outputs[0][0]  # [num_queries, num_classes]
                pred_boxes = outputs[1][0]   # [num_queries, 4]

                # Convert logits to scores (sigmoid)
                pred_scores = 1 / (1 + np.exp(-pred_logits))

                # Get max score and class for each query
                scores = pred_scores.max(axis=-1)
                labels = pred_scores.argmax(axis=-1) + 1  # Convert to 1-indexed

                # Filter by confidence threshold
                keep = scores > self.confidence_threshold
                boxes = pred_boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                if len(boxes) == 0:
                    continue

                # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                boxes = self._box_cxcywh_to_xyxy(boxes)
                boxes = boxes * patch_w  # Denormalize to patch size

                # Rescale boxes to original image coordinates
                boxes[:, [0, 2]] += x1  # Add x offset
                boxes[:, [1, 3]] += y1  # Add y offset

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # Combine all detections
        if len(all_boxes) == 0:
            return {
                'boxes': np.array([], dtype=np.float32).reshape(0, 4),
                'scores': np.array([], dtype=np.float32),
                'labels': np.array([], dtype=np.int64)
            }

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # No NMS for E2E models
        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'labels': all_labels
        }

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from center format to corner format."""
        if len(boxes) == 0:
            return boxes
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        return np.stack([x1, y1, x2, y2], axis=-1)
