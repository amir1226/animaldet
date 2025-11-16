"""Numpy-based image stitcher for ONNX inference without torch dependencies."""

import numpy as np
from typing import Dict, List, Tuple


class ONNXStitcher:
    """Stitcher for ONNX inference to handle large images using only numpy.

    This class divides large images into patches at the model's expected resolution,
    runs inference on each patch, and rescales the bounding box predictions back to
    the original image coordinates.

    Args:
        onnx_session: ONNX runtime session
        input_name: Name of the input tensor
        output_names: Names of the output tensors
        size: Patch size (height, width), typically (512, 512)
        overlap: Overlap between patches in pixels (default: 0)
        confidence_threshold: Minimum confidence score for detections (default: 0.5)
        nms_threshold: IoU threshold for non-maximum suppression (default: 0.45)
    """

    def __init__(
        self,
        onnx_session,
        input_name: str,
        output_names: List[str],
        size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        self.session = onnx_session
        self.input_name = input_name
        self.output_names = output_names
        self.size = size
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

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

        # Calculate patch grid
        patch_h, patch_w = self.size
        stride_h = patch_h - self.overlap
        stride_w = patch_w - self.overlap

        # Calculate number of patches
        n_patches_h = max(1, int(np.ceil((h - patch_h) / stride_h)) + 1)
        n_patches_w = max(1, int(np.ceil((w - patch_w) / stride_w)) + 1)

        all_boxes = []
        all_scores = []
        all_labels = []

        # Process each patch
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                y1 = min(i * stride_h, h - patch_h)
                x1 = min(j * stride_w, w - patch_w)
                y2 = y1 + patch_h
                x2 = x1 + patch_w

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

        # Apply NMS across all patches
        keep_nms = self._nms(all_boxes, all_scores, self.nms_threshold)

        return {
            'boxes': all_boxes[keep_nms],
            'scores': all_scores[keep_nms],
            'labels': all_labels[keep_nms]
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

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Non-maximum suppression using numpy."""
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
