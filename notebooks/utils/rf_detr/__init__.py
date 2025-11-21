"""RF-DETR specific helpers for notebooks."""

from .callbacks import HerdNetMetricsCallback
from .detections import Detection, DetectionSample, DEFAULT_CATEGORIES, write_coco_predictions
from .patcher import PatchSummary, generate_patch_dataset
from .stitcher import SimpleStitcher

__all__ = [
    "HerdNetMetricsCallback",
    "SimpleStitcher",
    "generate_patch_dataset",
    "PatchSummary",
    "Detection",
    "DetectionSample",
    "DEFAULT_CATEGORIES",
    "write_coco_predictions",
]

