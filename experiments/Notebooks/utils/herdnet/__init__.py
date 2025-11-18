"""HerdNet training, evaluation, and HNP generation helpers for notebooks."""

from .evaluate import EvalConfig, EvalResult, evaluate_full_images, export_detections_csv
from .hnp import HNPConfig, HNPResult, generate_hard_negative_patches
from .metrics import evaluate_points_from_csv, load_class_map
from .train import (
    TrainConfig,
    TrainResult,
    enrich_checkpoint_metadata,
    load_backbone_weights,
    train_stage1,
    train_stage2,
)

__all__ = [
    # Training
    "TrainConfig",
    "TrainResult",
    "train_stage1",
    "train_stage2",
    "load_backbone_weights",
    "enrich_checkpoint_metadata",
    # Hard Negative Patches
    "HNPConfig",
    "HNPResult",
    "generate_hard_negative_patches",
    # Evaluation
    "EvalConfig",
    "EvalResult",
    "evaluate_full_images",
    "export_detections_csv",
    # Metrics
    "evaluate_points_from_csv",
    "load_class_map",
]
