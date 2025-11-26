"""Evaluation functions for HerdNet on full-resolution images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import albumentations as A
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.eval.metrics import PointsMetrics
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.models import HerdNet, LossWrapper
from animaloc.train.losses import FocalLoss

from .metrics import evaluate_points_from_csv

__all__ = [
    "EvalConfig",
    "EvalResult",
    "evaluate_full_images",
    "export_detections_csv",
]

DEFAULT_CLASSES = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class EvalConfig:
    """Configuration for HerdNet evaluation on full images."""

    # Checkpoint and data
    checkpoint: Path
    csv: Path
    root: Path

    # Output
    output_dir: Optional[Path] = None
    detections_csv: Optional[Path] = None
    metrics_json: Optional[Path] = None

    # Inference settings
    device: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 4

    # Model parameters
    down_ratio: int = 2
    patch_size: int = 512
    overlap: int = 160
    upsample: bool = True

    # LMDS parameters
    kernel_size: Tuple[int, int] = (3, 3)
    adapt_ts: float = 0.3
    neg_ts: float = 0.1

    # Metrics
    match_radius: float = 5.0

    def __post_init__(self):
        self.checkpoint = Path(self.checkpoint)
        self.csv = Path(self.csv)
        self.root = Path(self.root)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
        if self.detections_csv:
            self.detections_csv = Path(self.detections_csv)
        if self.metrics_json:
            self.metrics_json = Path(self.metrics_json)


@dataclass
class EvalResult:
    """Results from HerdNet evaluation."""

    detections_csv: Path
    metrics: Dict
    output_dir: Path


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    """Load checkpoint from disk."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def _build_model(num_classes: int, device: torch.device) -> LossWrapper:
    """Build HerdNet model with loss wrapper."""
    base_model = HerdNet(
        num_classes=num_classes,
        down_ratio=2,
        num_layers=34,
        head_conv=64,
        pretrained=False,
    )

    class_weights = torch.tensor(
        [0.1, 1.0, 2.0, 1.0, 6.0, 12.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    losses = [
        {
            "loss": FocalLoss(reduction="mean"),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": CrossEntropyLoss(reduction="mean", weight=class_weights),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    wrapper = LossWrapper(base_model, losses=losses)
    wrapper = wrapper.to(device)
    return wrapper


def _to_numpy_points(target_entry: torch.Tensor) -> List[Tuple[float, float]]:
    """Convert tensor target to list of point tuples."""
    tensor = target_entry
    if isinstance(tensor, list):
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 3:
            tensor = tensor.squeeze(0)
        return [tuple(map(float, coords)) for coords in tensor.tolist()]
    raise TypeError("Unexpected type for target points")


def _to_numpy_labels(target_entry: torch.Tensor) -> List[int]:
    """Convert tensor labels to list of ints."""
    tensor = target_entry
    if isinstance(tensor, list):
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 2:
            tensor = tensor.squeeze(0)
        return [int(x) for x in tensor.tolist()]
    raise TypeError("Unexpected type for target labels")


def export_detections_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    """Export detections to CSV file.

    Args:
        path: Output CSV path
        records: Iterable of detection dictionaries
    """
    fieldnames = ["images", "x", "y", "labels", "scores", "det_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _split_stitcher_output(
    output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split stitcher output into heatmap and class map."""
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError("Expected tuple/list of length 2 from stitcher")
        heatmap, clsmap = output
        if isinstance(heatmap, torch.Tensor) and heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(0)
        if isinstance(clsmap, torch.Tensor) and clsmap.ndim == 3:
            clsmap = clsmap.unsqueeze(0)
        return heatmap, clsmap

    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            output = output.unsqueeze(0)
        if output.shape[1] < 2:
            raise ValueError("Stitcher output tensor must have at least 2 channels")
        heatmap = output[:, :1, ...]
        clsmap = output[:, 1:, ...]
        return heatmap, clsmap

    raise TypeError(f"Unsupported stitcher output type: {type(output)}")


def _normalize_image_name(value: Union[str, List[str], Tuple[str, ...]]) -> str:
    """Normalize image name from various formats."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        value = value[0]
    return str(value)


def evaluate_full_images(config: EvalConfig) -> EvalResult:
    """Evaluate HerdNet on full-resolution images.

    This function:
    1. Loads a trained checkpoint
    2. Runs sliding-window inference on full images
    3. Extracts detections using LMDS
    4. Computes metrics against ground truth
    5. Exports detections to CSV

    Args:
        config: Evaluation configuration

    Returns:
        EvalResult with metrics and output paths
    """
    device = torch.device(
        config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = _load_checkpoint(config.checkpoint, device)
    classes = {
        int(k): str(v) for k, v in checkpoint.get("classes", DEFAULT_CLASSES).items()
    }
    mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(checkpoint.get("std", DEFAULT_STD))

    num_classes = len(classes) + 1

    dataset_down_ratio = 1 if config.upsample else config.down_ratio

    dataset = CSVDataset(
        csv_file=str(config.csv),
        root_dir=str(config.root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=dataset_down_ratio, anno_type="point")],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_wrapper = _build_model(num_classes, device)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    model_wrapper.eval()

    stitcher = HerdNetStitcher(
        model=model_wrapper,
        size=(config.patch_size, config.patch_size),
        overlap=config.overlap,
        down_ratio=config.down_ratio,
        reduction="mean",
        up=config.upsample,
        device_name=device.type,
    )

    lmds = HerdNetLMDS(
        up=False,
        kernel_size=tuple(config.kernel_size),
        adapt_ts=config.adapt_ts,
        neg_ts=config.neg_ts,
    )

    metrics = PointsMetrics(radius=config.match_radius, num_classes=num_classes)
    detections: List[Dict[str, object]] = []

    print(f"[INFO] Evaluating {len(dataset)} images...")
    model_wrapper.eval()
    for images, target in tqdm(dataloader, desc="Collecting detections"):
        image_name = _normalize_image_name(target["image_name"][0])

        images = images.to(device)
        heatmap, clsmap = _split_stitcher_output(stitcher(images[0]))

        counts, locs, labels, scores, dscores = lmds((heatmap, clsmap))

        locs = locs[0]
        labels = labels[0]
        scores = scores[0]
        dscores = dscores[0]
        counts = counts[0]

        preds_xy = [(float(col), float(row)) for row, col in locs]
        pred_labels = [int(lbl) for lbl in labels]
        pred_scores = [float(s) for s in scores]

        for (x, y), lbl, score, dscore in zip(preds_xy, pred_labels, pred_scores, dscores):
            detections.append(
                {
                    "images": image_name,
                    "x": x,
                    "y": y,
                    "labels": lbl,
                    "scores": score,
                    "det_score": float(dscore),
                }
            )

        gt_points = _to_numpy_points(target["points"])
        gt_labels = _to_numpy_labels(target["labels"])
        gt_coords = [(float(x), float(y)) for x, y in gt_points]

        metrics.feed(
            gt={"loc": gt_coords, "labels": gt_labels},
            preds={"loc": preds_xy, "labels": pred_labels, "scores": pred_scores},
            est_count=counts,
        )

    metrics_per_class = metrics.copy()
    metrics.aggregate()

    overall = {
        "precision": metrics.precision(),
        "recall": metrics.recall(),
        "f1_score": metrics.fbeta_score(),
        "mae": metrics.mae(),
        "rmse": metrics.rmse(),
        "mse": metrics.mse(),
        "accuracy": metrics.accuracy(),
    }

    per_class = {}
    for class_id, class_name in classes.items():
        per_class[class_name] = {
            "precision": metrics_per_class.precision(class_id),
            "recall": metrics_per_class.recall(class_id),
            "f1_score": metrics_per_class.fbeta_score(class_id),
            "mae": metrics_per_class.mae(class_id),
            "rmse": metrics_per_class.rmse(class_id),
        }

    output_dir = config.output_dir or config.checkpoint.parent / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = config.detections_csv or (output_dir / "detections.csv")
    export_detections_csv(detections_path, detections)

    metrics_summary = {
        "overall": overall,
        "per_class": per_class,
        "classes": classes,
        "checkpoint": str(config.checkpoint),
        "csv": str(config.csv),
    }

    if config.metrics_json:
        metrics_path = config.metrics_json
    else:
        metrics_path = output_dir / "metrics.json"

    import json

    metrics_path.write_text(json.dumps(metrics_summary, indent=2))

    print("\n" + "=" * 70)
    print("HerdNet Evaluation Summary")
    print("=" * 70)
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1_score']:.4f}")
    print(f"MAE:       {overall['mae']:.4f}")
    print(f"RMSE:      {overall['rmse']:.4f}")
    print(f"\nDetections: {detections_path}")
    print(f"Metrics:    {metrics_path}")
    print("=" * 70 + "\n")

    return EvalResult(
        detections_csv=detections_path,
        metrics=metrics_summary,
        output_dir=output_dir,
    )

