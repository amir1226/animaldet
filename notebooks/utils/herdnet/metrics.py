"""Evaluate detection CSVs with HerdNet-style metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from animaloc.eval.metrics import PointsMetrics

DEFAULT_CLASSES = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}

__all__ = ["load_class_map", "evaluate_points_from_csv"]


def load_class_map(path: Optional[Path | str]) -> Dict[int, str]:
    """Load class-id to name mapping from JSON; default to HerdNet classes."""
    if path is None:
        return DEFAULT_CLASSES
    path = Path(path)
    data = json.loads(path.read_text())
    return {int(k): str(v) for k, v in data.items()}


def _extract_points(df: pd.DataFrame):
    coords = [(float(row["x"]), float(row["y"])) for _, row in df.iterrows()]
    labels = [int(row["labels"]) for _, row in df.iterrows()]
    scores = [float(row["scores"]) for _, row in df.iterrows()] if "scores" in df.columns else []
    return coords, labels, scores


def evaluate_points_from_csv(
    gt_csv: Path | str,
    detections_csv: Path | str,
    *,
    class_map_path: Optional[Path | str] = None,
    radius: float = 20.0,
    output_json: Optional[Path | str] = None,
) -> Dict:
    """Compute HerdNet metrics from ground-truth and detection CSVs."""
    gt_csv = Path(gt_csv)
    detections_csv = Path(detections_csv)
    class_map = load_class_map(class_map_path)
    num_classes = len(class_map) + 1

    gt_df = pd.read_csv(gt_csv)
    det_df = pd.read_csv(detections_csv)

    metrics = PointsMetrics(radius=radius, num_classes=num_classes)

    image_names = sorted(set(gt_df["images"]) | set(det_df["images"]))
    for image_name in image_names:
        gt_rows = gt_df[gt_df["images"] == image_name]
        det_rows = det_df[det_df["images"] == image_name]

        gt_coords, gt_labels, _ = _extract_points(gt_rows)
        pred_coords, pred_labels, pred_scores = _extract_points(det_rows)

        est_count = [pred_labels.count(cls_id) for cls_id in range(1, num_classes)]
        metrics.feed(
            gt={"loc": gt_coords, "labels": gt_labels},
            preds={"loc": pred_coords, "labels": pred_labels, "scores": pred_scores},
            est_count=est_count,
        )

    per_class_metrics = metrics.copy()
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
    for class_id, class_name in class_map.items():
        per_class[class_name] = {
            "precision": per_class_metrics.precision(class_id),
            "recall": per_class_metrics.recall(class_id),
            "f1_score": per_class_metrics.fbeta_score(class_id),
            "mae": per_class_metrics.mae(class_id),
            "rmse": per_class_metrics.rmse(class_id),
        }

    summary = {
        "overall": overall,
        "per_class": per_class,
        "radius": radius,
        "gt_csv": str(gt_csv),
        "detections_csv": str(detections_csv),
    }

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    return summary

