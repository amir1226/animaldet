"""Helpers for exporting RF-DETR detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

DEFAULT_CATEGORIES = [
    {"supercategory": "animal", "id": 1, "name": "Alcelaphinae"},
    {"supercategory": "animal", "id": 2, "name": "Buffalo"},
    {"supercategory": "animal", "id": 3, "name": "Kob"},
    {"supercategory": "animal", "id": 4, "name": "Warthog"},
    {"supercategory": "animal", "id": 5, "name": "Waterbuck"},
    {"supercategory": "animal", "id": 6, "name": "Elephant"},
]

__all__ = ["DetectionSample", "DEFAULT_CATEGORIES", "write_coco_predictions"]


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    label: int
    score: float


@dataclass
class DetectionSample:
    file_name: str
    width: int
    height: int
    detections: List[Detection]


def write_coco_predictions(
    samples: Iterable[DetectionSample],
    output_path: Path | str,
    *,
    categories: Iterable[dict] = DEFAULT_CATEGORIES,
) -> None:
    """Persist detections into a COCO-style JSON file."""
    output_path = Path(output_path)
    coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": list(categories),
    }

    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        coco["images"].append(
            {
                "license": 0,
                "file_name": sample.file_name,
                "coco_url": "none",
                "height": sample.height,
                "width": sample.width,
                "date_captured": "none",
                "flickr_url": "none",
                "id": image_id,
            }
        )

        for det in sample.detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            area = max(width, 0.0) * max(height, 0.0)
            coco["annotations"].append(
                {
                    "segmentation": [[]],
                    "area": float(area),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "category_id": int(det.label),
                    "id": annotation_id,
                    "score": float(det.score),
                }
            )
            annotation_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))

