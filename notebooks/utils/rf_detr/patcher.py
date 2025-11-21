"""COCO patch generation helpers for RF-DETR notebooks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

__all__ = ["PatchSummary", "generate_patch_dataset"]


@dataclass
class PatchSummary:
    images_processed: int
    patches_created: int
    annotations_original: int
    annotations_patches: int
    output_dir: Path
    output_json: Path


def _load_coco(json_path: Path) -> Dict:
    with json_path.open("r") as handle:
        return json.load(handle)


def _save_coco(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2)


def _calculate_patch_positions(
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
    overlap: int,
) -> List[Tuple[int, int, int, int]]:
    stride_x = max(1, patch_width - overlap)
    stride_y = max(1, patch_height - overlap)
    patches: List[Tuple[int, int, int, int]] = []

    y = 0
    while y < height:
        x = 0
        while x < width:
            x_max = min(x + patch_width, width)
            y_max = min(y + patch_height, height)
            x_min = max(0, x_max - patch_width)
            y_min = max(0, y_max - patch_height)
            patches.append((x_min, y_min, x_max, y_max))
            if x_max == width:
                break
            x += stride_x
        if y_max == height:
            break
        y += stride_y
    return patches


def _bbox_intersection(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _coco_to_xyxy(bbox: Iterable[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _xyxy_to_coco(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def generate_patch_dataset(
    images_dir: Path | str,
    json_file: Path | str,
    output_dir: Path | str,
    *,
    patch_width: int,
    patch_height: int,
    overlap: int,
    min_visibility: float = 0.1,
    include_background_category: bool = True,
) -> PatchSummary:
    """Split large images into patches, adjusting COCO annotations."""

    images_dir = Path(images_dir)
    json_file = Path(json_file)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not json_file.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_file}")

    coco = _load_coco(json_file)
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": list(coco.get("categories", [])),
    }

    if include_background_category:
        new_coco["categories"] = list(new_coco["categories"])
        new_coco["categories"].append(
            {"id": 0, "name": "Background", "supercategory": "none"}
        )

    new_image_id = 1
    new_annotation_id = 1

    for img_info in tqdm(coco.get("images", []), desc="Patching images"):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = images_dir / file_name
        if not img_path.exists():
            print(f"[WARN] Missing image: {img_path}")
            continue

        image = Image.open(img_path)
        width, height = image.size

        patch_positions = _calculate_patch_positions(
            width, height, patch_width, patch_height, overlap
        )

        image_anns = annotations_by_image.get(img_id, [])
        for idx, patch_pos in enumerate(patch_positions):
            patch_xmin, patch_ymin, patch_xmax, patch_ymax = patch_pos
            patch_box = (patch_xmin, patch_ymin, patch_xmax, patch_ymax)

            patch_annotations = []
            for ann in image_anns:
                intersection = _bbox_intersection(_coco_to_xyxy(ann["bbox"]), patch_box)
                if intersection is None:
                    continue
                inter_area = _bbox_area(intersection)
                visibility = inter_area / max(_bbox_area(_coco_to_xyxy(ann["bbox"])), 1e-6)
                if visibility < min_visibility:
                    continue

                patch_bbox = (
                    intersection[0] - patch_xmin,
                    intersection[1] - patch_ymin,
                    intersection[2] - patch_xmin,
                    intersection[3] - patch_ymin,
                )
                patch_annotations.append(
                    {
                        "id": new_annotation_id,
                        "image_id": new_image_id,
                        "category_id": ann["category_id"],
                        "bbox": _xyxy_to_coco(patch_bbox),
                        "area": _bbox_area(patch_bbox),
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                new_annotation_id += 1

            if not patch_annotations:
                continue

            base_name, ext = os.path.splitext(file_name)
            patch_filename = f"{base_name}_{idx}{ext}"
            patch_image = image.crop(patch_pos)
            if patch_image.size != (patch_width, patch_height):
                padded = Image.new("RGB", (patch_width, patch_height), color=(0, 0, 0))
                padded.paste(patch_image, (0, 0))
                patch_image = padded

            output_dir.mkdir(parents=True, exist_ok=True)
            patch_image.save(output_dir / patch_filename)

            new_coco["images"].append(
                {
                    "id": new_image_id,
                    "file_name": patch_filename,
                    "width": patch_width,
                    "height": patch_height,
                    "base_image": file_name,
                    "base_image_width": width,
                    "base_image_height": height,
                    "patch_position": patch_pos,
                }
            )
            new_coco["annotations"].extend(patch_annotations)
            new_image_id += 1

    output_json = output_dir / "_annotations.coco.json"
    _save_coco(new_coco, output_json)

    return PatchSummary(
        images_processed=len(coco.get("images", [])),
        patches_created=len(new_coco["images"]),
        annotations_original=len(coco.get("annotations", [])),
        annotations_patches=len(new_coco["annotations"]),
        output_dir=output_dir,
        output_json=output_json,
    )

