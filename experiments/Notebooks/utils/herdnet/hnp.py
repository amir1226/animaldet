"""Hard Negative Patch (HNP) generation for HerdNet Stage 2."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import PadIfNeeded
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from animaloc.data import PatchesBuffer
from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher
from animaloc.eval.metrics import PointsMetrics
from animaloc.models import HerdNet, LossWrapper
from animaloc.train.losses import FocalLoss

__all__ = ["HNPConfig", "HNPResult", "generate_hard_negative_patches"]

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
class HNPConfig:
    """Configuration for Hard Negative Patch generation."""

    # Checkpoint and data
    checkpoint: Path
    train_csv: Path
    train_root: Path
    output_root: Path

    # Inference
    device: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 4

    # Patch generation
    patch_size: int = 512
    patch_overlap: int = 160
    min_score: float = 0.0

    # LMDS parameters
    kernel_size: Tuple[int, int] = (3, 3)
    adapt_ts: float = 0.3
    neg_ts: float = 0.1

    # Output
    detections_csv: Optional[Path] = None

    def __post_init__(self):
        self.checkpoint = Path(self.checkpoint)
        self.train_csv = Path(self.train_csv)
        self.train_root = Path(self.train_root)
        self.output_root = Path(self.output_root)
        if self.detections_csv:
            self.detections_csv = Path(self.detections_csv)


@dataclass
class HNPResult:
    """Results from HNP generation."""

    hnp_patches_created: int
    detections_csv: Path
    output_root: Path
    reference_csv: Path  # gt.csv (for reference only, should be discarded)


def _normalize_image_column(value: str) -> str:
    """Parse image column that may be a string representation of list/tuple."""
    if isinstance(value, str) and (value.startswith("[") or value.startswith("(")):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return str(value)


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
            "loss": torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weights),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    wrapper = LossWrapper(base_model, losses=losses)
    wrapper = wrapper.to(device)
    return wrapper


def _collect_detections(
    checkpoint: Dict,
    dataloader: DataLoader,
    device: torch.device,
    down_ratio: int,
    patch_size: int,
    overlap: int,
    kernel_size: Tuple[int, int],
    adapt_ts: float,
    neg_ts: float,
    upsample: bool = True,
) -> List[Dict[str, object]]:
    """Run inference and collect all detections (TPs + FPs)."""
    classes = {
        int(k): str(v) for k, v in checkpoint.get("classes", DEFAULT_CLASSES).items()
    }
    num_classes = len(classes) + 1

    model_wrapper = _build_model(num_classes, device)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    model_wrapper.eval()

    stitcher = HerdNetStitcher(
        model=model_wrapper,
        size=(patch_size, patch_size),
        overlap=overlap,
        down_ratio=down_ratio,
        reduction="mean",
        up=upsample,
        device_name=device.type,
    )

    metrics = PointsMetrics(5, num_classes=num_classes)

    evaluator = HerdNetEvaluator(
        model=model_wrapper,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        lmds_kwargs={
            "kernel_size": kernel_size,
            "adapt_ts": adapt_ts,
            "neg_ts": neg_ts,
        },
        device_name=device.type,
        print_freq=10,
        work_dir=None,
        header="[HNP Generation]",
    )

    print("[INFO] Running inference with HerdNetEvaluator...")
    evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

    detections_df = evaluator.detections
    detections_df.dropna(inplace=True)

    detections = []
    for _, row in detections_df.iterrows():
        detections.append(
            {
                "images": str(row["images"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "labels": int(row["labels"]),
                "scores": float(row["scores"]) if "scores" in row else 1.0,
                "det_score": (
                    float(row["det_score"])
                    if "det_score" in row
                    else float(row["scores"])
                    if "scores" in row
                    else 1.0
                ),
            }
        )

    print(f"[INFO] Collected {len(detections)} detections")
    return detections


def _export_detections(path: Path, records: Iterable[Dict[str, object]]) -> None:
    """Export detections to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["images", "x", "y", "labels", "scores", "det_score"]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _create_hnp_patches(
    train_root: Path,
    patch_size: int,
    overlap: int,
    detections_csv: Path,
    dest_dir: Path,
    min_score: float,
) -> int:
    """Generate HNP patches from model detections."""
    detections = pd.read_csv(detections_csv)
    detections["images"] = detections["images"].apply(_normalize_image_column)

    if "scores" in detections.columns:
        detections = detections[detections["scores"] >= min_score]

    if detections.empty:
        print("[WARN] No detections found; no HNP patches will be generated.")
        return 0

    detections_path = detections_csv
    if min_score > 0:
        detections_path = detections_csv.parent / f"{detections_csv.stem}_filtered.csv"
        detections.to_csv(detections_path, index=False)

    dest_dir.mkdir(parents=True, exist_ok=True)

    buffer = PatchesBuffer(
        str(detections_path),
        str(train_root),
        (patch_size, patch_size),
        overlap=overlap,
        min_visibility=0.0,
    ).buffer

    buffer.drop(columns="limits").to_csv(dest_dir / "gt.csv", index=False)
    print(
        f"[INFO] Generated gt.csv with {len(buffer)} entries "
        "(for reference only, will be discarded)"
    )

    source_images = detections["images"].unique()
    image_paths = [train_root / img for img in source_images]

    padder = PadIfNeeded(
        patch_size,
        patch_size,
        position=PadIfNeeded.PositionType.TOP_LEFT,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )

    patch_count = 0
    for img_path in tqdm(image_paths, desc="Saving HNP patches"):
        pil_img = Image.open(img_path)
        img_name = img_path.name

        img_buffer = buffer[buffer["base_images"] == img_name]
        for row in img_buffer[["images", "limits"]].to_numpy().tolist():
            patch_name, limits = row
            cropped = np.array(pil_img.crop(limits.get_tuple))
            padded = Image.fromarray(padder(image=cropped)["image"])
            padded.save(dest_dir / patch_name)
            patch_count += 1

    return patch_count


def generate_hard_negative_patches(config: HNPConfig) -> HNPResult:
    """Generate Hard Negative Patches for HerdNet Stage 2 training.

    This function:
    1. Loads a Stage 1 checkpoint
    2. Runs inference on full training images
    3. Extracts patches around all detections (TPs + FPs)
    4. Saves patches to output directory

    The generated patches should be merged with original training patches.
    Use the original training CSV (not the generated gt.csv) for Stage 2 training.

    Args:
        config: HNP generation configuration

    Returns:
        HNPResult with patch count and output paths
    """
    device = torch.device(
        config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(config.checkpoint, map_location=device)
    mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(checkpoint.get("std", DEFAULT_STD))

    dataset = CSVDataset(
        csv_file=str(config.train_csv),
        root_dir=str(config.train_root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=2, anno_type="point")],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    detections = _collect_detections(
        checkpoint=checkpoint,
        dataloader=dataloader,
        device=device,
        down_ratio=2,
        patch_size=config.patch_size,
        overlap=config.patch_overlap,
        kernel_size=config.kernel_size,
        adapt_ts=config.adapt_ts,
        neg_ts=config.neg_ts,
        upsample=True,
    )

    config.output_root.mkdir(parents=True, exist_ok=True)

    detections_csv = config.detections_csv or (config.output_root / "detections.csv")
    _export_detections(detections_csv, detections)
    print(f"[INFO] Stored detections CSV at {detections_csv} ({len(detections)} total)")
    print("[INFO] This includes both True Positives and False Positives (not filtered)")

    hnp_count = _create_hnp_patches(
        train_root=config.train_root,
        patch_size=config.patch_size,
        overlap=config.patch_overlap,
        detections_csv=detections_csv,
        dest_dir=config.output_root,
        min_score=config.min_score,
    )

    reference_csv = config.output_root / "gt.csv"

    print(f"\n{'=' * 80}")
    print(f"[SUCCESS] Generated {hnp_count} HNP patches in {config.output_root}")
    print(f"[INFO] Detections CSV: {detections_csv}")
    print(f"[INFO] HNP patches: {config.output_root}/*.JPG")
    print(f"[INFO] gt.csv: {reference_csv} (DISCARD THIS - use original train CSV)")
    print(f"{'=' * 80}")

    return HNPResult(
        hnp_patches_created=hnp_count,
        detections_csv=detections_csv,
        output_root=config.output_root,
        reference_csv=reference_csv,
    )

