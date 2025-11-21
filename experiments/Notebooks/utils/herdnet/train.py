"""Training functions for HerdNet Stage 1 and Stage 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from animaloc.data.transforms import (
    DownSample,
    FIDT,
    MultiTransformsWrapper,
    PointsToMask,
)
from animaloc.datasets import CSVDataset, FolderDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher, PointsMetrics
from animaloc.models import HerdNet, LossWrapper
from animaloc.train import Trainer
from animaloc.train.losses import FocalLoss
from animaloc.utils.seed import set_seed

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train_stage1",
    "train_stage2",
    "load_backbone_weights",
    "enrich_checkpoint_metadata",
]

DEFAULT_CLASS_MAP = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}

DEFAULT_CLASS_WEIGHTS = {
    "hartebeest": 1.0,
    "buffalo": 2.0,
    "kob": 1.0,
    "warthog": 6.0,
    "waterbuck": 12.0,
    "elephant": 1.0,
}

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Configuration for HerdNet training."""

    # Paths
    train_root: Path
    train_csv: Path
    val_root: Path
    val_csv: Path
    work_dir: Path

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    num_workers: int = 4
    valid_freq: int = 1

    # Model parameters
    patch_size: int = 512
    down_ratio: int = 2
    num_layers: int = 34
    head_conv: int = 64

    # Preprocessing
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD

    # Class configuration
    class_map: Optional[Dict[int, str]] = None
    class_weights: Optional[Dict[str, float]] = None

    # WandB logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "disabled"
    wandb_run_name: str = "herdnet"

    # Pretrained backbone
    pretrained_backbone: str = "dla34.in1k"

    # Evaluation
    stitch_overlap: int = 160

    def __post_init__(self):
        self.train_root = Path(self.train_root)
        self.train_csv = Path(self.train_csv)
        self.val_root = Path(self.val_root)
        self.val_csv = Path(self.val_csv)
        self.work_dir = Path(self.work_dir)

        if self.class_map is None:
            self.class_map = DEFAULT_CLASS_MAP
        if self.class_weights is None:
            self.class_weights = DEFAULT_CLASS_WEIGHTS


@dataclass
class TrainResult:
    """Results from HerdNet training."""

    best_checkpoint: Path
    latest_checkpoint: Path
    work_dir: Path
    final_metrics: Optional[Dict] = None


def load_backbone_weights(model: torch.nn.Module, timm_model_id: str) -> None:
    """Load pretrained DLA backbone from timm.

    Args:
        model: The base network module (e.g., model.base_0)
        timm_model_id: timm model identifier (e.g., 'dla34.in1k')
    """
    if timm_model_id.lower() == "none":
        print("[INFO] Skipping backbone pretraining")
        return

    try:
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError("timm is required to load pretrained backbone") from exc

    print(f"[INFO] Loading backbone from timm: {timm_model_id}")
    timm_model = timm.create_model(timm_model_id, pretrained=True)
    state_dict = timm_model.state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    del timm_model


def enrich_checkpoint_metadata(
    checkpoint_path: Path,
    class_map: Dict[int, str],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    stage: str,
) -> None:
    """Add metadata to a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pth file
        class_map: Mapping from class ID to class name
        mean: Normalization mean
        std: Normalization std
        stage: Training stage identifier (e.g., 'stage1', 'stage2')
    """
    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["classes"] = class_map
    checkpoint["mean"] = list(mean)
    checkpoint["std"] = list(std)
    checkpoint["stage"] = stage

    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Enriched checkpoint: {checkpoint_path.name}")


def _init_wandb(
    project: Optional[str],
    entity: Optional[str],
    mode: str,
    run_name: str,
    config: Dict,
) -> bool:
    """Initialize Weights & Biases logging."""
    if not project or mode == "disabled":
        return False

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is not installed but logging was requested") from exc

    wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=run_name,
        config=config,
    )
    return True


def train_stage1(config: TrainConfig) -> TrainResult:
    """Train HerdNet Stage 1 on patches.

    Args:
        config: Training configuration

    Returns:
        TrainResult with checkpoint paths and metrics
    """
    set_seed(9292)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.work_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(config.class_map) + 1  # + background

    # Datasets
    train_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(mean=config.mean, std=config.std, p=1.0),
    ]
    train_end_transforms = [
        MultiTransformsWrapper(
            [
                FIDT(num_classes=2, add_bg=False, down_ratio=config.down_ratio),
                PointsToMask(
                    radius=2,
                    num_classes=num_classes,
                    squeeze=True,
                    down_ratio=32,
                ),
            ]
        )
    ]

    val_transforms = [A.Normalize(mean=config.mean, std=config.std, p=1.0)]
    val_end_transforms = [DownSample(down_ratio=config.down_ratio, anno_type="point")]

    train_dataset = CSVDataset(
        csv_file=str(config.train_csv),
        root_dir=str(config.train_root),
        albu_transforms=train_transforms,
        end_transforms=train_end_transforms,
    )
    val_dataset = CSVDataset(
        csv_file=str(config.val_csv),
        root_dir=str(config.val_root),
        albu_transforms=val_transforms,
        end_transforms=val_end_transforms,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for stitcher
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = HerdNet(
        num_classes=num_classes,
        down_ratio=config.down_ratio,
        num_layers=config.num_layers,
        head_conv=config.head_conv,
        pretrained=False,
    ).to(device)

    load_backbone_weights(model.base_0, config.pretrained_backbone)

    # Loss weights
    per_class_weights = [
        config.class_weights.get(config.class_map[idx + 1].lower(), 1.0)
        for idx in range(len(config.class_map))
    ]
    weight_vector = [0.1, *per_class_weights]
    class_weights = torch.tensor(weight_vector, dtype=torch.float32, device=device)

    losses = [
        {
            "loss": FocalLoss(reduction="mean", normalize=False),
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
    model = LossWrapper(model, losses=losses).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Evaluator
    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    stitcher = HerdNetStitcher(
        model=model,
        size=(config.patch_size, config.patch_size),
        overlap=config.stitch_overlap,
        down_ratio=config.down_ratio,
        reduction="mean",
        up=False,
    )
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=val_loader,
        metrics=metrics,
        stitcher=stitcher,
        work_dir=str(config.work_dir),
        header="validation",
        device_name=device.type,
        lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 0.3},
        print_freq=10,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=config.epochs,
        evaluator=evaluator,
        work_dir=str(config.work_dir),
        print_freq=100,
        valid_freq=config.valid_freq,
        device_name=device.type,
        auto_lr={
            "mode": "max",
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 10,
            "min_lr": 1e-6,
        },
    )

    # WandB
    wandb_flag = _init_wandb(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        run_name=config.wandb_run_name,
        config={
            "stage": "stage1",
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "down_ratio": config.down_ratio,
        },
    )

    trainer.start(
        warmup_iters=100,
        checkpoints="best",
        select="max",
        validate_on="f1_score",
        wandb_flag=wandb_flag,
    )

    best_path = config.work_dir / "best_model.pth"
    latest_path = config.work_dir / "latest_model.pth"
    enrich_checkpoint_metadata(best_path, config.class_map, config.mean, config.std, "stage1")
    enrich_checkpoint_metadata(latest_path, config.class_map, config.mean, config.std, "stage1")

    return TrainResult(
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        work_dir=config.work_dir,
    )


def train_stage2(
    config: TrainConfig,
    stage1_checkpoint: Path,
    *,
    learning_rate: float = 1e-6,
) -> TrainResult:
    """Train HerdNet Stage 2 with Hard Negative Patches.

    Args:
        config: Training configuration
        stage1_checkpoint: Path to Stage 1 best checkpoint
        learning_rate: Learning rate for stage 2 (typically lower than stage 1)

    Returns:
        TrainResult with checkpoint paths and metrics
    """
    set_seed(9292)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.work_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(stage1_checkpoint, map_location=device)

    if "classes" in checkpoint:
        class_map = {int(k): str(v) for k, v in checkpoint["classes"].items()}
    else:
        class_map = config.class_map

    mean = tuple(checkpoint.get("mean", config.mean))
    std = tuple(checkpoint.get("std", config.std))
    num_classes = len(class_map) + 1

    # Model
    model = HerdNet(
        num_classes=num_classes,
        down_ratio=config.down_ratio,
        num_layers=config.num_layers,
        head_conv=config.head_conv,
        pretrained=False,
    )

    class_weights_list = [
        config.class_weights.get(class_map[idx + 1].lower(), 1.0)
        for idx in range(len(class_map))
    ]
    weight_vector = [0.1, *class_weights_list]
    weight_tensor = torch.tensor(weight_vector, dtype=torch.float32, device=device)

    losses = [
        {
            "loss": FocalLoss(reduction="mean"),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": torch.nn.CrossEntropyLoss(reduction="mean", weight=weight_tensor),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    model = LossWrapper(model, losses=losses).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Datasets - use FolderDataset for train (handles HNPs as background)
    train_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(mean=mean, std=std, p=1.0),
    ]
    train_end_transforms = [
        MultiTransformsWrapper(
            [
                FIDT(num_classes=num_classes, add_bg=False, down_ratio=config.down_ratio),
                PointsToMask(
                    radius=2,
                    num_classes=num_classes,
                    squeeze=True,
                    down_ratio=int(config.patch_size // 16),
                ),
            ]
        )
    ]

    train_dataset = FolderDataset(
        csv_file=str(config.train_csv),
        root_dir=str(config.train_root),
        albu_transforms=train_transforms,
        end_transforms=train_end_transforms,
    )

    val_dataset = CSVDataset(
        csv_file=str(config.val_csv),
        root_dir=str(config.val_root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=config.down_ratio, anno_type="point")],
    )

    print(f"[INFO] Training samples: {len(train_dataset)} (includes HNPs as background)")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # Must be 1
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # Evaluator
    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    stitcher = HerdNetStitcher(
        model=model,
        size=(config.patch_size, config.patch_size),
        overlap=config.stitch_overlap,
        down_ratio=config.down_ratio,
        reduction="mean",
        up=False,
        device_name=device.type,
    )
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=val_loader,
        metrics=metrics,
        stitcher=stitcher,
        work_dir=str(config.work_dir),
        header="validation",
        device_name=device.type,
        lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 0.3},
        print_freq=10,
    )

    optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=config.epochs,
        evaluator=evaluator,
        work_dir=str(config.work_dir),
        print_freq=100,
        valid_freq=config.valid_freq,
        device_name=device.type,
        auto_lr={
            "mode": "max",
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 10,
            "min_lr": 1e-6,
        },
    )

    wandb_flag = _init_wandb(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        run_name=config.wandb_run_name,
        config={
            "stage": "stage2",
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": learning_rate,
            "down_ratio": config.down_ratio,
            "total_patches": len(train_dataset),
        },
    )

    trainer.start(
        warmup_iters=1,
        checkpoints="best",
        select="max",
        validate_on="f1_score",
        wandb_flag=wandb_flag,
    )

    best_path = config.work_dir / "best_model.pth"
    latest_path = config.work_dir / "latest_model.pth"
    enrich_checkpoint_metadata(best_path, class_map, mean, std, "stage2")
    enrich_checkpoint_metadata(latest_path, class_map, mean, std, "stage2")

    return TrainResult(
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        work_dir=config.work_dir,
    )

