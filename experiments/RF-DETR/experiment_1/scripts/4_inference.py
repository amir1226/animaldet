#!/usr/bin/env python3
"""Self-contained RF-DETR inference script."""

import torch
from rfdetr.detr import RFDETRSmall
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torchvision
import sys

# Embedded configuration
cfg = OmegaConf.create({
    "data": {
        "test_root": "data/herdnet/raw/test",
    },
    "inference": {
        "device": "cuda",
        "checkpoint_path": "./checkpoint_phase_2.pth",
        "threshold": 0.1,
        "batch_size": 4,
        "output_path": "./results",
        "detections_csv": "rfdetr_detections.csv",
        "results_csv": "rfdetr_scores.csv",
    }
})

class SimpleStitcher:
    """Simple image stitcher for RF-DETR inference on large images."""

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int = 512,
        overlap: int = 0,
        batch_size: int = 4,
        confidence_threshold: float = 0.1,
        device: str = "cuda",
    ):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.model.to(device)
        self.model.eval()

    def _make_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Divide image into patches.

        Args:
            image: Image tensor [C, H, W]

        Returns:
            patches: Tensor of patches [N, C, patch_size, patch_size]
            positions: List of (y_offset, x_offset) for each patch
        """
        C, H, W = image.shape
        stride = self.patch_size - self.overlap

        patches = []
        positions = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Extract patch
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)

                patch = image[:, y:y_end, x:x_end]

                # Pad if needed
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded = torch.zeros(C, self.patch_size, self.patch_size, dtype=image.dtype)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded

                patches.append(patch)
                positions.append((y, x))

        return torch.stack(patches), positions

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference on image with stitching.

        Args:
            image: Image tensor [C, H, W]

        Returns:
            Dictionary with 'boxes', 'scores', 'labels'
        """
        # Get patches
        patches, positions = self._make_patches(image)

        # Run inference on patches
        all_boxes = []
        all_scores = []
        all_labels = []

        dataset = TensorDataset(patches)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        patch_idx = 0
        for batch in dataloader:
            batch_patches = batch[0].to(self.device)
            outputs = self.model(batch_patches)

            for i in range(len(batch_patches)):
                y_offset, x_offset = positions[patch_idx]
                patch_idx += 1

                # Extract predictions
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]

                # Convert to scores and labels
                pred_scores = pred_logits.sigmoid()
                scores, labels = pred_scores.max(dim=-1)
                labels = labels + 1  # Convert to 1-indexed

                # Filter by confidence
                keep = scores > self.confidence_threshold
                boxes = pred_boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                if len(boxes) == 0:
                    continue

                # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
                cx, cy, w, h = boxes.unbind(-1)
                x1 = (cx - 0.5 * w) * self.patch_size + x_offset
                y1 = (cy - 0.5 * h) * self.patch_size + y_offset
                x2 = (cx + 0.5 * w) * self.patch_size + x_offset
                y2 = (cy + 0.5 * h) * self.patch_size + y_offset
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                all_boxes.append(boxes.cpu())
                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        if len(all_boxes) == 0:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }

        # Concatenate all detections
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

# Load checkpoint
checkpoint = torch.load(cfg.inference.checkpoint_path, weights_only=False)
state_dict = checkpoint.get('model', checkpoint.get('ema_model'))
num_classes = state_dict['class_embed.weight'].shape[0] - 1

# Create model with defaults and extract the PyTorch module
rfdetr_wrapper = RFDETRSmall(num_classes=num_classes)
model = rfdetr_wrapper.model.model
model.load_state_dict(state_dict, strict=True)
model = model.to(cfg.inference.device)
model.eval()

# Setup stitcher
stitcher = SimpleStitcher(
    model=model,
    patch_size=512,
    overlap=0,
    batch_size=cfg.inference.batch_size,
    confidence_threshold=cfg.inference.threshold,
    device=cfg.inference.device,
)

# Load images
test_root = Path(cfg.data.test_root)
image_files = (
    list(test_root.glob('*.jpg')) +
    list(test_root.glob('*.JPG')) +
    list(test_root.glob('*.png')) +
    list(test_root.glob('*.PNG'))
)

# Transforms
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Run inference
all_detections = []
for img_path in tqdm(image_files, desc="Inference"):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image=np.array(image))['image']

    detections = stitcher(image_tensor)

    for i in range(len(detections['scores'])):
        all_detections.append({
            'images': img_path.name,
            'x': float(detections['boxes'][i, 0]),
            'y': float(detections['boxes'][i, 1]),
            'x_max': float(detections['boxes'][i, 2]),
            'y_max': float(detections['boxes'][i, 3]),
            'labels': int(detections['labels'][i]),
            'scores': float(detections['scores'][i]),
        })

# Save
output_path = Path(cfg.inference.output_path)
output_path.mkdir(parents=True, exist_ok=True)
pd.DataFrame(all_detections).to_csv(output_path / cfg.inference.detections_csv, index=False)
print(f"✓ Saved {len(all_detections)} detections")
