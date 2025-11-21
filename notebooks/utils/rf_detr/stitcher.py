"""Sliding-window inference helpers for RF-DETR."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image

TensorLike = Union[torch.Tensor, str, Path]

__all__ = ["SimpleStitcher"]


class SimpleStitcher:
    """Minimal sliding-window stitcher for RF-DETR models."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        patch_size: int = 512,
        overlap: int = 0,
        batch_size: int = 4,
        confidence_threshold: float = 0.5,
        device: str | torch.device = "cuda",
        label_offset: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)
        self.label_offset = label_offset
        self.model.eval()

    def _make_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        stride = self.patch_size - self.overlap
        patches: List[torch.Tensor] = []
        positions: List[Tuple[int, int]] = []
        channels, height, width = image.shape

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_end = min(y + self.patch_size, height)
                x_end = min(x + self.patch_size, width)
                patch = image[:, y:y_end, x:x_end]
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded = torch.zeros(
                        channels, self.patch_size, self.patch_size, dtype=image.dtype, device=image.device
                    )
                    padded[:, : patch.shape[1], : patch.shape[2]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((y, x))

        stacked = torch.stack(patches) if patches else torch.empty(0, channels, self.patch_size, self.patch_size)
        return stacked, positions

    def _load_tensor(self, image: TensorLike) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image.detach().cpu()
        data = read_image(str(image)).float() / 255.0
        return data

    @torch.no_grad()
    def __call__(self, image: TensorLike) -> Dict[str, torch.Tensor]:
        tensor = self._load_tensor(image)
        patches, positions = self._make_patches(tensor)
        if patches.numel() == 0:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
            }

        dataset = TensorDataset(patches)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        collected_boxes: List[torch.Tensor] = []
        collected_scores: List[torch.Tensor] = []
        collected_labels: List[torch.Tensor] = []

        patch_idx = 0
        for batch in loader:
            batch_patches = batch[0].to(self.device)
            outputs = self.model(batch_patches)
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]

            for i in range(len(batch_patches)):
                y_off, x_off = positions[patch_idx]
                patch_idx += 1

                pred_scores = logits[i].sigmoid()
                scores, labels = pred_scores.max(dim=-1)
                keep = scores > self.confidence_threshold
                if keep.sum() == 0:
                    continue

                kept_boxes = boxes[i][keep]
                kept_scores = scores[keep].detach()
                kept_labels = labels[keep].to(torch.long) + self.label_offset

                cx, cy, w, h = kept_boxes.unbind(-1)
                x1 = (cx - 0.5 * w) * self.patch_size + x_off
                y1 = (cy - 0.5 * h) * self.patch_size + y_off
                x2 = (cx + 0.5 * w) * self.patch_size + x_off
                y2 = (cy + 0.5 * h) * self.patch_size + y_off
                xyxy = torch.stack([x1, y1, x2, y2], dim=-1).detach().cpu()

                collected_boxes.append(xyxy)
                collected_scores.append(kept_scores.detach().cpu())
                collected_labels.append(kept_labels.detach().cpu())

        if not collected_boxes:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
            }

        return {
            "boxes": torch.cat(collected_boxes),
            "scores": torch.cat(collected_scores),
            "labels": torch.cat(collected_labels),
        }
