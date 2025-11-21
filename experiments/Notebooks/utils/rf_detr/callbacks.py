"""Callbacks tailored for RF-DETR notebooks."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

DEFAULT_CLASS_NAMES = [
    "Hartebeest",
    "Buffalo",
    "Kob",
    "Warthog",
    "Waterbuck",
    "Elephant",
]


class HerdNetMetricsCallback:
    """Evaluates RF-DETR predictions with HerdNet point-based metrics."""

    def __init__(
        self,
        model,
        val_dataset_path: str,
        val_images_dir: str,
        *,
        threshold_px: int = 20,
        eval_every_n_epochs: int = 1,
        confidence_threshold: float = 0.3,
        wandb_log: bool = True,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
        max_val_images: Optional[int] = None,
    ) -> None:
        self.model = model
        self.val_dataset_path = Path(val_dataset_path)
        self.val_images_dir = Path(val_images_dir)
        self.threshold_px = threshold_px
        self.eval_every_n_epochs = eval_every_n_epochs
        self.confidence_threshold = confidence_threshold
        self.wandb_log = wandb_log
        self.device = device
        self.max_val_images = max_val_images

        if not self.val_dataset_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_dataset_path}")
        if not self.val_images_dir.exists():
            raise FileNotFoundError(f"Validation images dir not found: {val_images_dir}")

        try:
            from pycocotools.coco import COCO
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("pycocotools is required for HerdNet metrics") from exc

        self.coco = COCO(str(self.val_dataset_path))
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.max_class_id = max((cat["id"] for cat in cats), default=0)
        num_classes = self.max_class_id + 1

        from animaloc.eval.metrics import PointsMetrics

        self.metrics = PointsMetrics(radius=threshold_px, num_classes=num_classes)

    def update(self, log_stats: Dict) -> Optional[Dict]:
        epoch = int(log_stats.get("epoch", 0))
        if epoch % self.eval_every_n_epochs != 0:
            return None

        print(f"\n{'=' * 70}\n[HerdNet Metrics] Evaluating at epoch {epoch}...\n{'=' * 70}")
        self.metrics.flush()
        metrics = self._evaluate()
        if self.wandb_log:
            self._log_to_wandb(metrics, epoch)
        self._print_metrics(metrics, epoch)
        return metrics

    def _evaluate(self) -> Dict:
        img_ids = self.coco.getImgIds()
        if self.max_val_images:
            img_ids = img_ids[: self.max_val_images]
            print(f"  Evaluating on {len(img_ids)} images (limited)")
        else:
            print(f"  Evaluating on {len(img_ids)} images")

        n_processed = 0
        n_skipped = 0

        for img_id in img_ids:
            info = self.coco.loadImgs(img_id)[0]
            img_path = self.val_images_dir / info["file_name"]
            if not img_path.exists():
                n_skipped += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                detections = self.model.predict(image, threshold=self.confidence_threshold)
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                if not anns:
                    continue

                gt_centers, gt_labels = [], []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    gt_centers.append((x + 0.5 * w, y + 0.5 * h))
                    gt_labels.append(ann["category_id"])

                pred_centers, pred_labels, pred_scores = [], [], []
                if len(detections.xyxy) > 0:
                    for box, cls_id, score in zip(
                        detections.xyxy, detections.class_id, detections.confidence
                    ):
                        cx = (box[0] + box[2]) * 0.5
                        cy = (box[1] + box[3]) * 0.5
                        pred_centers.append((cx, cy))
                        pred_labels.append(int(cls_id))
                        pred_scores.append(float(score))

                est_count = [
                    pred_labels.count(cls_id) for cls_id in range(1, self.max_class_id + 1)
                ]

                gt_dict = {"loc": gt_centers, "labels": gt_labels}
                pred_dict = {"loc": pred_centers, "labels": pred_labels, "scores": pred_scores}
                self.metrics.feed(gt_dict, pred_dict, est_count=est_count)
                n_processed += 1
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  Warning: Error processing {img_path.name}: {exc}")
                print(f"    Traceback: {traceback.format_exc()}")
                n_skipped += 1

        print(f"  ✓ Processed {n_processed} images ({n_skipped} skipped)")
        per_class_copy = self.metrics.copy()
        self.metrics.aggregate()

        overall = {
            "f1": self.metrics.fbeta_score(c=1),
            "precision": self.metrics.precision(c=1),
            "recall": self.metrics.recall(c=1),
            "mae": self.metrics.mae(c=1),
            "rmse": self.metrics.rmse(c=1),
        }

        per_class = {}
        for idx, class_id in enumerate(range(1, len(self.class_names) + 1), start=1):
            name = self.class_names[idx - 1] if idx - 1 < len(self.class_names) else f"class_{class_id}"
            per_class[name] = {
                "f1": per_class_copy.fbeta_score(c=class_id),
                "precision": per_class_copy.precision(c=class_id),
                "recall": per_class_copy.recall(c=class_id),
            }

        return {"overall": overall, "per_class": per_class}

    def _log_to_wandb(self, metrics: Dict, epoch: int) -> None:
        try:
            import wandb
        except ModuleNotFoundError:  # pragma: no cover - optional dep
            print("  Warning: wandb not installed, skipping logging")
            return

        if wandb.run is None:
            print("  Warning: No active WandB run, skipping logging")
            return

        payload = {
            "val_herdnet/f1": metrics["overall"]["f1"],
            "val_herdnet/precision": metrics["overall"]["precision"],
            "val_herdnet/recall": metrics["overall"]["recall"],
            "val_herdnet/mae": metrics["overall"]["mae"],
            "val_herdnet/rmse": metrics["overall"]["rmse"],
        }
        for class_name, class_metrics in metrics["per_class"].items():
            payload[f"val_herdnet_class/{class_name}/f1"] = class_metrics["f1"]
            payload[f"val_herdnet_class/{class_name}/precision"] = class_metrics["precision"]
            payload[f"val_herdnet_class/{class_name}/recall"] = class_metrics["recall"]

        wandb.log(payload, step=epoch)
        print(f"  ✓ Logged HerdNet metrics to WandB ({wandb.run.name})")

    def _print_metrics(self, metrics: Dict, epoch: int) -> None:
        print(f"\n{'=' * 70}\nHerdNet Metrics @ Epoch {epoch + 1} (threshold={self.threshold_px}px)\n{'=' * 70}")
        print("\nOverall (Binary Detection):")
        print(f"  F1:        {metrics['overall']['f1']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  MAE:       {metrics['overall']['mae']:.4f}")
        print(f"  RMSE:      {metrics['overall']['rmse']:.4f}")
        print("\nPer-Class Metrics:")
        print(f"  {'Class':<15} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-' * 15} {'-' * 8} {'-' * 10} {'-' * 8}")
        for class_name, values in metrics["per_class"].items():
            print(
                f"  {class_name:<15} "
                f"{values['f1']:>8.4f} "
                f"{values['precision']:>10.4f} "
                f"{values['recall']:>8.4f}"
            )
        print(f"{'=' * 70}\n")
