"""
HerdNet Metrics Callback for RF-DETR Training

This callback calculates HerdNet-style metrics (point-based matching) during 
RF-DETR training to enable fair comparison with HerdNet baseline.
"""
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from PIL import Image


class HerdNetMetricsCallback:
    """
    Callback to calculate HerdNet-style metrics during RF-DETR training.
    
    Converts bounding box predictions to center points and evaluates using
    Euclidean distance matching (like HerdNet) instead of IoU.
    
    Args:
        model: RF-DETR model instance
        val_dataset_path (str): Path to COCO JSON file with validation annotations
        val_images_dir (str): Directory containing validation images
        threshold_px (int): Distance threshold in pixels for matching (default: 20)
        eval_every_n_epochs (int): Evaluate every N epochs (default: 1)
        confidence_threshold (float): Confidence threshold for predictions (default: 0.3)
        wandb_log (bool): Whether to log to WandB (default: True)
        class_names (list, optional): List of class names
        device (str): Device to run inference on (default: 'cuda')
        max_val_images (int, optional): Limit validation to N images for speed
    """
    
    # Default class names (matching HerdNet evaluation script)
    DEFAULT_CLASS_NAMES = [
        "Hartebeest",  # Class 1 (Alcelaphinae in COCO)
        "Buffalo",     # Class 2
        "Kob",         # Class 3
        "Warthog",     # Class 4
        "Waterbuck",   # Class 5
        "Elephant",    # Class 6
    ]
    
    def __init__(
        self,
        model,
        val_dataset_path: str,
        val_images_dir: str,
        threshold_px: int = 20,
        eval_every_n_epochs: int = 1,
        confidence_threshold: float = 0.3,
        wandb_log: bool = True,
        class_names: Optional[List[str]] = None,
        device: str = 'cuda',
        max_val_images: Optional[int] = None
    ):
        self.model = model
        self.val_dataset_path = Path(val_dataset_path)
        self.val_images_dir = Path(val_images_dir)
        self.threshold_px = threshold_px
        self.eval_every_n_epochs = eval_every_n_epochs
        self.confidence_threshold = confidence_threshold
        self.wandb_log = wandb_log
        self.device = device
        self.max_val_images = max_val_images
        
        # Validate paths
        if not self.val_dataset_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_dataset_path}")
        if not self.val_images_dir.exists():
            raise FileNotFoundError(f"Validation images dir not found: {val_images_dir}")
        
        # Load COCO dataset
        from pycocotools.coco import COCO
        print(f"Loading validation dataset from {val_dataset_path}...")
        self.coco = COCO(str(val_dataset_path))
        
        # Get class info
        cats = self.coco.loadCats(self.coco.getCatIds())
        # Use provided names, or default names (matching HerdNet evaluation)
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        
        # IMPORTANT: num_classes must be max(class_id) + 1, not len(cats) + 1
        # because COCO class IDs might not be consecutive (e.g., 1, 2, 4, 5, 6 - missing 3)
        self.max_class_id = max(cat['id'] for cat in cats) if cats else 0
        num_classes = self.max_class_id + 1
        
        print(f"  Found {len(cats)} classes: {self.class_names}")
        print(f"  Max class ID: {self.max_class_id}, num_classes: {num_classes}")
        print(f"  Found {len(self.coco.getImgIds())} validation images")
        
        # Initialize HerdNet metrics
        # Single PointsMetrics instance for both overall (after aggregate) 
        # and per-class metrics (without aggregate)
        from animaloc.eval.metrics import PointsMetrics
        self.metrics = PointsMetrics(
            radius=threshold_px,
            num_classes=num_classes
        )
        
        print(f"✓ HerdNet Metrics Callback initialized")
        print(f"  - Threshold: {threshold_px}px")
        print(f"  - Eval frequency: every {eval_every_n_epochs} epoch(s)")
        print(f"  - Confidence threshold: {confidence_threshold}")
        
    def update(self, log_stats):
        """
        Called at the end of each epoch by RF-DETR.
        
        Args:
            log_stats (dict): Dictionary with epoch info and training metrics
                - 'epoch': current epoch number
                - 'train_loss': training loss
                - 'test_coco_eval_bbox': validation metrics
                - etc.
        """
        epoch = log_stats.get('epoch', 0)
        
        # Evaluate at epoch 0 and then every N epochs (0, N, 2N, 3N, ...)
        if epoch % self.eval_every_n_epochs != 0:
            return
        
        print(f"\n{'='*70}")
        print(f"[HerdNet Metrics] Evaluating at epoch {epoch}...")
        print(f"{'='*70}")
        
        # Flush previous metrics
        self.metrics.flush()
        
        # Run evaluation
        herdnet_metrics = self._evaluate()
        
        # Log to WandB
        if self.wandb_log:
            self._log_to_wandb(herdnet_metrics, epoch)
        
        # Print metrics
        self._print_metrics(herdnet_metrics, epoch)
        
        return herdnet_metrics
    
    def _evaluate(self) -> Dict:
        """
        Run inference on validation set and calculate HerdNet metrics.
        
        Returns:
            dict: Dictionary with overall and per-class metrics
        """
        # Note: No need to call model.eval() here because self.model.predict() 
        # already handles it internally
        
        img_ids = self.coco.getImgIds()
        if self.max_val_images:
            img_ids = img_ids[:self.max_val_images]
            print(f"  Evaluating on {len(img_ids)} images (limited)")
        else:
            print(f"  Evaluating on {len(img_ids)} images")
        
        n_processed = 0
        n_skipped = 0
        
        for img_id in img_ids:
            # Load image info
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = self.val_images_dir / img_info['file_name']
            
            if not img_path.exists():
                n_skipped += 1
                continue
            
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Run prediction
                detections = self.model.predict(
                    image, 
                    threshold=self.confidence_threshold
                )
                
                # Get ground truth
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                if len(anns) == 0:
                    continue
                
                # Convert GT bboxes to centers
                gt_centers = []
                gt_labels = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    cx, cy = x + w/2, y + h/2
                    gt_centers.append((cx, cy))
                    gt_labels.append(ann['category_id'])
                
                # Convert prediction bboxes to centers
                pred_centers = []
                pred_labels = []
                pred_scores = []
                
                if len(detections.xyxy) > 0:
                    for box, cls, score in zip(
                        detections.xyxy, 
                        detections.class_id, 
                        detections.confidence
                    ):
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        pred_centers.append((cx, cy))
                        # Note: RF-DETR class_id starts at 0, COCO at 1
                        pred_labels.append(int(cls) + 1)
                        pred_scores.append(float(score))
                
                # Prepare dicts for metrics
                gt_dict = {
                    'loc': gt_centers,
                    'labels': gt_labels
                }
                
                # Handle empty predictions
                if pred_centers:
                    pred_dict = {
                        'loc': pred_centers,
                        'labels': pred_labels,
                        'scores': pred_scores
                    }
                else:
                    pred_dict = {
                        'loc': [],
                        'labels': [],
                        'scores': []
                    }
                
                # Calculate count per class (for MAE/RMSE)
                # Count for ALL possible class IDs (1 to max_class_id), not just present ones
                counts = [pred_labels.count(cls_id) for cls_id in range(1, self.max_class_id + 1)]
                
                # Feed to metrics (multiclass - will be used for both overall and per-class)
                self.metrics.feed(gt_dict, pred_dict, est_count=counts)
                
                n_processed += 1
                
            except Exception as e:
                print(f"  Warning: Error processing {img_path.name}: {e}")
                print(f"    Traceback: {traceback.format_exc()}")
                n_skipped += 1
                continue
        
        print(f"  ✓ Processed {n_processed} images ({n_skipped} skipped)")
        
        # Copy metrics before aggregating (for per-class calculation)
        per_class_copy = self.metrics.copy()
        
        # Debug: Print confusion matrix and counts
        print(f"\n  Debug - TP/FP/FN per class:")
        for i in range(len(per_class_copy.tp)):
            class_id = i + 1
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{class_id}"
            tp = per_class_copy.tp[i]
            fp = per_class_copy.fp[i]
            fn = per_class_copy.fn[i]
            print(f"    {class_name:<12} (id={class_id}): TP={tp:3d}, FP={fp:3d}, FN={fn:3d}")
        
        # Aggregate overall metrics (binary: object vs background)
        self.metrics.aggregate()
        
        # Calculate overall metrics (binary: any object vs background)
        overall_metrics = {
            'f1': self.metrics.fbeta_score(c=1),
            'precision': self.metrics.precision(c=1),
            'recall': self.metrics.recall(c=1),
            'mae': self.metrics.mae(c=1),
            'rmse': self.metrics.rmse(c=1),
        }
        
        # Calculate per-class metrics (multiclass, without aggregating)
        per_class_metrics = {}
        for i, class_id in enumerate(range(1, len(self.class_names) + 1)):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{class_id}"
            per_class_metrics[class_name] = {
                'f1': per_class_copy.fbeta_score(c=class_id),
                'precision': per_class_copy.precision(c=class_id),
                'recall': per_class_copy.recall(c=class_id),
            }
        
        return {
            'overall': overall_metrics,
            'per_class': per_class_metrics
        }
    
    def _log_to_wandb(self, metrics: Dict, epoch: int):
        """
        Log metrics to WandB.
        
        Args:
            metrics (dict): Metrics dictionary
            epoch (int): Current epoch
        """
        try:
            import wandb
            
            # Check if WandB run is active
            if wandb.run is None:
                print("  Warning: No active WandB run, skipping logging")
                return
            
            # Log overall metrics
            log_dict = {
                'val_herdnet/f1': metrics['overall']['f1'],
                'val_herdnet/precision': metrics['overall']['precision'],
                'val_herdnet/recall': metrics['overall']['recall'],
                'val_herdnet/mae': metrics['overall']['mae'],
                'val_herdnet/rmse': metrics['overall']['rmse'],
            }
            
            # Log per-class metrics
            for class_name, class_metrics in metrics['per_class'].items():
                log_dict[f'val_herdnet_class/{class_name}/f1'] = class_metrics['f1']
                log_dict[f'val_herdnet_class/{class_name}/precision'] = class_metrics['precision']
                log_dict[f'val_herdnet_class/{class_name}/recall'] = class_metrics['recall']
            
            wandb.log(log_dict, step=epoch)
            print(f"  ✓ Logged to WandB run: {wandb.run.name}")
            
        except Exception as e:
            print(f"  Warning: Could not log to WandB: {e}")
    
    def _print_metrics(self, metrics: Dict, epoch: int):
        """
        Print metrics to console.
        
        Args:
            metrics (dict): Metrics dictionary
            epoch (int): Current epoch
        """
        print(f"\n{'='*70}")
        print(f"HerdNet Metrics @ Epoch {epoch + 1} (threshold={self.threshold_px}px)")
        print(f"{'='*70}")
        
        # Overall metrics
        print("\nOverall (Binary Detection):")
        print(f"  F1:        {metrics['overall']['f1']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  MAE:       {metrics['overall']['mae']:.4f}")
        print(f"  RMSE:      {metrics['overall']['rmse']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print(f"  {'Class':<15} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8}")
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name:<15} "
                  f"{class_metrics['f1']:>8.4f} "
                  f"{class_metrics['precision']:>10.4f} "
                  f"{class_metrics['recall']:>8.4f}")
        
        print(f"{'='*70}\n")

