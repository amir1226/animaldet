# We must detect all objects in train set images of arbitrary size using a model trained on 512x512 patches.
# The results are saved in COCO format for evaluation, as hard negative patches to be used in further training.

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from rfdetr.detr import RFDETRSmall
from PIL import Image

# Simple stitcher for 512x512 patches
class SimpleStitcher:
    def __init__(self, model, patch_size=512, conf_threshold=0.01):
        self.model = model
        self.patch_size = patch_size
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_path):
        # Load image [C, H, W]
        img = read_image(str(image_path)).float() / 255.0
        C, H, W = img.shape

        # Calculate patches
        patches_y = (H + self.patch_size - 1) // self.patch_size
        patches_x = (W + self.patch_size - 1) // self.patch_size

        all_boxes, all_scores, all_labels = [], [], []

        for py in range(patches_y):
            for px in range(patches_x):
                y1 = py * self.patch_size
                x1 = px * self.patch_size
                y2 = min(y1 + self.patch_size, H)
                x2 = min(x1 + self.patch_size, W)

                # Extract and pad patch
                patch = img[:, y1:y2, x1:x2]
                ph, pw = patch.shape[1:]
                if ph < self.patch_size or pw < self.patch_size:
                    patch = F.pad(patch, (0, self.patch_size - pw, 0, self.patch_size - ph))

                # Run inference
                patch_batch = patch.unsqueeze(0).to(self.device)
                outputs = self.model(patch_batch)

                # Process detections
                logits = outputs['pred_logits'][0].sigmoid()
                boxes = outputs['pred_boxes'][0]

                scores, labels = logits.max(dim=-1)
                keep = scores > self.conf_threshold

                if keep.sum() > 0:
                    # Convert boxes from [cx,cy,w,h] normalized to [x1,y1,x2,y2] absolute
                    boxes = boxes[keep]
                    cx, cy, w, h = boxes.unbind(-1)
                    x1_box = (cx - 0.5 * w) * self.patch_size + x1
                    y1_box = (cy - 0.5 * h) * self.patch_size + y1
                    x2_box = (cx + 0.5 * w) * self.patch_size + x1
                    y2_box = (cy + 0.5 * h) * self.patch_size + y1

                    all_boxes.append(torch.stack([x1_box, y1_box, x2_box, y2_box], dim=-1))
                    all_scores.append(scores[keep])
                    all_labels.append(labels[keep] + 1)  # 1-indexed

        if len(all_boxes) == 0:
            return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

        return torch.cat(all_boxes).cpu(), torch.cat(all_scores).cpu(), torch.cat(all_labels).cpu()

# Load model
model = RFDETRSmall()
checkpoint = torch.load('checkpoint_phase_1.pth', map_location='cpu', weights_only=False)
model.model.model.load_state_dict(checkpoint['model'])

stitcher = SimpleStitcher(model.model.model, patch_size=512, conf_threshold=0.01)

# Evaluation data path
data_path = Path('data/herdnet/raw/train')
images = list(data_path.glob('*.JPG'))

# Build COCO format structure
coco_output = {
    "info": {
        "description": "RF-DETR Test Set Predictions",
        "url": "None",
        "version": "1.0",
        "year": "2025",
        "contributor": "RF-DETR Experiment 1",
        "date_created": datetime.now().strftime("%Y-%m-%d")
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "animal", "id": 1, "name": "Alcelaphinae"},
        {"supercategory": "animal", "id": 2, "name": "Buffalo"},
        {"supercategory": "animal", "id": 3, "name": "Kob"},
        {"supercategory": "animal", "id": 4, "name": "Warthog"},
        {"supercategory": "animal", "id": 5, "name": "Waterbuck"},
        {"supercategory": "animal", "id": 6, "name": "Elephant"}
    ]
}

# Run inference and collect results
annotation_id = 1
for image_id, img_path in enumerate(tqdm(images, desc="Evaluating images"), start=1):
    # Get image dimensions
    with Image.open(img_path) as img:
        width, height = img.size

    # Add image entry
    coco_output["images"].append({
        "license": 0,
        "file_name": img_path.name,
        "coco_url": "None",
        "height": height,
        "width": width,
        "date_captured": "None",
        "flickr_url": "None",
        "id": image_id
    })

    # Run detection
    boxes, scores, labels = stitcher(img_path)

    # Add annotations
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        area = bbox_width * bbox_height

        coco_output["annotations"].append({
            "segmentation": [[]],
            "area": float(area),
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
            "category_id": int(label.item()),
            "id": annotation_id,
            "score": float(score.item())
        })
        annotation_id += 1

# Save to JSON
output_file = 'eval_results_coco.json'
with open(output_file, 'w') as f:
    json.dump(coco_output, f, indent=2)

print(f"Evaluated {len(images)} images with {len(coco_output['annotations'])} detections.")
print(f"Results saved to {output_file}")
