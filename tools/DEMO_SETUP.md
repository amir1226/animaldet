# Demo Images Setup for Docker

## Overview
The Dockerfile includes experiment CSVs but needs demo images to be prepared before building.

## CSV Files Location
- **RF-DETR**: `experiments/RF-DETR/results/detections.csv` → `/static/experiments/rfdetr_detections.csv`
- **HerdNet**: `experiments/HerdNet/results/detections.csv` → `/static/experiments/herdnet_detections.csv`

## Preparing Demo Images

### Option 1: Automatic Preparation (Recommended)
Run the preparation script to compress and copy images:

```bash
# Run from project root
python tools/prepare_demo_images.py
```

This will:
1. Extract first 10 unique image names from `experiments/RF-DETR/results/detections.csv`
2. Search for these images in common locations
3. Compress each image to ~300KB (JPEG quality optimized)
4. Save to `demo_images/` directory
5. Target total size: 3-5MB for ~10 images

### Option 2: Manual Selection
1. Check which images are in the CSV:
   ```bash
   cut -d',' -f1 experiments/RF-DETR/results/detections.csv | sort -u | head -10
   ```

2. Create demo_images directory:
   ```bash
   mkdir -p demo_images
   ```

3. Copy and compress images manually:
   ```bash
   # Using ImageMagick
   convert input.JPG -quality 75 -resize 4000x4000\> demo_images/output.JPG
   ```

### Option 3: Skip Demo Images
If you don't have the images, comment out this line in the Dockerfile:
```dockerfile
# COPY demo_images/*.JPG ./static/demo_images/
```

The app will still work, but the Experiments page won't display images (CSVs will still load).

## Updating Dockerfile

Once demo images are ready:

1. Uncomment the line in Dockerfile:
   ```dockerfile
   COPY demo_images/*.JPG ./static/demo_images/
   ```

2. Rebuild the Docker image:
   ```bash
   docker build -t animaldet .
   ```

## Size Budget
- **Target**: 10-50MB total for images
- **Per image**: ~300-500KB (compressed JPEG)
- **Number of images**: 5-10 unique images
- **CSVs**: ~400KB total (already included)

## Verifying Size
```bash
# Check demo_images size
du -sh demo_images

# Check individual file sizes
ls -lh demo_images/*.JPG
```

## Notes
- Images are served as static files via FastAPI
- CSVs reference image filenames, UI fetches them from `/demo_images/<filename>`
- For production, consider using a CDN or S3 for images
