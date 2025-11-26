#!/usr/bin/env python3
"""
Prepare demo images for Docker deployment.
Extracts first N unique images from CSV and compresses them to fit size budget.
"""
import csv
import os
import sys
from pathlib import Path
from PIL import Image

def get_unique_images_from_csv(csv_path, max_images=None):
    """Extract unique image names from CSV."""
    unique_images = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different column names
            image_name = row.get('images') or row.get('image') or row.get('Image')
            if image_name:
                unique_images.add(image_name)
                if max_images and len(unique_images) >= max_images:
                    break
    return sorted(list(unique_images))

def compress_image(input_path, output_path, max_size_kb=500, quality=85):
    """Compress image to fit within size budget while maintaining dimensions."""
    img = Image.open(input_path)
    original_size = os.path.getsize(input_path) / 1024

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Try different quality levels to fit size budget
    for q in range(quality, 15, -5):
        img.save(output_path, 'JPEG', quality=q, optimize=True)
        size_kb = os.path.getsize(output_path) / 1024
        if size_kb <= max_size_kb:
            print(f"  Compressed to {size_kb:.1f}KB at quality {q} (original: {original_size:.1f}KB)")
            return True

    # If still too large, save with lowest quality found
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Compressed to {size_kb:.1f}KB (best effort, original: {original_size:.1f}KB)")
    return True

def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "experiments/HerdNet/results/test_big_size_A_B_E_K_WH_WB.csv"

    # Get ALL unique images from CSV
    print(f"Reading CSV: {csv_path}")
    all_images = get_unique_images_from_csv(csv_path, max_images=None)
    print(f"Found {len(all_images)} unique images in CSV")

    # Create output directory
    output_dir = project_root / "demo_images"
    output_dir.mkdir(exist_ok=True)

    # Search for images in common locations
    search_paths = [
        Path("/home/lmanrique/Do/animaldet/data/herdnet/raw/test"),
        project_root / "data",
        project_root / "animaldet/data",
        Path.home() / "data/herdnet",
    ]

    compressed_count = 0
    total_budget_mb = 50  # 50MB total budget
    target_size_kb = 500  # Start with 500KB per image

    # Calculate how many images we can fit
    max_images = int((total_budget_mb * 1024) / target_size_kb)
    print(f"Target: {max_images} images at ~{target_size_kb}KB each = {total_budget_mb}MB total")

    sample_images = all_images[:max_images]  # Take first N images

    for image_name in sample_images:
        found = False
        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Find image recursively
            for img_path in search_path.rglob(image_name):
                print(f"Processing: {image_name}")
                output_path = output_dir / image_name

                try:
                    compress_image(img_path, output_path, max_size_kb=target_size_kb)
                    compressed_count += 1
                    found = True
                    break
                except Exception as e:
                    print(f"  Error: {e}")

            if found:
                break

        if not found:
            print(f"  Not found: {image_name}")

    print(f"\nCompressed {compressed_count} images to {output_dir}")
    total_size_mb = sum(f.stat().st_size for f in output_dir.glob("*.JPG")) / (1024**2)
    print(f"Total size: {total_size_mb:.2f}MB")

if __name__ == "__main__":
    main()
