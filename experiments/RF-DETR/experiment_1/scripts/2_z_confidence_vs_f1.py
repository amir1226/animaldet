#!/usr/bin/env python3
"""
Calculate Confidence vs F1 Score from CSV detection files
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def read_predictions_from_csv(csv_path: str):
    """
    Read predictions from CSV file.

    Expected format: images, x, y, x_max, y_max, labels, scores

    Returns:
        predictions: List of dicts with image_id, category_id, bbox, score
    """
    print(f"Loading predictions from {csv_path}")
    df = pd.read_csv(csv_path)

    predictions = []
    for _, row in df.iterrows():
        # Convert from [x, y, x_max, y_max] to [x, y, w, h]
        x, y, x_max, y_max = row['x'], row['y'], row['x_max'], row['y_max']
        w = x_max - x
        h = y_max - y

        predictions.append({
            'image_id': row['images'],
            'category_id': int(row['labels']),
            'bbox': [x, y, w, h],
            'score': float(row['scores'])
        })

    print(f"Loaded {len(predictions)} predictions from {len(df['images'].unique())} images")
    return predictions


def read_ground_truths_from_csv(csv_path: str):
    """
    Read ground truths from CSV file.

    Expected format: Image, x1, y1, x2, y2, Label

    Returns:
        ground_truths: List of dicts with image_id, category_id, bbox
    """
    print(f"Loading ground truths from {csv_path}")
    df = pd.read_csv(csv_path)

    ground_truths = []
    for _, row in df.iterrows():
        # Convert from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        w = x2 - x1
        h = y2 - y1

        ground_truths.append({
            'image_id': row['Image'],
            'category_id': int(row['Label']),
            'bbox': [x1, y1, w, h]
        })

    print(f"Loaded {len(ground_truths)} ground truths from {len(df['Image'].unique())} images")
    return ground_truths


def get_bbox_center(bbox):
    """Get center coordinates of a bbox [x, y, w, h]."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx, cy


def calculate_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between centers of two bboxes."""
    cx1, cy1 = get_bbox_center(bbox1)
    cx2, cy2 = get_bbox_center(bbox2)
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def match_detections(predictions, ground_truths, center_threshold=20.0):
    """
    Match predictions to ground truths based on center distance and class.

    A prediction is a true positive if:
    - Its center is within center_threshold pixels of a ground truth center
    - It has the same class as the ground truth
    - Each ground truth can only be matched once (greedy matching by confidence)

    Args:
        predictions: List of prediction dicts with image_id, category_id, bbox, score
        ground_truths: List of ground truth dicts with image_id, category_id, bbox
        center_threshold: Maximum center distance in pixels for a match

    Returns:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives
    """
    # Group by image
    pred_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)

    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)

    for gt in ground_truths:
        gt_by_image[gt['image_id']].append(gt)

    tp = 0
    fp = 0
    total_gt = len(ground_truths)

    # Process each image
    for image_id in pred_by_image:
        preds = pred_by_image[image_id]
        gts = gt_by_image.get(image_id, [])

        # Sort predictions by confidence (highest first)
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)

        # Track which ground truths have been matched
        matched_gts = set()

        for pred in preds:
            best_match_idx = None
            best_distance = float('inf')

            # Find the closest ground truth with matching class
            for i, gt in enumerate(gts):
                if i in matched_gts:
                    continue

                # Check class match
                if pred['category_id'] != gt['category_id']:
                    continue

                # Calculate center distance
                distance = calculate_center_distance(pred['bbox'], gt['bbox'])

                if distance < best_distance and distance <= center_threshold:
                    best_distance = distance
                    best_match_idx = i

            if best_match_idx is not None:
                # True positive
                tp += 1
                matched_gts.add(best_match_idx)
            else:
                # False positive
                fp += 1

    # False negatives are unmatched ground truths
    fn = total_gt - tp

    return tp, fp, fn


def calculate_f1_score(tp, fp, fn):
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def calculate_f1_at_thresholds(
    predictions,
    ground_truths,
    confidence_thresholds,
    center_threshold: float = 20.0
):
    """
    Calculate F1 score at different confidence thresholds.

    Args:
        predictions: List of all predictions with scores
        ground_truths: List of all ground truths
        confidence_thresholds: List of confidence thresholds to evaluate
        center_threshold: Center distance threshold in pixels

    Returns:
        results: Dict with thresholds and corresponding F1, precision, recall
    """
    results = {
        'thresholds': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'num_predictions': [],
        'true_positives': [],
        'false_positives': [],
        'false_negatives': []
    }

    print(f"\nEvaluating F1 at {len(confidence_thresholds)} confidence thresholds...")
    print(f"Total predictions: {len(predictions)}")
    print(f"Total ground truths: {len(ground_truths)}")
    print(f"Center distance threshold: {center_threshold}px")

    for threshold in tqdm(confidence_thresholds, desc="Calculating F1"):
        # Filter predictions by confidence threshold
        filtered_preds = [p for p in predictions if p['score'] >= threshold]

        # Match detections
        tp, fp, fn = match_detections(
            filtered_preds,
            ground_truths,
            center_threshold=center_threshold
        )

        # Calculate F1
        precision, recall, f1 = calculate_f1_score(tp, fp, fn)

        results['thresholds'].append(threshold)
        results['f1_scores'].append(f1)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['num_predictions'].append(len(filtered_preds))
        results['true_positives'].append(tp)
        results['false_positives'].append(fp)
        results['false_negatives'].append(fn)

    return results


def plot_confidence_vs_f1(results, output_path: str = 'confidence_vs_f1.png'):
    """Plot confidence threshold vs F1 score."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: F1 vs Confidence
    axes[0, 0].plot(results['thresholds'], results['f1_scores'],
                     marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score vs Confidence Threshold')
    axes[0, 0].grid(True, alpha=0.3)

    # Find and mark best F1
    best_idx = np.argmax(results['f1_scores'])
    best_threshold = results['thresholds'][best_idx]
    best_f1 = results['f1_scores'][best_idx]
    axes[0, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5,
                        label=f'Best: {best_threshold:.2f} (F1={best_f1:.4f})')
    axes[0, 0].legend()

    # Plot 2: Precision and Recall vs Confidence
    axes[0, 1].plot(results['thresholds'], results['precisions'],
                     marker='o', label='Precision', linewidth=2, markersize=4)
    axes[0, 1].plot(results['thresholds'], results['recalls'],
                     marker='s', label='Recall', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision & Recall vs Confidence Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    # Plot 3: Number of predictions vs Confidence
    axes[1, 0].plot(results['thresholds'], results['num_predictions'],
                     marker='o', linewidth=2, markersize=4, color='green')
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('Number of Predictions')
    axes[1, 0].set_title('Number of Predictions vs Confidence Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    # Plot 4: TP, FP, FN vs Confidence
    axes[1, 1].plot(results['thresholds'], results['true_positives'],
                     marker='o', label='True Positives', linewidth=2, markersize=4)
    axes[1, 1].plot(results['thresholds'], results['false_positives'],
                     marker='s', label='False Positives', linewidth=2, markersize=4)
    axes[1, 1].plot(results['thresholds'], results['false_negatives'],
                     marker='^', label='False Negatives', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Confidence Threshold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('TP/FP/FN vs Confidence Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    return best_threshold, best_f1


def main():
    # Configuration
    CSV_PATH = 'results/rfdetr_detections.csv'
    GT_CSV_PATH = 'data/herdnet/raw/groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv'
    CENTER_THRESHOLD = 20.0  # pixels

    # Confidence thresholds to evaluate
    CONFIDENCE_THRESHOLDS = np.arange(0.05, 0.96, 0.05)

    # Load predictions and ground truths from CSV
    predictions = read_predictions_from_csv(CSV_PATH)
    ground_truths = read_ground_truths_from_csv(GT_CSV_PATH)

    print(f"\nLoaded:")
    print(f"  {len(predictions)} predictions")
    print(f"  {len(ground_truths)} ground truths")

    # Calculate F1 at different confidence thresholds
    results = calculate_f1_at_thresholds(
        predictions,
        ground_truths,
        CONFIDENCE_THRESHOLDS,
        center_threshold=CENTER_THRESHOLD
    )

    # Plot results
    best_threshold, best_f1 = plot_confidence_vs_f1(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best confidence threshold: {best_threshold:.2f}")
    print(f"Best F1 score: {best_f1:.4f}")

    best_idx = np.argmax(results['f1_scores'])
    print(f"Precision at best threshold: {results['precisions'][best_idx]:.4f}")
    print(f"Recall at best threshold: {results['recalls'][best_idx]:.4f}")
    print(f"Predictions at best threshold: {results['num_predictions'][best_idx]}")
    print(f"True Positives: {results['true_positives'][best_idx]}")
    print(f"False Positives: {results['false_positives'][best_idx]}")
    print(f"False Negatives: {results['false_negatives'][best_idx]}")

    # Save results to file
    import json
    output_file = 'confidence_vs_f1_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
