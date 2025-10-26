"""Example usage of experiment storage system."""

from animaldet.app.storage import ExperimentStorage

# Initialize storage
storage = ExperimentStorage(db_path="experiments.db")

# Load categories
categories = {
    0: "background",
    1: "bovine",
    2: "cervid",
    3: "moose",
    4: "equine",
}
storage.load_categories(categories)

# Load experiment predictions
experiment_id = storage.load_experiment_predictions(
    experiment_name="rfdetr_small",
    model_name="RF-DETR Small",
    detections_csv="outputs/inference/rfdetr_detections_small.csv",
    scores_csv="outputs/inference/rfdetr_scores_small.csv",
    config={
        "confidence_threshold": 0.5,
        "resolution": 512,
    },
)

# Get experiment summary
summary = storage.get_experiment_summary(experiment_id)
print(f"Experiment: {summary['name']}")
print(f"Model: {summary['model_name']}")
print(f"Images: {summary['num_images']}")
print(f"Predictions: {summary['num_predictions']}")
if summary["metrics"]:
    print(f"Precision: {summary['metrics']['precision']:.4f}")
    print(f"Recall: {summary['metrics']['recall']:.4f}")
    print(f"F1: {summary['metrics']['f1_score']:.4f}")

# Get predictions for a specific image
image_results = storage.get_image_predictions(
    experiment_id=experiment_id,
    image_file_name="S_07_05_16_DSC00094.JPG",
)
print(f"\nImage: {image_results['image']['file_name']}")
print(f"Predictions: {len(image_results['predictions'])}")
print(f"Ground truths: {len(image_results['annotations'])}")
