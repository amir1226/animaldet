"""Storage logic for loading experiment results into database."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from animaldet.app.database import (
    Annotation,
    Category,
    Experiment,
    ExperimentMetrics,
    Image,
    Prediction,
    create_database,
    get_session,
)


class ExperimentStorage:
    """Storage manager for experiment results."""

    def __init__(self, db_path: str = "experiments.db"):
        """Initialize storage manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        create_database(db_path)

    def get_session(self) -> Session:
        """Get database session."""
        return get_session(self.db_path)

    def load_categories(self, category_names: Dict[int, str]) -> None:
        """Load categories into database.

        Args:
            category_names: Mapping of category ID to name
        """
        with self.get_session() as session:
            for cat_id, name in category_names.items():
                existing = session.query(Category).filter_by(id=cat_id).first()
                if not existing:
                    category = Category(id=cat_id, name=name)
                    session.add(category)
            session.commit()

    def load_ground_truths(
        self,
        annotations_csv: str,
        dataset_split: Optional[str] = None,
    ) -> None:
        """Load ground truth annotations from CSV.

        Args:
            annotations_csv: Path to annotations CSV
            dataset_split: Dataset split (train/val/test)

        Expected CSV columns: images, x_min, y_min, x_max, y_max, labels
        """
        df = pd.read_csv(annotations_csv)

        with self.get_session() as session:
            for _, row in df.iterrows():
                file_name = row["images"]

                # Get or create image
                image = session.query(Image).filter_by(file_name=file_name).first()
                if not image:
                    image = Image(
                        file_name=file_name,
                        dataset_split=dataset_split,
                    )
                    session.add(image)
                    session.flush()

                # Create annotation
                x_min, y_min = float(row["x_min"]), float(row["y_min"])
                x_max, y_max = float(row["x_max"]), float(row["y_max"])
                area = (x_max - x_min) * (y_max - y_min)

                annotation = Annotation(
                    image_id=image.id,
                    category_id=int(row["labels"]),
                    bbox_x_min=x_min,
                    bbox_y_min=y_min,
                    bbox_x_max=x_max,
                    bbox_y_max=y_max,
                    area=area,
                )
                session.add(annotation)

            session.commit()

    def load_experiment_predictions(
        self,
        experiment_name: str,
        model_name: str,
        detections_csv: str,
        scores_csv: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> int:
        """Load experiment predictions from CSV files.

        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            detections_csv: Path to detections CSV
            scores_csv: Path to scores CSV (optional)
            config: Additional experiment configuration

        Returns:
            Experiment ID

        Expected detections CSV columns: images, x_min, y_min, x_max, y_max, labels, scores
        Expected scores CSV columns: precision, recall, f1_score, true_positives, false_positives, false_negatives
        """
        with self.get_session() as session:
            # Create or get experiment
            experiment = session.query(Experiment).filter_by(name=experiment_name).first()
            if not experiment:
                experiment = Experiment(
                    name=experiment_name,
                    model_name=model_name,
                    config=config,
                )
                session.add(experiment)
                session.flush()
            else:
                # Delete existing predictions for this experiment
                session.query(Prediction).filter_by(experiment_id=experiment.id).delete()
                session.query(ExperimentMetrics).filter_by(experiment_id=experiment.id).delete()

            # Load detections
            df = pd.read_csv(detections_csv)

            for _, row in df.iterrows():
                file_name = row["images"]

                # Get or create image
                image = session.query(Image).filter_by(file_name=file_name).first()
                if not image:
                    image = Image(file_name=file_name)
                    session.add(image)
                    session.flush()

                # Create prediction
                prediction = Prediction(
                    experiment_id=experiment.id,
                    image_id=image.id,
                    category_id=int(row["labels"]),
                    bbox_x_min=float(row["x_min"]),
                    bbox_y_min=float(row["y_min"]),
                    bbox_x_max=float(row["x_max"]),
                    bbox_y_max=float(row["y_max"]),
                    confidence=float(row["scores"]),
                )
                session.add(prediction)

            # Load metrics if provided
            if scores_csv and Path(scores_csv).exists():
                metrics_df = pd.read_csv(scores_csv)
                if not metrics_df.empty:
                    row = metrics_df.iloc[0]
                    metrics = ExperimentMetrics(
                        experiment_id=experiment.id,
                        precision=float(row["precision"]),
                        recall=float(row["recall"]),
                        f1_score=float(row["f1_score"]),
                        true_positives=int(row["true_positives"]),
                        false_positives=int(row["false_positives"]),
                        false_negatives=int(row["false_negatives"]),
                    )
                    session.add(metrics)

            session.commit()
            return experiment.id

    def get_experiment_summary(self, experiment_id: int) -> dict:
        """Get summary of experiment results.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with experiment summary
        """
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            metrics = session.query(ExperimentMetrics).filter_by(experiment_id=experiment_id).first()
            num_predictions = session.query(Prediction).filter_by(experiment_id=experiment_id).count()
            num_images = (
                session.query(Prediction.image_id)
                .filter_by(experiment_id=experiment_id)
                .distinct()
                .count()
            )

            return {
                "id": experiment.id,
                "name": experiment.name,
                "model_name": experiment.model_name,
                "created_at": experiment.created_at.isoformat(),
                "num_predictions": num_predictions,
                "num_images": num_images,
                "metrics": {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "true_positives": metrics.true_positives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives,
                }
                if metrics
                else None,
            }

    def get_image_predictions(
        self,
        experiment_id: int,
        image_file_name: str,
    ) -> dict:
        """Get predictions and ground truths for a specific image.

        Args:
            experiment_id: Experiment ID
            image_file_name: Image file name

        Returns:
            Dictionary with predictions and ground truths
        """
        with self.get_session() as session:
            image = session.query(Image).filter_by(file_name=image_file_name).first()
            if not image:
                raise ValueError(f"Image {image_file_name} not found")

            # Get predictions
            predictions = (
                session.query(Prediction)
                .filter_by(experiment_id=experiment_id, image_id=image.id)
                .all()
            )

            # Get ground truths
            annotations = session.query(Annotation).filter_by(image_id=image.id).all()

            return {
                "image": {
                    "id": image.id,
                    "file_name": image.file_name,
                    "width": image.width,
                    "height": image.height,
                },
                "predictions": [
                    {
                        "id": pred.id,
                        "category_id": pred.category_id,
                        "bbox": [
                            pred.bbox_x_min,
                            pred.bbox_y_min,
                            pred.bbox_x_max,
                            pred.bbox_y_max,
                        ],
                        "confidence": pred.confidence,
                    }
                    for pred in predictions
                ],
                "annotations": [
                    {
                        "id": ann.id,
                        "category_id": ann.category_id,
                        "bbox": [
                            ann.bbox_x_min,
                            ann.bbox_y_min,
                            ann.bbox_x_max,
                            ann.bbox_y_max,
                        ],
                        "area": ann.area,
                    }
                    for ann in annotations
                ],
            }
