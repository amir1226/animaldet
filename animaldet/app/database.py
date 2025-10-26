"""SQLite database models for storing experiment results."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Experiment(Base):
    """Experiment run metadata."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    config = Column(JSON, nullable=True)

    # Relationships
    predictions = relationship("Prediction", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("ExperimentMetrics", back_populates="experiment", cascade="all, delete-orphan")


class Image(Base):
    """Image metadata (COCO-inspired)."""

    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False, unique=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    dataset_split = Column(String, nullable=True)  # train/val/test

    # Relationships
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")


class Category(Base):
    """Object category (COCO-inspired)."""

    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    # Relationships
    annotations = relationship("Annotation", back_populates="category")
    predictions = relationship("Prediction", back_populates="category")


class Annotation(Base):
    """Ground truth annotation (COCO-inspired)."""

    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    bbox_x_min = Column(Float, nullable=False)
    bbox_y_min = Column(Float, nullable=False)
    bbox_x_max = Column(Float, nullable=False)
    bbox_y_max = Column(Float, nullable=False)
    area = Column(Float, nullable=False)

    # Relationships
    image = relationship("Image", back_populates="annotations")
    category = relationship("Category", back_populates="annotations")
    predictions = relationship("Prediction", back_populates="matched_annotation")


class Prediction(Base):
    """Model prediction."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    bbox_x_min = Column(Float, nullable=False)
    bbox_y_min = Column(Float, nullable=False)
    bbox_x_max = Column(Float, nullable=False)
    bbox_y_max = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    matched_annotation_id = Column(Integer, ForeignKey("annotations.id"), nullable=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="predictions")
    image = relationship("Image", back_populates="predictions")
    category = relationship("Category", back_populates="predictions")
    matched_annotation = relationship("Annotation", back_populates="predictions")


class ExperimentMetrics(Base):
    """Overall metrics for an experiment."""

    __tablename__ = "experiment_metrics"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, unique=True)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    true_positives = Column(Integer, nullable=False)
    false_positives = Column(Integer, nullable=False)
    false_negatives = Column(Integer, nullable=False)

    # Relationships
    experiment = relationship("Experiment", back_populates="metrics")


def create_database(db_path: str = "experiments.db") -> None:
    """Create database and all tables.

    Args:
        db_path: Path to SQLite database file
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)


def get_session(db_path: str = "experiments.db") -> Session:
    """Get database session.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLAlchemy session
    """
    engine = create_engine(f"sqlite:///{db_path}")
    return Session(engine)
