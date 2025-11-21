"""Bounding-box helpers shared across notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["bbox_dataframe_to_points", "convert_bbox_csv_to_points"]


def bbox_dataframe_to_points(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` where boxes become center points."""
    points = df.copy()
    if not {"x", "y", "x_max", "y_max"}.issubset(points.columns):
        missing = {"x", "y", "x_max", "y_max"} - set(points.columns)
        raise ValueError(f"missing columns: {sorted(missing)}")
    points["x"] = (points["x"] + points["x_max"]) * 0.5
    points["y"] = (points["y"] + points["y_max"]) * 0.5
    cols = ["images", "x", "y", "labels"]
    if "scores" in points.columns:
        cols.append("scores")
    return points[cols].copy()


def convert_bbox_csv_to_points(input_csv: Path | str, output_csv: Path | str) -> pd.DataFrame:
    """Convert a CSV with boxes into points and write it to disk."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    frame = pd.read_csv(input_path)
    points = bbox_dataframe_to_points(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_path, index=False)
    return points
