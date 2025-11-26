"""Model registry for RF-DETR models.

This module provides a registry pattern for managing different model variants.
Models can be registered using the @MODELS.register() decorator.
"""

from typing import Dict, Optional, Any
from pathlib import Path


class ModelConfig:
    """Configuration for a registered model."""

    def __init__(
        self,
        name: str,
        model_path: str,
        resolution: int,
        num_classes: int,
        description: str = "",
        class_offset: int = 0,
    ):
        self.name = name
        self.model_path = model_path
        self.resolution = resolution
        self.num_classes = num_classes
        self.description = description
        self.class_offset = class_offset

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "resolution": self.resolution,
            "num_classes": self.num_classes,
            "description": self.description,
            "class_offset": self.class_offset,
        }


class ModelRegistry:
    """Registry for managing model configurations."""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._default_model: Optional[str] = None

    def register(
        self,
        name: str,
        model_path: str,
        resolution: int,
        num_classes: int,
        description: str = "",
        default: bool = False,
        class_offset: int = 0,
    ):
        """Register a model configuration.

        Args:
            name: Model identifier (e.g., "nano", "small")
            model_path: Path to the model file
            resolution: Model input resolution
            num_classes: Number of classes
            description: Optional description
            default: Whether this is the default model
            class_offset: Offset to apply to predicted class IDs (e.g., -1 for nano)
        """
        config = ModelConfig(
            name=name,
            model_path=model_path,
            resolution=resolution,
            num_classes=num_classes,
            description=description,
            class_offset=class_offset,
        )
        self._models[name] = config

        if default or self._default_model is None:
            self._default_model = name

        return config

    def get(self, name: Optional[str] = None) -> ModelConfig:
        """Get a model configuration by name.

        Args:
            name: Model name, or None to get the default model

        Returns:
            Model configuration

        Raises:
            ValueError: If model not found
        """
        if name is None:
            name = self._default_model

        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )

        return self._models[name]

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        return {
            name: config.to_dict()
            for name, config in self._models.items()
        }

    def get_default(self) -> str:
        """Get the default model name."""
        if self._default_model is None:
            raise ValueError("No models registered")
        return self._default_model


# Global model registry
MODELS = ModelRegistry()

# Register available models (both trained for 6 animal classes)
MODELS.register(
    name="nano",
    model_path="rf-detr-nano-animaldet.onnx",
    resolution=384,
    num_classes=6,  # 6 animal classes
    description="RF-DETR Nano - Lightweight and fast (384x384)",
    default=False,
    class_offset=-1,  # Nano model predicts classes 1 higher than expected
)

MODELS.register(
    name="small",
    model_path="rf-detr-small-animaldet.onnx",
    resolution=512,
    num_classes=6,  # 6 animal classes
    description="RF-DETR Small - Higher resolution and accuracy (512x512)",
    default=True,
)
