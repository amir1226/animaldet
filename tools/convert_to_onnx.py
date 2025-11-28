#!/usr/bin/env python3
"""Export RF-DETR checkpoint to ONNX format.

This script loads a PyTorch checkpoint and exports the model to ONNX format
for efficient inference deployment.

Usage:
    uv run scripts/export_rfdetr_to_onnx.py <checkpoint_path> <output_path> [--web]
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Patch F.interpolate to disable antialias for ONNX export
_original_interpolate = F.interpolate

def _onnx_safe_interpolate(*args, **kwargs):
    """Remove antialias parameter which is not supported in ONNX."""
    kwargs.pop('antialias', None)
    return _original_interpolate(*args, **kwargs)


def patch_layernorm_for_onnx():
    """Patch RF-DETR's custom LayerNorm to be ONNX-compatible."""
    import rfdetr.models.backbone.projector as projector_module

    class ONNXLayerNorm(torch.nn.Module):
        """ONNX-compatible LayerNorm that uses static normalized_shape."""
        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.normalized_shape = (normalized_shape,)

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            # Use self.normalized_shape[0] instead of x.size(3) for ONNX compatibility
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

    projector_module.LayerNorm = ONNXLayerNorm


class ONNXWrapper(torch.nn.Module):
    """Wrapper to convert dict outputs to tuple for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        # Return only main predictions as tuple (no aux outputs for export)
        return outputs['pred_logits'], outputs['pred_boxes']


def load_checkpoint_info(checkpoint_path: str):
    """Load checkpoint and extract model config."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    # Determine variant from pretrain_weights or args
    variant = "small"
    if hasattr(args, "pretrain_weights") and args.pretrain_weights:
        for v in ["nano", "small", "medium", "base", "large"]:
            if v in args.pretrain_weights.lower():
                variant = v
                break

    # Get actual num_classes from weights (more reliable than args)
    # RF-DETR adds +1 internally, so we subtract 1 from the weight shape
    state_dict = ckpt["ema_model"] if "ema_model" in ckpt and ckpt["ema_model"] else ckpt["model"]
    actual_num_classes = state_dict["class_embed.weight"].shape[0] - 1

    return {
        "variant": variant,
        "num_classes": actual_num_classes,
        "resolution": args.resolution,
        "encoder": args.encoder,
        "hidden_dim": args.hidden_dim,
        "patch_size": args.patch_size,
        "num_windows": args.num_windows,
        "dec_layers": args.dec_layers,
        "sa_nheads": args.sa_nheads,
        "ca_nheads": args.ca_nheads,
        "dec_n_points": args.dec_n_points,
        "num_queries": args.num_queries,
        "num_select": args.num_select,
        "projector_scale": args.projector_scale,
        "out_feature_indexes": args.out_feature_indexes,
        "positional_encoding_size": args.positional_encoding_size if hasattr(args, "positional_encoding_size") else None,
        "state_dict": state_dict,
    }


def export_to_onnx(checkpoint_path: str, output_path: str, web: bool = False):
    """Export RF-DETR model to ONNX format.

    Args:
        checkpoint_path: Path to the model checkpoint
        output_path: Path to save the ONNX model
        web: If True, optimize for web deployment (fixed batch size, no dynamic axes)
    """
    # Patch LayerNorm before importing/building model
    patch_layernorm_for_onnx()

    # Import model classes after patching
    from rfdetr.detr import RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge, RFDETRNano

    MODEL_VARIANTS = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }

    print(f"Loading checkpoint from {checkpoint_path}")
    info = load_checkpoint_info(checkpoint_path)

    # Build model
    # num_classes is already the actual value from checkpoint weights
    print(f"Building {info['variant']} model with num_classes={info['num_classes']}...")
    model_class = MODEL_VARIANTS[info["variant"]]
    wrapper = model_class(
        num_classes=info["num_classes"],
        encoder=info["encoder"],
        patch_size=info["patch_size"],
        num_windows=info["num_windows"],
        hidden_dim=info["hidden_dim"],
        dec_layers=info["dec_layers"],
        sa_nheads=info["sa_nheads"],
        ca_nheads=info["ca_nheads"],
        dec_n_points=info["dec_n_points"],
        num_queries=info["num_queries"],
        num_select=info["num_select"],
        projector_scale=info["projector_scale"],
        out_feature_indexes=info["out_feature_indexes"],
        resolution=info["resolution"],
        pretrain_weights=None,
    )

    model = wrapper.model.model
    model.load_state_dict(info["state_dict"], strict=True)

    # Use CPU for export (CUDA can cause segfaults during ONNX export)
    device = "cpu"
    torch.set_default_device(device)

    model.to(device)
    model.eval()

    # Wrap model for ONNX export (converts dict output to tuple)
    export_model = ONNXWrapper(model)
    export_model.eval()

    # Create dummy input
    print(f"Creating dummy input (resolution: {info['resolution']}) on {device}")
    dummy_input = torch.randn(1, 3, info["resolution"], info["resolution"], device=device)

    # Export to ONNX
    export_mode = "web-optimized" if web else "standard"
    print(f"Exporting to ONNX ({export_mode})...")

    # Patch F.interpolate to remove antialias parameter during export
    F.interpolate = _onnx_safe_interpolate
    try:
        with torch.no_grad():
            # For web deployment, use fixed batch size (no dynamic axes)
            # For standard deployment, allow dynamic batch size
            # Use legacy exporter (dynamo=False) for compatibility
            # Use opset 16 for grid_sampler support
            if web:
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["pred_logits", "pred_boxes"],
                    dynamo=False,
                )
            else:
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["pred_logits", "pred_boxes"],
                    dynamic_axes={
                        "images": {0: "batch_size"},
                        "pred_logits": {0: "batch_size"},
                        "pred_boxes": {0: "batch_size"}
                    },
                    dynamo=False,
                )
    finally:
        # Restore original interpolate function
        F.interpolate = _original_interpolate

    # Save metadata JSON
    import json
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'variant': info['variant'],
            'num_classes': info['num_classes'],
            'resolution': info['resolution'],
            'input_names': ['images'],
            'output_names': ['pred_logits', 'pred_boxes']
        }, f, indent=2)

    print(f"✓ ONNX model saved to {output_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    print(f"  Model: RF-DETR {info['variant']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Resolution: {info['resolution']}x{info['resolution']}")
    print(f"  Device: {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RF-DETR models to ONNX format")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["nano", "small", "all"],
        default=["all"],
        help="Which models to convert (default: all)",
    )
    parser.add_argument("--web", action="store_true", help="Optimize for web deployment")
    args = parser.parse_args()

    # Define model configurations
    models_to_convert = {
        "nano": "rf-detr-nano-animaldet.pth",
        "small": "rf-detr-small-animaldet.pth",
    }

    # Determine which models to convert
    if "all" in args.models:
        selected_models = list(models_to_convert.keys())
    else:
        selected_models = args.models

    modelos_dir = Path(__file__).parent.parent / "modelos"
    success_count = 0

    for model_name in selected_models:
        checkpoint_file = models_to_convert[model_name]
        checkpoint_path = modelos_dir / checkpoint_file
        output_path = modelos_dir / checkpoint_file.replace(".pth", ".onnx")

        print(f"\n{'=' * 80}")
        print(f"Converting {model_name.upper()} model")
        print(f"{'=' * 80}")

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
            continue

        try:
            export_to_onnx(str(checkpoint_path), str(output_path), web=args.web)
            success_count += 1
        except Exception as e:
            print(f"Error converting {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Conversion completed: {success_count}/{len(selected_models)} models converted successfully")
    print(f"{'=' * 80}")
