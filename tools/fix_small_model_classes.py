#!/usr/bin/env python3
"""Fix RF-DETR small model to have the same number of classes as nano.

This script trims the classification head of the small model from 91 classes (COCO)
to 7 classes (6 animals + background) to match the nano model.
"""

import sys
import torch
from pathlib import Path

def fix_classification_head(checkpoint_path: str, output_path: str, target_num_classes: int = 7):
    """Trim classification head to target number of classes.

    Args:
        checkpoint_path: Path to input checkpoint
        output_path: Path to save fixed checkpoint
        target_num_classes: Number of classes to keep (default: 7 for 6 animals)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Determine which state dict to use
    if 'ema_model' in ckpt and ckpt['ema_model']:
        state_dict_key = 'ema_model'
        state_dict = ckpt['ema_model']
        print("Using EMA model weights")
    elif 'model' in ckpt:
        state_dict_key = 'model'
        state_dict = ckpt['model']
        print("Using model weights")
    else:
        raise ValueError("No valid state dict found in checkpoint")

    # Check current number of classes
    current_num_classes = state_dict['class_embed.weight'].shape[0]
    print(f"Current num_classes: {current_num_classes}")
    print(f"Target num_classes: {target_num_classes}")

    if current_num_classes == target_num_classes:
        print("Model already has the correct number of classes!")
        return

    if current_num_classes < target_num_classes:
        raise ValueError(f"Cannot expand from {current_num_classes} to {target_num_classes} classes")

    # Find all classification head layers to trim
    class_layers = [
        'class_embed.weight',
        'class_embed.bias',
    ]

    # Also trim encoder classification heads
    encoder_class_layers = []
    for key in state_dict.keys():
        if 'enc_out_class_embed' in key and ('weight' in key or 'bias' in key):
            encoder_class_layers.append(key)

    all_class_layers = class_layers + encoder_class_layers
    print(f"\nTrimming {len(all_class_layers)} classification layers:")

    # Trim each layer
    for layer_name in all_class_layers:
        if layer_name not in state_dict:
            print(f"  Warning: {layer_name} not found, skipping")
            continue

        original_shape = state_dict[layer_name].shape

        if 'weight' in layer_name:
            # Weight: [num_classes, ...] -> trim first dimension
            state_dict[layer_name] = state_dict[layer_name][:target_num_classes, ...]
        elif 'bias' in layer_name:
            # Bias: [num_classes] -> trim
            state_dict[layer_name] = state_dict[layer_name][:target_num_classes]

        new_shape = state_dict[layer_name].shape
        print(f"  {layer_name}: {original_shape} -> {new_shape}")

    # Update args.num_classes if it exists
    if 'args' in ckpt and hasattr(ckpt['args'], 'num_classes'):
        original_args_classes = ckpt['args'].num_classes
        ckpt['args'].num_classes = target_num_classes
        print(f"\nUpdated args.num_classes: {original_args_classes} -> {target_num_classes}")

    # Update the state dict in checkpoint
    ckpt[state_dict_key] = state_dict

    # Also update 'model' key if it exists and we modified ema_model
    if state_dict_key == 'ema_model' and 'model' in ckpt:
        print("Also trimming 'model' state dict")
        model_state = ckpt['model']
        for layer_name in all_class_layers:
            if layer_name not in model_state:
                continue
            if 'weight' in layer_name:
                model_state[layer_name] = model_state[layer_name][:target_num_classes, ...]
            elif 'bias' in layer_name:
                model_state[layer_name] = model_state[layer_name][:target_num_classes]
        ckpt['model'] = model_state

    # Save fixed checkpoint
    print(f"\nSaving fixed checkpoint to {output_path}")
    torch.save(ckpt, output_path)

    # Verify the fix
    print("\nVerifying fix...")
    ckpt_verify = torch.load(output_path, map_location='cpu', weights_only=False)
    verify_state = ckpt_verify.get('ema_model', ckpt_verify.get('model'))
    verify_classes = verify_state['class_embed.weight'].shape[0]

    if verify_classes == target_num_classes:
        print(f"✓ Success! Model now has {verify_classes} classes")
    else:
        print(f"✗ Error! Model has {verify_classes} classes instead of {target_num_classes}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Fix RF-DETR small model classification head")
    parser.add_argument("--yes", "-y", action="store_true", help="Replace original without asking")
    parser.add_argument("--classes", type=int, default=7, help="Target number of classes (default: 7)")
    args = parser.parse_args()

    # Paths
    input_path = Path(__file__).parent.parent / "modelos" / "rf-detr-small-animaldet.pth"
    output_path = Path(__file__).parent.parent / "modelos" / "rf-detr-small-animaldet-fixed.pth"
    backup_path = Path(__file__).parent.parent / "modelos" / "rf-detr-small-animaldet-backup.pth"

    print("=" * 80)
    print("RF-DETR Small Model Classification Head Fix")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Fix the model
    try:
        fix_classification_head(str(input_path), str(output_path), target_num_classes=args.classes)

        # Replace original
        print("\n" + "=" * 80)
        if args.yes:
            replace = True
        else:
            try:
                response = input("Replace original file with fixed version? (y/N): ")
                replace = response.lower() == 'y'
            except EOFError:
                replace = False
                print("N (non-interactive mode)")

        if replace:
            # Create backup
            if input_path.exists():
                print(f"Creating backup at {backup_path}")
                shutil.copy2(input_path, backup_path)

            # Replace
            shutil.move(str(output_path), str(input_path))
            print(f"✓ Replaced {input_path}")
            print(f"  Backup saved at {backup_path}")
        else:
            print(f"✓ Fixed model saved at {output_path}")
            print(f"  Original unchanged at {input_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
