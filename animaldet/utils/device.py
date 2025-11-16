"""Device utilities for inference."""

import torch


def get_device(device_name: str = "cuda", verbose: bool = True) -> torch.device:
    """Get PyTorch device for inference.

    Args:
        device_name: Device name ('cuda' or 'cpu')
        verbose: Print device information

    Returns:
        torch.device instance
    """
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU device")

    return device
