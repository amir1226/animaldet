"""Class name mappings for detection models."""

# Animal classes for the RF-DETR model
# Class 0 is background, classes 1-6 are animals
CLASS_NAMES = {
    0: "Background",  # Background class (should be filtered out)
    1: "Topi",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
    7: "Elephant",  # Temporary fix for nano model - maps to Elephant
}


def get_class_name(class_id: int) -> str:
    """Get class name for a given class ID.

    Args:
        class_id: Class ID (1-indexed)

    Returns:
        Class name or 'unknown'
    """
    return CLASS_NAMES.get(class_id, f"unknown_{class_id}")
