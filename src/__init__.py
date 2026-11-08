# Marine Reef Vision — Source Package
"""
Underwater Marine Species and Coral Reef Health Visual Assessment.

Modules:
    preprocessing  – Dark Channel Prior & CLAHE image enhancement
    dataset        – Data loading, splits, and augmentation
    model          – ResNet-34 model factory with progressive unfreezing
    train          – Training loop with mixed precision & early stopping
    evaluate       – Metrics, confusion matrices, and per-class reports
    gradcam        – Grad-CAM heatmap generation
    severity       – Coral bleaching severity scoring (heuristic)
    utils          – Seed setting, config loading, logging setup
"""

__version__ = "1.0.0"

