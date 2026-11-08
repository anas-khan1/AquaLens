"""Pytest shared fixtures for Marine Reef Vision test suite."""

import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def sample_bgr_image() -> np.ndarray:
    """Fixture providing a synthetic 128x128 BGR image with geometric shapes."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = (120, 80, 40)
    cv2.circle(img, (64, 64), 30, (200, 150, 100), -1)
    cv2.rectangle(img, (20, 20), (50, 50), (40, 200, 120), -1)
    return img


@pytest.fixture(scope="session")
def sample_underwater_image() -> np.ndarray:
    """Fixture providing a synthetic underwater image with blue/cyan cast and low contrast."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Underwater profile: high blue, medium green, low red
    img[:, :, 0] = 160  # Blue
    img[:, :, 1] = 110  # Green
    img[:, :, 2] = 45   # Red
    # Add a blurry foreground object
    cv2.circle(img, (64, 64), 35, (130, 95, 35), -1)
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture(scope="session")
def mock_config() -> DictConfig:
    """Fixture providing a valid minimal DictConfig for testing."""
    config_dict = {
        "project_name": "marine-reef-vision-test",
        "seed": 42,
        "device": "cpu",
        "cpu_mode": True,
        "num_workers": 0,
        "pin_memory": False,
        "paths": {
            "data_root": "data",
            "fish_data": "data/fish",
            "coral_data": "data/coral",
            "output_dir": "outputs",
            "checkpoint_dir": "outputs/checkpoints",
            "figures_dir": "outputs/figures",
            "gradcam_dir": "outputs/gradcam_samples",
            "logs_dir": "outputs/logs",
            "metrics_file": "outputs/metrics.json",
        },
        "data": {
            "fish_dataset_slug": "crowai/large-scale-fish-dataset",
            "coral_dataset_slug": "sonainjamil/coral-reef-bleaching-dataset",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "cpu_subset_per_class": 10,
            "image_size_gpu": 224,
            "image_size_cpu": 128,
        },
        "preprocessing": {
            "dark_channel": {
                "patch_size": 15,
                "omega": 0.95,
                "t_min": 0.1,
                "guided_filter_radius": 30,
                "guided_filter_eps": 0.001,
            },
            "clahe": {
                "clip_limit": 3.0,
                "tile_grid_size": [8, 8],
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.2,
            "rotation_degrees": 15,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
            "underwater_color_cast_p": 0.3,
            "underwater_turbidity_p": 0.2,
        },
        "model": {
            "backbone": "resnet34",
            "pretrained": False,
            "unfreeze_schedule": {
                0: ["fc"],
                1: ["layer4", "fc"],
                2: "all",
            },
            "dropout": 0.2,
        },
        "training": {
            "batch_size_gpu": 16,
            "epochs_gpu": 2,
            "batch_size_cpu": 8,
            "epochs_cpu": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "max_lr": 0.005,
            "pct_start": 0.3,
            "mixed_precision": False,
            "early_stopping": {
                "patience": 3,
                "min_delta": 0.001,
                "monitor": "val_loss",
            },
            "use_weighted_sampler": True,
        },
        "evaluation": {
            "top_k_accuracy": [1, 3],
            "compute_roc_auc": True,
            "confusion_matrix": True,
        },
        "gradcam": {
            "target_layer": "layer4",
            "num_samples": 5,
            "alpha": 0.5,
        },
        "severity": {
            "thresholds": {
                "healthy": 0.15,
                "mild": 0.35,
                "moderate": 0.60,
            },
            "pale_pixel": {
                "min_value": 180,
                "max_saturation": 50,
            },
        },
        "ablation": {
            "variants": [
                {"name": "raw", "dark_channel": False, "clahe": False},
                {"name": "clahe_only", "dark_channel": False, "clahe": True},
                {"name": "dcp_only", "dark_channel": True, "clahe": False},
                {"name": "dcp_clahe", "dark_channel": True, "clahe": True},
            ]
        },
    }
    return OmegaConf.create(config_dict)

