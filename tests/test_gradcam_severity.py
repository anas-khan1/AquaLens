"""Unit tests for src/gradcam.py and src/severity.py."""

import numpy as np
import pytest
import torch

from src.gradcam import GradCAM, overlay_heatmap
from src.model import create_model
from src.severity import (
    classify_severity,
    compute_bleaching_score,
    compute_reef_health_index,
    detect_pale_pixels,
)


def test_gradcam_generation() -> None:
    """GradCAM must produce a 2D float32 heatmap bounded in [0, 1]."""
    model = create_model("fish", num_classes=3, pretrained=False)
    gradcam = GradCAM(model, target_layer_name="layer4")

    dummy_tensor = torch.randn(1, 3, 64, 64)
    heatmap = gradcam.generate(dummy_tensor, target_class=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5
    gradcam.remove_hooks()


def test_overlay_heatmap(sample_bgr_image: np.ndarray) -> None:
    """Overlay must create a valid uint8 BGR image matching input dimensions."""
    h, w = sample_bgr_image.shape[:2]
    heatmap = np.linspace(0, 1, h * w).reshape(h, w).astype(np.float32)

    overlay = overlay_heatmap(sample_bgr_image, heatmap, alpha=0.5)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == sample_bgr_image.shape
    assert overlay.dtype == np.uint8


def test_detect_pale_pixels() -> None:
    """Pale pixel detection must correctly isolate desaturated high-value pixels."""
    # Create an image with a white square on a saturated dark red background
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 200)      # Saturated red (BGR)
    img[10:30, 10:30] = (240, 240, 240)  # White patch

    mask = detect_pale_pixels(img, min_value=180, max_saturation=50)
    assert mask.shape == (50, 50)
    assert mask.dtype == np.uint8
    assert mask[20, 20] == 255  # Inside white patch
    assert mask[0, 0] == 0      # Outside


def test_classify_severity() -> None:
    """Severity classification thresholds must map properly to labels and indices."""
    lbl, idx = classify_severity(0.05, threshold_healthy=0.15, threshold_mild=0.35, threshold_moderate=0.60)
    assert lbl == "Healthy" and idx == 0

    lbl, idx = classify_severity(0.25, threshold_healthy=0.15, threshold_mild=0.35, threshold_moderate=0.60)
    assert lbl == "Mild Bleaching" and idx == 1

    lbl, idx = classify_severity(0.45, threshold_healthy=0.15, threshold_mild=0.35, threshold_moderate=0.60)
    assert lbl == "Moderate Bleaching" and idx == 2

    lbl, idx = classify_severity(0.85, threshold_healthy=0.15, threshold_mild=0.35, threshold_moderate=0.60)
    assert lbl == "Severe Bleaching" and idx == 3


def test_compute_reef_health_index() -> None:
    """Reef Health Index (RHI) should weight healthy reefs near 1.0 and severe near 0.0."""
    assessments_healthy = [
        {"severity_index": 0},
        {"severity_index": 0},
    ]
    res_healthy = compute_reef_health_index(assessments_healthy)
    assert res_healthy["rhi"] == pytest.approx(1.0)

    assessments_mixed = [
        {"severity_index": 0},  # weight 1.0
        {"severity_index": 3},  # weight 0.0
    ]
    res_mixed = compute_reef_health_index(assessments_mixed)
    assert res_mixed["rhi"] == pytest.approx(0.5)
