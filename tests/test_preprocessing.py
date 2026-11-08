"""Unit tests for src/preprocessing.py (Dark Channel Prior & CLAHE)."""

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    apply_clahe,
    compute_colorfulness,
    compute_contrast,
    compute_dark_channel,
    compute_entropy,
    compute_image_quality_metrics,
    dark_channel_prior_correction,
    estimate_atmospheric_light,
    estimate_transmission,
    guided_filter,
    preprocess_image,
)


def test_compute_dark_channel_shape(sample_bgr_image: np.ndarray) -> None:
    """Dark channel output must be a 2D array with same (H, W) as input."""
    img_float = sample_bgr_image.astype(np.float64) / 255.0
    dark = compute_dark_channel(img_float, patch_size=15)
    assert dark.ndim == 2
    assert dark.shape == sample_bgr_image.shape[:2]


def test_compute_dark_channel_values(sample_bgr_image: np.ndarray) -> None:
    """Dark channel pixel values must lie in [0, 1]."""
    img_float = sample_bgr_image.astype(np.float64) / 255.0
    dark = compute_dark_channel(img_float, patch_size=15)
    assert dark.min() >= 0.0
    assert dark.max() <= 1.0


def test_dark_channel_uniform_image() -> None:
    """A uniform bright image should produce a high dark channel everywhere."""
    uniform = np.ones((64, 64, 3), dtype=np.float64) * 0.8
    dark = compute_dark_channel(uniform, patch_size=7)
    np.testing.assert_allclose(dark, 0.8, atol=1e-5)


def test_estimate_atmospheric_light_shape(sample_bgr_image: np.ndarray) -> None:
    """Atmospheric light vector must have shape (3,) with non-negative entries."""
    img_float = sample_bgr_image.astype(np.float64) / 255.0
    dark = compute_dark_channel(img_float, patch_size=15)
    atm_light = estimate_atmospheric_light(img_float, dark, top_percent=0.1)
    assert atm_light.shape == (3,)
    assert np.all(atm_light >= 0.0)
    assert np.all(atm_light <= 1.0)


def test_estimate_transmission_range(sample_bgr_image: np.ndarray) -> None:
    """Transmission map must be strictly bounded in [0, 1]."""
    img_float = sample_bgr_image.astype(np.float64) / 255.0
    dark = compute_dark_channel(img_float, patch_size=15)
    atm_light = estimate_atmospheric_light(img_float, dark)
    transmission = estimate_transmission(img_float, atm_light, patch_size=15, omega=0.95)
    assert transmission.ndim == 2
    assert transmission.shape == sample_bgr_image.shape[:2]
    assert np.all(transmission >= 0.0)
    assert np.all(transmission <= 1.0)


def test_guided_filter_preserves_shape(sample_bgr_image: np.ndarray) -> None:
    """Guided filter output must match source image dimensions."""
    gray = cv2.cvtColor(sample_bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    src = gray.copy()
    filtered = guided_filter(guide=gray, src=src, radius=15, eps=1e-3)
    assert filtered.shape == src.shape
    assert not np.isnan(filtered).any()


def test_dark_channel_prior_correction_output(sample_underwater_image: np.ndarray) -> None:
    """DCP correction must output a uint8 BGR image of identical dimensions."""
    corrected = dark_channel_prior_correction(sample_underwater_image)
    assert isinstance(corrected, np.ndarray)
    assert corrected.dtype == np.uint8
    assert corrected.shape == sample_underwater_image.shape


def test_apply_clahe_output(sample_underwater_image: np.ndarray) -> None:
    """CLAHE must return uint8 BGR image preserving dimensions."""
    enhanced = apply_clahe(sample_underwater_image, clip_limit=3.0, tile_grid_size=(8, 8))
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.uint8
    assert enhanced.shape == sample_underwater_image.shape


def test_apply_clahe_enhances_contrast(sample_underwater_image: np.ndarray) -> None:
    """CLAHE on L channel should generally increase standard deviation of L."""
    lab_orig = cv2.cvtColor(sample_underwater_image, cv2.COLOR_BGR2LAB)
    enhanced = apply_clahe(sample_underwater_image, clip_limit=4.0, tile_grid_size=(8, 8))
    lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

    orig_l_std = lab_orig[:, :, 0].std()
    enh_l_std = lab_enh[:, :, 0].std()
    assert enh_l_std >= orig_l_std * 0.95  # Contrast should not degrade


def test_preprocess_image_resize(sample_bgr_image: np.ndarray, mock_config) -> None:
    """Pipeline must resize output to specified target dimensions."""
    target_dim = 96
    out = preprocess_image(
        sample_bgr_image,
        cfg=mock_config,
        apply_dark_channel=True,
        apply_clahe_flag=True,
        target_size=target_dim,
    )
    assert out.shape == (target_dim, target_dim, 3)
    assert out.dtype == np.uint8


def test_compute_entropy_range(sample_bgr_image: np.ndarray) -> None:
    """Entropy must be strictly non-negative."""
    ent = compute_entropy(sample_bgr_image)
    assert ent >= 0.0


def test_compute_entropy_uniform_image() -> None:
    """A completely uniform image has zero entropy (no uncertainty)."""
    uniform = np.full((64, 64), 128, dtype=np.uint8)
    ent = compute_entropy(uniform)
    assert ent == pytest.approx(0.0, abs=1e-5)


def test_compute_contrast_nonneg(sample_bgr_image: np.ndarray) -> None:
    """RMS contrast metric must be non-negative."""
    contrast = compute_contrast(sample_bgr_image)
    assert contrast >= 0.0


def test_compute_colorfulness_nonneg(sample_bgr_image: np.ndarray) -> None:
    """Colorfulness score must be non-negative."""
    colorfulness = compute_colorfulness(sample_bgr_image)
    assert colorfulness >= 0.0


def test_compute_image_quality_metrics_keys(sample_underwater_image: np.ndarray) -> None:
    """Quality metrics dictionary must contain entropy, contrast, colorfulness."""
    metrics = compute_image_quality_metrics(sample_underwater_image)
    assert "entropy" in metrics
    assert "contrast" in metrics
    assert "colorfulness" in metrics
    for v in metrics.values():
        assert isinstance(v, float)

