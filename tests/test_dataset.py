"""Unit tests for src/dataset.py (Data loading, splits, augmentations)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.dataset import (
    MarineDataset,
    PlaceholderDatasetGenerator,
    UnderwaterColorCast,
    UnderwaterTurbidity,
    create_stratified_splits,
)


def test_placeholder_generator_creates_files(tmp_path: Path, mock_config) -> None:
    """Placeholder generator must create expected class folders and image files."""
    cfg = OmegaConf.create(OmegaConf.to_container(mock_config))
    cfg.paths.fish_data = str(tmp_path / "fish")
    cfg.paths.coral_data = str(tmp_path / "coral")

    generator = PlaceholderDatasetGenerator(cfg, images_per_class=4, image_size=(64, 64))
    generator.generate_fish_images()
    generator.generate_coral_images()

    for cls_name in generator.FISH_CLASSES:
        cls_dir = Path(cfg.paths.fish_data) / cls_name
        assert cls_dir.exists(), f"Class folder missing: {cls_dir}"
        assert len(list(cls_dir.glob("*.png"))) == 4

    for label in ["healthy", "bleached"]:
        cls_dir = Path(cfg.paths.coral_data) / label
        assert cls_dir.exists(), f"Coral folder missing: {cls_dir}"
        assert len(list(cls_dir.glob("*.png"))) == 4


def test_underwater_color_cast_transform(sample_bgr_image: np.ndarray) -> None:
    """UnderwaterColorCast transform must preserve tensor shape and valid [0, 1] range."""
    tensor = torch.from_numpy(sample_bgr_image).permute(2, 0, 1).float() / 255.0
    transform = UnderwaterColorCast(p=1.0)
    out = transform(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == tensor.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_underwater_turbidity_transform(sample_bgr_image: np.ndarray) -> None:
    """UnderwaterTurbidity transform must apply blur/noise while preserving shape."""
    tensor = torch.from_numpy(sample_bgr_image).permute(2, 0, 1).float() / 255.0
    transform = UnderwaterTurbidity(p=1.0, blur_kernel_size=5)
    out = transform(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == tensor.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_create_stratified_splits(tmp_path: Path) -> None:
    """Stratified splits must partition data into train/val/test according to ratios."""
    # Create mock dataset with 2 classes, 20 items each
    for cls in ["cls_a", "cls_b"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True)
        for i in range(20):
            (cls_dir / f"img_{i:02d}.jpg").write_text("dummy")

    splits = create_stratified_splits(
        dataset_dir=tmp_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        save_json=False,
    )

    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]) == 28  # 14 per class * 2
    assert len(splits["val"]) == 6     # 3 per class * 2
    assert len(splits["test"]) == 6    # 3 per class * 2


def test_marine_dataset_returns_tensor_and_label(tmp_path: Path, mock_config) -> None:
    """MarineDataset must return (torch.FloatTensor [3, H, W], int_label)."""
    cls_dir = tmp_path / "mock_class"
    cls_dir.mkdir(parents=True)
    img_bgr = np.full((64, 64, 3), 120, dtype=np.uint8)
    img_path = str(cls_dir / "sample.png")
    cv2.imwrite(img_path, img_bgr)

    samples = [(img_path, 0)]
    dataset = MarineDataset(
        samples=samples,
        cfg=mock_config,
        split="val",
        apply_dcp=False,
        apply_clahe=False,
    )

    assert len(dataset) == 1
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, dataset._image_size, dataset._image_size)
    assert tensor.dtype == torch.float32
    assert label == 0
