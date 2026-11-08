"""Unit tests for src/model.py (Architecture, Progressive unfreezing, Checkpointing)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.model import (
    MarineClassifier,
    ProgressiveUnfreezer,
    create_model,
    load_checkpoint,
    save_checkpoint,
)


def test_create_model_output_shape() -> None:
    """MarineClassifier output shape must match (batch_size, num_classes)."""
    model = create_model(task_name="fish", num_classes=5, pretrained=False, dropout=0.2)
    dummy_input = torch.randn(2, 3, 64, 64)
    out = model(dummy_input)
    assert out.shape == (2, 5)


def test_progressive_unfreezer() -> None:
    """ProgressiveUnfreezer should correctly enable/disable gradients according to schedule."""
    model = create_model(task_name="fish", num_classes=3, pretrained=False)
    schedule = {
        0: ["fc"],
        1: ["layer4", "fc"],
        2: "all",
    }
    unfreezer = ProgressiveUnfreezer(model, schedule)

    # Epoch 0: only fc trainable
    unfreezer.step(epoch=0)
    for name, p in model.named_parameters():
        if "fc" in name:
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False

    # Epoch 1: layer4 + fc trainable
    unfreezer.step(epoch=1)
    for name, p in model.named_parameters():
        if "fc" in name or "layer4" in name:
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False

    # Epoch 2: all trainable
    unfreezer.step(epoch=2)
    for name, p in model.named_parameters():
        assert p.requires_grad is True


def test_save_and_load_checkpoint(tmp_path: Path, mock_config) -> None:
    """Checkpoints must preserve model weights and metadata faithfully."""
    model = create_model(task_name="coral", num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ckpt_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_metric=0.88,
        cfg=mock_config,
        checkpoint_dir=tmp_path,
        filename="test_model.pth",
    )

    assert ckpt_path.exists()
    new_model = create_model(task_name="coral", num_classes=2, pretrained=False)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    meta = load_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=new_model,
        optimizer=new_optimizer,
    )

    assert meta["epoch"] == 3
    assert meta["best_metric"] == pytest.approx(0.88)
