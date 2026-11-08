"""
Utility functions for the Marine Reef Vision project.

Provides:
    - Reproducible seed setting across Python, NumPy, PyTorch
    - YAML configuration loading with OmegaConf
    - Logging setup with consistent formatting
    - Device detection (CUDA / CPU)
    - Effective config resolution (CPU mode overrides)
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Set safe writable matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations for reproducibility (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging with a consistent format.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to a log file.
    """
    fmt = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logger.info("Logging initialized (level=%s)", logging.getLevelName(level))


def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Load and return the project configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        OmegaConf DictConfig object.
    """
    path = Path(config_path)
    if not path.exists():
        for candidate_dir in [Path("AquaLens"), Path("marine-reef-vision"), Path(__file__).resolve().parent.parent]:
            alt = candidate_dir / config_path
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)
    logger.info("Configuration loaded from %s", path.resolve())
    return cfg


def get_device(cfg: DictConfig) -> torch.device:
    """Determine the compute device from config.

    Args:
        cfg: Project configuration.

    Returns:
        torch.device for training/inference.
    """
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_mem / 1e9,
        )
    return device


def get_effective_config(cfg: DictConfig) -> DictConfig:
    """Resolve CPU-mode overrides and return the effective configuration.

    When cpu_mode is True or device is CPU, this adjusts image size,
    batch size, epochs, and disables mixed precision.

    Args:
        cfg: Raw project configuration.

    Returns:
        Resolved configuration with CPU overrides applied if needed.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    device = get_device(cfg)

    is_cpu = cfg.get("cpu_mode", False) or device.type == "cpu"

    # Resolve effective values
    if is_cpu:
        cfg._effective_image_size = cfg.data.image_size_cpu
        cfg._effective_batch_size = cfg.training.batch_size_cpu
        cfg._effective_epochs = cfg.training.epochs_cpu
        cfg._effective_mixed_precision = False
        logger.info(
            "CPU mode: image_size=%d, batch_size=%d, epochs=%d",
            cfg._effective_image_size,
            cfg._effective_batch_size,
            cfg._effective_epochs,
        )
    else:
        cfg._effective_image_size = cfg.data.image_size_gpu
        cfg._effective_batch_size = cfg.training.batch_size_gpu
        cfg._effective_epochs = cfg.training.epochs_gpu
        cfg._effective_mixed_precision = cfg.training.mixed_precision
        logger.info(
            "GPU mode: image_size=%d, batch_size=%d, epochs=%d, mixed_precision=%s",
            cfg._effective_image_size,
            cfg._effective_batch_size,
            cfg._effective_epochs,
            cfg._effective_mixed_precision,
        )

    cfg._device = str(device)
    return cfg


def ensure_dirs(cfg: DictConfig) -> None:
    """Create all output directories specified in config.

    Args:
        cfg: Project configuration.
    """
    for key in ["output_dir", "checkpoint_dir", "figures_dir", "gradcam_dir", "logs_dir"]:
        p = Path(cfg.paths.get(key, f"outputs/{key}"))
        p.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created/verified")


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metric names to values.
        path: Output file path.
    """
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", path)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total', 'trainable', and 'frozen' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


if __name__ == "__main__":
    setup_logging()
    cfg = load_config()
    cfg = get_effective_config(cfg)
    ensure_dirs(cfg)
    logger.info("Utils module self-test passed")

