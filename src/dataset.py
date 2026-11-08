"""
Dataset module for the Marine Reef Vision project.

Provides:
    - KaggleDownloader: Downloads fish and coral reef datasets from Kaggle.
    - PlaceholderDatasetGenerator: Creates synthetic placeholder data for
      pipeline testing when real data is unavailable.
    - create_stratified_splits: Creates reproducible 70/15/15 stratified splits.
    - MarineDataset: PyTorch Dataset with underwater-specific augmentations.
    - UnderwaterColorCast / UnderwaterTurbidity: Custom torchvision transforms
      simulating underwater optical degradations.
    - get_dataloaders: Factory for train/val/test DataLoaders with
      WeightedRandomSampler support.
    - print_dataset_summary: Pretty-prints class distributions and image stats.
"""

import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)


# =============================================================================
# Kaggle Dataset Downloader
# =============================================================================

class KaggleDownloader:
    """Downloads datasets from Kaggle using the Kaggle API.

    Reads KAGGLE_USERNAME and KAGGLE_KEY from environment variables or
    from the ``~/.kaggle/kaggle.json`` file.  Falls back gracefully with
    instructions when credentials are missing.

    Attributes:
        data_root: Root directory where datasets are stored.
        fish_slug: Kaggle dataset slug for the fish dataset.
        coral_slug: Kaggle dataset slug for the coral dataset.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the downloader from project configuration.

        Args:
            cfg: Project configuration (``config.yaml``).
        """
        self.data_root: Path = Path(cfg.paths.data_root)
        self.fish_slug: str = cfg.data.fish_dataset_slug
        self.coral_slug: str = cfg.data.coral_dataset_slug
        self.fish_dir: Path = Path(cfg.paths.fish_data)
        self.coral_dir: Path = Path(cfg.paths.coral_data)

    # --------------------------------------------------------------------- #
    # Credential helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _has_credentials() -> bool:
        """Check whether Kaggle credentials are available.

        Returns:
            True if both ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` are set in
            the environment **or** a valid ``~/.kaggle/kaggle.json`` exists.
        """
        # Check environment variables first
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True

        # Fall back to kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                with open(kaggle_json, "r", encoding="utf-8") as f:
                    creds = json.load(f)
                if creds.get("username") and creds.get("key"):
                    return True
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read %s: %s", kaggle_json, exc)

        return False

    @staticmethod
    def print_instructions() -> None:
        """Print manual download instructions when Kaggle auth fails."""
        instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE CREDENTIALS NOT FOUND                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Option 1 — Environment variables:                                   ║
║      export KAGGLE_USERNAME="your_username"                          ║
║      export KAGGLE_KEY="your_api_key"                                ║
║                                                                      ║
║  Option 2 — kaggle.json file:                                        ║
║      1. Go to https://www.kaggle.com/settings → API → Create Token   ║
║      2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json     ║
║      3. On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json                ║
║                                                                      ║
║  Option 3 — Manual download:                                        ║
║      Fish:  https://www.kaggle.com/datasets/crowai/large-scale-fish-dataset  ║
║      Coral: https://www.kaggle.com/datasets/sonainjamil/coral-reef-bleaching-dataset ║
║                                                                      ║
║      Extract to:                                                     ║
║        data/fish/   (with class sub-folders)                         ║
║        data/coral/  (with healthy/bleached sub-folders)              ║
║                                                                      ║
║  The pipeline will fall back to placeholder data generation.         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        logger.warning(instructions)

    # --------------------------------------------------------------------- #
    # Download methods
    # --------------------------------------------------------------------- #

    def _download_dataset(self, slug: str, dest: Path) -> bool:
        """Download and unzip a single Kaggle dataset.

        Args:
            slug: Kaggle dataset slug (e.g. ``user/dataset-name``).
            dest: Destination directory.

        Returns:
            True if download succeeded, False otherwise.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

            api = KaggleApi()
            api.authenticate()

            dest.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading dataset '%s' → %s", slug, dest)
            api.dataset_download_files(slug, path=str(dest), unzip=True)
            logger.info("Download complete: %s", slug)
            return True

        except ImportError:
            logger.error(
                "The 'kaggle' package is not installed.  "
                "Run: pip install kaggle"
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Kaggle download failed for '%s': %s", slug, exc)
            return False

    def download_fish(self) -> bool:
        """Download the large-scale fish dataset.

        Returns:
            True on success.
        """
        if self.fish_dir.exists() and any(self.fish_dir.iterdir()):
            logger.info("Fish data already present at %s — skipping download.", self.fish_dir)
            return True
        return self._download_dataset(self.fish_slug, self.fish_dir)

    def download_coral(self) -> bool:
        """Download the coral reef bleaching dataset.

        Returns:
            True on success.
        """
        if self.coral_dir.exists() and any(self.coral_dir.iterdir()):
            logger.info("Coral data already present at %s — skipping download.", self.coral_dir)
            return True
        return self._download_dataset(self.coral_slug, self.coral_dir)

    def download_all(self) -> bool:
        """Download both fish and coral datasets.

        Returns:
            True if **both** downloads succeeded.
        """
        if not self._has_credentials():
            self.print_instructions()
            return False

        fish_ok = self.download_fish()
        coral_ok = self.download_coral()
        return fish_ok and coral_ok


# =============================================================================
# Placeholder (Synthetic) Dataset Generator
# =============================================================================

class PlaceholderDatasetGenerator:
    """Generates synthetic placeholder images for pipeline testing.

    Fish images are coloured rectangles with simple fish-like shapes
    drawn using OpenCV drawing primitives.  Coral images are textured
    rectangles — green-ish for healthy, white-ish for bleached.

    Attributes:
        fish_dir: Output directory for fish class folders.
        coral_dir: Output directory for coral class folders.
        images_per_class: Number of images to generate per class.
        image_size: Generated image dimensions ``(H, W)``.
    """

    # Five synthetic fish classes
    FISH_CLASSES: List[str] = [
        "clownfish",
        "blue_tang",
        "lionfish",
        "parrotfish",
        "grouper",
    ]

    # Base colours (BGR) for each fake fish class
    _FISH_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "clownfish": (20, 80, 220),     # orange-red
        "blue_tang": (200, 140, 30),     # blue
        "lionfish": (60, 60, 180),       # reddish-brown
        "parrotfish": (100, 200, 50),    # greenish
        "grouper": (80, 80, 80),         # grey-brown
    }

    def __init__(
        self,
        cfg: DictConfig,
        images_per_class: int = 50,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialise the generator.

        Args:
            cfg: Project configuration.
            images_per_class: Number of images per class to generate.
            image_size: ``(height, width)`` of generated images.
        """
        self.fish_dir: Path = Path(cfg.paths.fish_data)
        self.coral_dir: Path = Path(cfg.paths.coral_data)
        self.images_per_class: int = images_per_class
        self.image_size: Tuple[int, int] = image_size

    # --------------------------------------------------------------------- #
    # Fish generation
    # --------------------------------------------------------------------- #

    def _draw_fish_shape(
        self,
        img: np.ndarray,
        colour: Tuple[int, int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a simple fish-like shape on the image.

        Uses an ellipse for the body and a triangle for the tail, with
        random position / size jitter for variability.

        Args:
            img: Background image to draw on (modified in-place).
            colour: Base BGR colour for the fish body.
            rng: NumPy random generator for reproducibility.

        Returns:
            Image with fish shape drawn.
        """
        h, w = img.shape[:2]

        # Random centre and size
        cx = rng.integers(w // 4, 3 * w // 4)
        cy = rng.integers(h // 4, 3 * h // 4)
        body_w = rng.integers(w // 5, w // 3)
        body_h = rng.integers(h // 8, h // 4)

        # Slight colour jitter
        jitter = rng.integers(-20, 21, size=3)
        fish_colour = tuple(
            int(np.clip(c + j, 0, 255)) for c, j in zip(colour, jitter)
        )

        # Body — filled ellipse
        cv2.ellipse(
            img,
            (int(cx), int(cy)),
            (int(body_w // 2), int(body_h // 2)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=fish_colour,
            thickness=-1,
        )

        # Tail — filled triangle behind the body
        tail_x = int(cx + body_w // 2)
        tail_pts = np.array([
            [tail_x, int(cy)],
            [int(tail_x + body_w // 3), int(cy - body_h // 2)],
            [int(tail_x + body_w // 3), int(cy + body_h // 2)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [tail_pts], color=fish_colour)

        # Eye — small white circle
        eye_x = int(cx - body_w // 4)
        eye_y = int(cy - body_h // 6)
        cv2.circle(img, (eye_x, eye_y), max(2, int(body_h // 10)), (255, 255, 255), -1)
        cv2.circle(img, (eye_x, eye_y), max(1, int(body_h // 16)), (0, 0, 0), -1)

        return img

    def generate_fish_images(self) -> None:
        """Generate synthetic fish images for all fake classes.

        Images are saved to ``{fish_dir}/{class_name}/img_NNNN.png``.
        """
        logger.info(
            "Generating placeholder fish images: %d classes × %d images",
            len(self.FISH_CLASSES),
            self.images_per_class,
        )
        rng = np.random.default_rng(seed=42)

        for cls_name in self.FISH_CLASSES:
            cls_dir = self.fish_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            base_colour = self._FISH_COLOURS[cls_name]

            for idx in range(self.images_per_class):
                # Blue-ish underwater background with random noise
                bg_blue = rng.integers(100, 180)
                bg_green = rng.integers(60, 120)
                bg_red = rng.integers(20, 60)
                img = np.full(
                    (*self.image_size, 3),
                    (bg_blue, bg_green, bg_red),
                    dtype=np.uint8,
                )
                # Add slight Gaussian noise for texture
                noise = rng.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Draw 1–2 fish shapes
                n_fish = rng.integers(1, 3)
                for _ in range(n_fish):
                    self._draw_fish_shape(img, base_colour, rng)

                out_path = cls_dir / f"img_{idx:04d}.png"
                cv2.imwrite(str(out_path), img)

            logger.info("  %-12s → %s (%d images)", cls_name, cls_dir, self.images_per_class)

    # --------------------------------------------------------------------- #
    # Coral generation
    # --------------------------------------------------------------------- #

    def _generate_coral_texture(
        self,
        healthy: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a textured coral-like image.

        Healthy coral has a green/brown tint; bleached coral is pale/white.

        Args:
            healthy: If True, generate healthy (colourful) coral texture.
            rng: NumPy random generator.

        Returns:
            Synthetic coral image as uint8 BGR array.
        """
        h, w = self.image_size

        if healthy:
            # Green-ish / brown-ish tones
            base_b = rng.integers(30, 70)
            base_g = rng.integers(100, 170)
            base_r = rng.integers(50, 100)
        else:
            # White-ish / pale tones (bleached)
            base_val = rng.integers(180, 240)
            base_b = base_val + rng.integers(-10, 11)
            base_g = base_val + rng.integers(-10, 11)
            base_r = base_val + rng.integers(-10, 11)

        img = np.full((h, w, 3), (base_b, base_g, base_r), dtype=np.uint8)

        # Add bumpy texture using random circles (simulating coral polyps)
        n_circles = rng.integers(30, 80)
        for _ in range(n_circles):
            cx = rng.integers(0, w)
            cy = rng.integers(0, h)
            radius = rng.integers(3, max(4, min(h, w) // 10))
            jitter = rng.integers(-25, 26, size=3)
            colour = tuple(
                int(np.clip(c + j, 0, 255))
                for c, j in zip((base_b, base_g, base_r), jitter)
            )
            cv2.circle(img, (int(cx), int(cy)), int(radius), colour, -1)

        # Gaussian noise for texture variation
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def generate_coral_images(self) -> None:
        """Generate synthetic coral images (healthy + bleached).

        Images are saved to ``{coral_dir}/healthy/`` and
        ``{coral_dir}/bleached/``.
        """
        logger.info(
            "Generating placeholder coral images: 2 classes × %d images",
            self.images_per_class,
        )
        rng = np.random.default_rng(seed=123)

        for label, healthy in [("healthy", True), ("bleached", False)]:
            cls_dir = self.coral_dir / label
            cls_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(self.images_per_class):
                img = self._generate_coral_texture(healthy, rng)
                out_path = cls_dir / f"img_{idx:04d}.png"
                cv2.imwrite(str(out_path), img)

            logger.info("  %-12s → %s (%d images)", label, cls_dir, self.images_per_class)

    def generate_all(self) -> None:
        """Generate both fish and coral placeholder datasets."""
        self.generate_fish_images()
        self.generate_coral_images()
        logger.info("Placeholder dataset generation complete.")


# =============================================================================
# Stratified Splits
# =============================================================================

def create_stratified_splits(
    dataset_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_json: bool = True,
) -> Dict[str, List[Tuple[str, int]]]:
    """Create stratified train/val/test splits from a directory of class folders.

    The directory is expected to have the structure::

        dataset_dir/
            class_a/
                img_001.png
                ...
            class_b/
                img_001.png
                ...

    Args:
        dataset_dir: Root directory containing one sub-folder per class.
        train_ratio: Fraction of data for the training split.
        val_ratio: Fraction of data for the validation split.
        test_ratio: Fraction of data for the test split.
        seed: Random seed for reproducibility.
        save_json: Whether to save the split as a JSON file alongside
            ``dataset_dir``.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each mapping
        to a list of ``(image_path, label_index)`` tuples.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within tolerance).
        FileNotFoundError: If ``dataset_dir`` does not exist or is empty.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    # Discover classes (sorted for deterministic label assignment)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    class_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not class_dirs:
        raise FileNotFoundError(f"No class sub-directories found in {root}")

    # Build class-to-index mapping
    class_to_idx: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    logger.info("Found %d classes: %s", len(class_to_idx), list(class_to_idx.keys()))

    # Collect samples per class
    rng = random.Random(seed)
    splits: Dict[str, List[Tuple[str, int]]] = {"train": [], "val": [], "test": []}

    for cls_dir in class_dirs:
        label = class_to_idx[cls_dir.name]
        images = sorted(
            [
                str(p)
                for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )

        if not images:
            logger.warning("Class '%s' has no valid images — skipping.", cls_dir.name)
            continue

        # Shuffle deterministically, then split
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining go to test to handle rounding
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        splits["train"].extend((p, label) for p in train_imgs)
        splits["val"].extend((p, label) for p in val_imgs)
        splits["test"].extend((p, label) for p in test_imgs)

        logger.debug(
            "  %-20s  total=%d  train=%d  val=%d  test=%d",
            cls_dir.name, n, len(train_imgs), len(val_imgs), len(test_imgs),
        )

    # Shuffle each split internally
    for key in splits:
        rng.shuffle(splits[key])

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # Save split metadata
    if save_json:
        split_info = {
            "class_to_idx": class_to_idx,
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "counts": {k: len(v) for k, v in splits.items()},
            "splits": {k: [(p, l) for p, l in v] for k, v in splits.items()},
        }
        json_path = root / "split_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)
        logger.info("Split info saved to %s", json_path)

    return splits


# =============================================================================
# Custom Underwater Augmentation Transforms
# =============================================================================

class UnderwaterColorCast(torch.nn.Module):
    """Randomly shifts blue and green channels to simulate underwater colour cast.

    Underwater images typically exhibit a blue or green colour cast due to
    wavelength-dependent absorption.  This transform randomly boosts the
    blue and/or green channels by a small multiplicative factor.

    Args:
        blue_range: ``(min, max)`` multiplicative factor for the blue channel.
        green_range: ``(min, max)`` multiplicative factor for the green channel.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        blue_range: Tuple[float, float] = (1.0, 1.3),
        green_range: Tuple[float, float] = (1.0, 1.2),
        p: float = 0.3,
    ) -> None:
        super().__init__()
        self.blue_range = blue_range
        self.green_range = green_range
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply underwater colour cast.

        Args:
            img: Tensor of shape ``(C, H, W)`` with values in ``[0, 1]``.

        Returns:
            Colour-shifted tensor, clamped to ``[0, 1]``.
        """
        if torch.rand(1).item() > self.p:
            return img

        # img is (C, H, W) with C = [R, G, B] in torchvision convention
        blue_factor = (
            torch.empty(1).uniform_(self.blue_range[0], self.blue_range[1]).item()
        )
        green_factor = (
            torch.empty(1).uniform_(self.green_range[0], self.green_range[1]).item()
        )

        # Channel order for torchvision: [R=0, G=1, B=2]
        result = img.clone()
        result[1] = result[1] * green_factor  # Green
        result[2] = result[2] * blue_factor   # Blue

        return torch.clamp(result, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"blue_range={self.blue_range}, "
            f"green_range={self.green_range}, "
            f"p={self.p})"
        )


class UnderwaterTurbidity(torch.nn.Module):
    """Simulates underwater turbidity via Gaussian blur and additive noise.

    Turbid water scatters light, reducing contrast and introducing
    sensor noise.  This transform:

    1. Applies a small Gaussian blur (simulating forward scatter).
    2. Adds Gaussian noise (simulating backscatter / sensor noise).

    Args:
        blur_kernel_size: Kernel size for Gaussian blur (must be odd).
        blur_sigma_range: ``(min, max)`` for blur sigma.
        noise_std_range: ``(min, max)`` standard deviation of additive noise.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        blur_kernel_size: int = 3,
        blur_sigma_range: Tuple[float, float] = (0.1, 1.5),
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        p: float = 0.2,
    ) -> None:
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply turbidity simulation.

        Args:
            img: Tensor of shape ``(C, H, W)`` with values in ``[0, 1]``.

        Returns:
            Degraded tensor, clamped to ``[0, 1]``.
        """
        if torch.rand(1).item() > self.p:
            return img

        result = img.clone()

        # --- Gaussian blur ---
        sigma = (
            torch.empty(1)
            .uniform_(self.blur_sigma_range[0], self.blur_sigma_range[1])
            .item()
        )
        k = self.blur_kernel_size
        # Create 1-D Gaussian kernel
        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Outer product → 2-D kernel, then reshape for conv2d
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

        # Apply per-channel (groups = C)
        c = result.shape[0]
        kernel_2d = kernel_2d.expand(c, 1, k, k).to(result.device)
        padding = k // 2
        result_4d = result.unsqueeze(0)  # (1, C, H, W)
        result_4d = F.conv2d(result_4d, kernel_2d, padding=padding, groups=c)
        result = result_4d.squeeze(0)

        # --- Additive Gaussian noise ---
        noise_std = (
            torch.empty(1)
            .uniform_(self.noise_std_range[0], self.noise_std_range[1])
            .item()
        )
        noise = torch.randn_like(result) * noise_std
        result = result + noise

        return torch.clamp(result, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"blur_kernel_size={self.blur_kernel_size}, "
            f"blur_sigma_range={self.blur_sigma_range}, "
            f"noise_std_range={self.noise_std_range}, "
            f"p={self.p})"
        )


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MarineDataset(Dataset):
    """PyTorch Dataset for loading marine images with preprocessing and augmentation.

    Supports configurable preprocessing (DCP / CLAHE) and training-time
    augmentations including custom underwater-specific transforms.

    Attributes:
        samples: List of ``(image_path, label)`` tuples.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        transform: Composed torchvision transform pipeline.
        class_to_idx: Mapping of class name to integer label.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        cfg: DictConfig,
        split: str = "train",
        apply_dcp: bool = True,
        apply_clahe: bool = True,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            samples: List of ``(image_path, label_index)`` tuples.
            cfg: Project configuration.
            split: Dataset split (``"train"``, ``"val"``, or ``"test"``).
            apply_dcp: Whether to apply Dark Channel Prior preprocessing.
            apply_clahe: Whether to apply CLAHE preprocessing.
            class_to_idx: Optional class name → index mapping.
        """
        super().__init__()
        self.samples = samples
        self.cfg = cfg
        self.split = split
        self.apply_dcp = apply_dcp
        self.apply_clahe = apply_clahe
        self.class_to_idx = class_to_idx or {}

        # Resolve effective image size
        self._image_size: int = cfg.get(
            "_effective_image_size", cfg.data.image_size_gpu
        )

        # Build the transform pipeline
        self.transform = self._build_transforms()

        logger.info(
            "MarineDataset created: split=%s, samples=%d, dcp=%s, clahe=%s, size=%d",
            split,
            len(samples),
            apply_dcp,
            apply_clahe,
            self._image_size,
        )

    def _build_transforms(self) -> transforms.Compose:
        """Build the torchvision transform pipeline.

        Training split gets augmentation; val/test get only resize + normalise.

        Returns:
            Composed transform.
        """
        aug_cfg = self.cfg.augmentation
        norm_cfg = self.cfg.preprocessing.normalize

        # Common final transforms
        common_tail = [
            transforms.ToTensor(),  # (H, W, C) uint8 → (C, H, W) float [0,1]
        ]

        if self.split == "train":
            spatial_augs = [
                transforms.Resize((self._image_size, self._image_size)),
                transforms.RandomHorizontalFlip(p=aug_cfg.horizontal_flip_p),
                transforms.RandomVerticalFlip(p=aug_cfg.vertical_flip_p),
                transforms.RandomRotation(degrees=aug_cfg.rotation_degrees),
                transforms.ColorJitter(
                    brightness=aug_cfg.brightness,
                    contrast=aug_cfg.contrast,
                    saturation=aug_cfg.saturation,
                    hue=aug_cfg.hue,
                ),
            ]

            # Custom underwater augmentations (applied on tensors)
            underwater_augs = [
                UnderwaterColorCast(
                    p=aug_cfg.underwater_color_cast_p,
                ),
                UnderwaterTurbidity(
                    p=aug_cfg.underwater_turbidity_p,
                ),
            ]

            normalize = [
                transforms.Normalize(
                    mean=list(norm_cfg.mean),
                    std=list(norm_cfg.std),
                ),
            ]

            return transforms.Compose(spatial_augs + common_tail + underwater_augs + normalize)
        else:
            # Val / test — deterministic
            return transforms.Compose(
                [
                    transforms.Resize((self._image_size, self._image_size)),
                ]
                + common_tail
                + [
                    transforms.Normalize(
                        mean=list(norm_cfg.mean),
                        std=list(norm_cfg.std),
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load, preprocess, augment, and return a single sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of ``(image_tensor, label)``.
        """
        img_path, label = self.samples[index]

        # Load image via OpenCV (BGR)
        image = cv2.imread(img_path)
        if image is None:
            logger.error("Failed to load image: %s — returning zeros", img_path)
            return torch.zeros(3, self._image_size, self._image_size), label

        # Apply underwater-specific preprocessing (DCP / CLAHE) via
        # the preprocessing module — operates on BGR uint8 arrays
        from src.preprocessing import preprocess_image

        image = preprocess_image(
            image,
            self.cfg,
            apply_dark_channel=self.apply_dcp,
            apply_clahe_flag=self.apply_clahe,
            target_size=self._image_size,
        )

        # Convert BGR → RGB for torchvision (PIL convention)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms compatibility
        pil_image = Image.fromarray(image)

        # Apply transforms (resize, augmentation, ToTensor, normalize)
        tensor = self.transform(pil_image)

        return tensor, label


# =============================================================================
# DataLoader Factory
# =============================================================================

class DataLoaderResult(dict):
    """Container holding DataLoaders and class names, accessible as tuple or dict."""

    def __init__(
        self,
        train: Optional[DataLoader],
        val: Optional[DataLoader],
        test: Optional[DataLoader],
        class_names: List[str],
    ) -> None:
        super().__init__(train=train, val=val, test=test, class_names=class_names)
        self.train = train
        self.val = val
        self.test = test
        self.class_names = class_names

    def __iter__(self):
        yield self.train
        yield self.val
        yield self.test
        yield self.class_names


def get_dataloaders(
    splits_or_cfg: Any,
    cfg: Optional[DictConfig] = None,
    task: Optional[str] = None,
    apply_dcp: bool = True,
    apply_clahe: bool = True,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Any:
    """Create train/val/test DataLoaders from split data or directly from task configuration.

    Supports:
        1. Direct invocation with config and task:
           `train_l, val_l, test_l, classes = get_dataloaders(cfg, task="coral")`
        2. Splitted dictionary invocation:
           `loaders = get_dataloaders(splits, cfg)`
    """
    class_names: List[str] = []

    # Handle case where cfg is passed as first argument and task is specified
    if isinstance(splits_or_cfg, DictConfig) or hasattr(splits_or_cfg, "paths"):
        actual_cfg = splits_or_cfg
        target_task = task or "coral"
        if target_task == "coral":
            data_dir = Path(actual_cfg.paths.coral_data)
        else:
            data_dir = Path(actual_cfg.paths.fish_data)

        # Resolve paths relative to repo root if needed
        if not data_dir.exists():
            repo_alt = Path(__file__).resolve().parent.parent / data_dir
            if repo_alt.exists():
                data_dir = repo_alt

        splits = create_stratified_splits(
            dataset_dir=data_dir,
            train_ratio=actual_cfg.data.train_ratio,
            val_ratio=actual_cfg.data.val_ratio,
            test_ratio=actual_cfg.data.test_ratio,
            seed=actual_cfg.seed,
        )
        # Extract class names
        class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        class_names = class_dirs
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    else:
        splits = splits_or_cfg
        actual_cfg = cfg
        if class_to_idx:
            class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    batch_size: int = actual_cfg.get("_effective_batch_size", actual_cfg.training.batch_size_gpu)
    num_workers: int = actual_cfg.get("num_workers", 0)  # Default 0 on CPU/Windows to avoid multiprocessing stalls
    pin_memory: bool = actual_cfg.get("pin_memory", True) and actual_cfg.get("_device", "cpu") != "cpu"
    is_cpu: bool = actual_cfg.get("cpu_mode", False)

    dataloaders: Dict[str, DataLoader] = {}

    for split_name in ("train", "val", "test"):
        samples = splits.get(split_name, [])
        if not samples:
            logger.warning("No samples for split '%s'", split_name)
            dataloaders[split_name] = None
            continue

        # CPU-mode subset: limit samples per class
        if is_cpu:
            subset_n = actual_cfg.data.get("cpu_subset_per_class", 50)
            samples = _subsample_per_class(samples, subset_n)
            logger.info(
                "CPU-mode: subsampled %s to %d samples (%d per class max)",
                split_name,
                len(samples),
                subset_n,
            )

        dataset = MarineDataset(
            samples=samples,
            cfg=actual_cfg,
            split=split_name,
            apply_dcp=apply_dcp,
            apply_clahe=apply_clahe,
            class_to_idx=class_to_idx,
        )

        sampler = None
        shuffle = (split_name == "train")
        if split_name == "train" and actual_cfg.training.get("use_weighted_sampler", False):
            sampler = _make_weighted_sampler(samples)
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train" and len(dataset) > batch_size),
        )
        dataloaders[split_name] = loader
        logger.info(
            "DataLoader '%s': %d samples, batch_size=%d, workers=%d",
            split_name,
            len(dataset),
            batch_size,
            num_workers,
        )

    return DataLoaderResult(
        train=dataloaders.get("train"),
        val=dataloaders.get("val"),
        test=dataloaders.get("test"),
        class_names=class_names,
    )


def _make_weighted_sampler(
    samples: List[Tuple[str, int]],
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes.

    Each sample is weighted inversely proportional to its class frequency,
    so that every class contributes equally in expectation.

    Args:
        samples: List of ``(path, label)`` tuples.

    Returns:
        WeightedRandomSampler instance.
    """
    labels = [label for _, label in samples]
    class_counts = Counter(labels)
    total = len(labels)

    # Weight per class: total / (num_classes * count_for_class)
    num_classes = len(class_counts)
    class_weights = {
        cls: total / (num_classes * count) for cls, count in class_counts.items()
    }

    # Per-sample weight
    sample_weights = [class_weights[label] for label in labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=total,
        replacement=True,
    )

    logger.info(
        "WeightedRandomSampler: %d samples, %d classes, class_weights=%s",
        total,
        num_classes,
        {k: f"{v:.3f}" for k, v in class_weights.items()},
    )

    return sampler


def _subsample_per_class(
    samples: List[Tuple[str, int]],
    max_per_class: int,
) -> List[Tuple[str, int]]:
    """Subsample data to at most ``max_per_class`` samples per class.

    Useful in CPU mode to keep iteration times manageable.

    Args:
        samples: Full list of ``(path, label)`` tuples.
        max_per_class: Maximum number of samples per class to keep.

    Returns:
        Subsampled list.
    """
    rng = random.Random(42)
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    result: List[Tuple[str, int]] = []
    for cls_label, cls_samples in sorted(by_class.items()):
        if len(cls_samples) > max_per_class:
            rng.shuffle(cls_samples)
            cls_samples = cls_samples[:max_per_class]
        result.extend(cls_samples)

    return result


# =============================================================================
# Dataset Summary
# =============================================================================

def print_dataset_summary(
    dataset_dir: str,
    splits: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> None:
    """Print a summary table of the dataset: class distribution, image stats.

    Uses ``tabulate`` for pretty-printing.

    Args:
        dataset_dir: Root dataset directory with class sub-folders.
        splits: Optional pre-computed splits to report counts from.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed — falling back to plain output")
        tabulate = None  # type: ignore[assignment]

    root = Path(dataset_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    # Gather per-class information
    rows = []
    total_images = 0
    all_widths: List[int] = []
    all_heights: List[int] = []

    class_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    for cls_dir in class_dirs:
        images = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
        count = len(images)
        total_images += count

        # Sample a few images for dimension stats (avoid reading all)
        sample_imgs = images[:min(10, len(images))]
        for img_path in sample_imgs:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                all_heights.append(h)
                all_widths.append(w)

        rows.append([cls_dir.name, count])

    # Print class distribution table
    headers = ["Class", "Images"]

    # Add split columns if available
    if splits:
        headers.extend(["Train", "Val", "Test"])
        # Build per-class split counts
        split_counts: Dict[str, Dict[str, int]] = {}
        for split_name, samples in splits.items():
            for _, label in samples:
                split_counts.setdefault(str(label), {})[split_name] = (
                    split_counts.setdefault(str(label), {}).get(split_name, 0) + 1
                )

        for i, row in enumerate(rows):
            cls_idx_str = str(i)
            counts = split_counts.get(cls_idx_str, {})
            row.extend([
                counts.get("train", 0),
                counts.get("val", 0),
                counts.get("test", 0),
            ])

    if tabulate is not None:
        table = tabulate(rows, headers=headers, tablefmt="grid")
    else:
        # Fallback to simple formatting
        lines = [" | ".join(str(h).ljust(15) for h in headers)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v).ljust(15) for v in row))
        table = "\n".join(lines)

    summary_lines = [
        "",
        "=" * 60,
        f"  Dataset Summary: {root}",
        "=" * 60,
        table,
        f"\n  Total images:  {total_images}",
        f"  Total classes: {len(rows)}",
    ]

    if all_widths:
        summary_lines.extend([
            f"  Min dimensions: {min(all_widths)} × {min(all_heights)}",
            f"  Max dimensions: {max(all_widths)} × {max(all_heights)}",
        ])

    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    setup_logging()
    cfg = load_config()

    logger.info("Dataset module self-test")

    # Generate placeholder data
    generator = PlaceholderDatasetGenerator(cfg)
    generator.generate_all()

    # Create splits for fish
    fish_splits = create_stratified_splits(
        dataset_dir=cfg.paths.fish_data,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )

    # Print summary
    print_dataset_summary(cfg.paths.fish_data, splits=fish_splits)

    logger.info("Dataset module self-test PASSED")
