"""
Underwater image preprocessing module.

Implements from scratch (OpenCV/NumPy only, no black-box libraries):
    1. Dark Channel Prior (DCP) based underwater color correction / dehazing
       Reference: He, K., Sun, J., & Tang, X. (2009).
       "Single Image Haze Removal Using Dark Channel Prior." CVPR.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
       Reference: Zuiderveld, K. (1994).
       "Contrast Limited Adaptive Histogram Equalization." Academic Press.

Also provides:
    - Full preprocessing pipeline (DCP → CLAHE → resize → normalize)
    - Visual before/after comparison grid
    - Quantitative image quality metrics (entropy, contrast, colorfulness)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Dark Channel Prior — Underwater Color Correction / Dehazing
# =============================================================================

def compute_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel of an image.

    The dark channel is defined as:
        J_dark(x) = min_{c ∈ {R,G,B}} ( min_{y ∈ Ω(x)} J^c(y) )

    For each pixel x, we look at all pixels y in a local patch Ω(x) of size
    `patch_size × patch_size`, and take the minimum across all color channels.
    Intuitively, in haze-free outdoor images, at least one color channel has
    very low intensity in most local patches (the "dark channel prior").

    Args:
        image: Input BGR image, float64 in [0, 1].
        patch_size: Size of the local patch Ω(x). Must be odd.

    Returns:
        Dark channel image (H × W), float64 in [0, 1].
    """
    # Step 1: Take the per-pixel minimum across color channels
    #   min_{c ∈ {R,G,B}} J^c(x)
    min_channel = np.min(image, axis=2)

    # Step 2: Apply minimum filter over the local patch
    #   min_{y ∈ Ω(x)} of the channel-wise minimum
    # We use cv2.erode with a square kernel, which computes the local minimum
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel


def estimate_atmospheric_light(
    image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.1
) -> np.ndarray:
    """Estimate the atmospheric light A from the brightest pixels in the dark channel.

    We pick the top `top_percent`% brightest pixels in the dark channel,
    then among those pixels in the original image, select the one with the
    highest intensity as the atmospheric light.

    For underwater images, this approximates the dominant water color (blue/green cast).

    Args:
        image: Input BGR image, float64 in [0, 1].
        dark_channel: Precomputed dark channel (H × W).
        top_percent: Percentage of brightest dark-channel pixels to consider.

    Returns:
        Atmospheric light vector A of shape (3,), one value per channel.
    """
    h, w = dark_channel.shape
    num_pixels = max(int(h * w * top_percent / 100.0), 1)

    # Flatten and find indices of the brightest dark-channel pixels
    flat_dark = dark_channel.ravel()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

    # Among those pixels in the original image, find the brightest one
    flat_image = image.reshape(-1, 3)
    brightest_idx = indices[np.argmax(np.sum(flat_image[indices], axis=1))]
    atmospheric_light = flat_image[brightest_idx]

    logger.debug("Estimated atmospheric light: %s", atmospheric_light)
    return atmospheric_light


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
) -> np.ndarray:
    """Estimate the transmission map t(x).

    The transmission represents how much scene light passes through the medium
    (water/haze) without being scattered. It is derived from:

        t(x) = 1 - ω · min_{c} ( min_{y ∈ Ω(x)} ( I^c(y) / A^c ) )

    where ω < 1 keeps a small amount of haze for natural appearance,
    and A is the atmospheric light.

    Args:
        image: Input BGR image, float64 in [0, 1].
        atmospheric_light: Atmospheric light vector (3,).
        patch_size: Local patch size for dark channel.
        omega: Haze removal strength (0 = no removal, 1 = full removal).

    Returns:
        Transmission map (H × W), float64 in [0, 1].
    """
    # Normalize each channel by the atmospheric light
    # This makes the "normalized dark channel" represent the haze contribution
    normalized = image / (atmospheric_light + 1e-8)

    # Compute dark channel of the normalized image
    dark_normalized = compute_dark_channel(normalized, patch_size)

    # Transmission: t(x) = 1 - ω · dark_normalized(x)
    transmission = 1.0 - omega * dark_normalized

    return transmission


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3,
) -> np.ndarray:
    """Apply guided image filtering to refine the transmission map.

    The guided filter smooths `src` while preserving edges present in `guide`.
    This is essential because the raw transmission map has block artifacts
    from the patch-based dark channel computation.

    Reference: He, K., Sun, J., & Tang, X. (2010).
    "Guided Image Filtering." ECCV.

    The filter is defined as:
        q_i = a_k · I_i + b_k,  ∀i ∈ ω_k
    where a_k and b_k are linear coefficients computed from local statistics
    of the guide image I and the filter input p in window ω_k.

    Args:
        guide: Guide image (grayscale, H × W), float64.
        src: Source image to be filtered (H × W), float64.
        radius: Filter window radius.
        eps: Regularization parameter (controls smoothness vs. edge preservation).

    Returns:
        Filtered output (H × W), float64.
    """
    # Box filter for computing local means
    # Using cv2.boxFilter is equivalent to a mean filter over (2r+1 × 2r+1)
    ksize = 2 * radius + 1

    mean_guide = cv2.boxFilter(guide, ddepth=-1, ksize=(ksize, ksize))
    mean_src = cv2.boxFilter(src, ddepth=-1, ksize=(ksize, ksize))
    mean_guide_src = cv2.boxFilter(guide * src, ddepth=-1, ksize=(ksize, ksize))
    mean_guide_sq = cv2.boxFilter(guide * guide, ddepth=-1, ksize=(ksize, ksize))

    # Covariance of (guide, src) in the local window
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    # Variance of guide in the local window
    var_guide = mean_guide_sq - mean_guide * mean_guide

    # Linear coefficients: a = cov / (var + eps),  b = mean_src - a * mean_guide
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    # Average the coefficients over all overlapping windows
    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(ksize, ksize))
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(ksize, ksize))

    # Output: q = mean_a * guide + mean_b
    output = mean_a * guide + mean_b

    return output


def recover_scene(
    image: np.ndarray,
    transmission: np.ndarray,
    atmospheric_light: np.ndarray,
    t_min: float = 0.1,
) -> np.ndarray:
    """Recover the dehazed/color-corrected scene using the imaging model.

    The haze imaging model is:
        I(x) = J(x) · t(x) + A · (1 - t(x))

    Solving for the scene radiance J(x):
        J(x) = ( I(x) - A ) / max(t(x), t_min) + A

    where t_min prevents division by near-zero transmission values.

    Args:
        image: Input hazy/underwater BGR image, float64 in [0, 1].
        transmission: Refined transmission map (H × W).
        atmospheric_light: Atmospheric light vector (3,).
        t_min: Minimum transmission threshold.

    Returns:
        Recovered scene image (H × W × 3), float64 clipped to [0, 1].
    """
    # Ensure transmission has minimum value to avoid artifacts
    t = np.maximum(transmission, t_min)

    # Recover each channel: J = (I - A) / t + A
    recovered = np.empty_like(image)
    for c in range(3):
        recovered[:, :, c] = (image[:, :, c] - atmospheric_light[c]) / t + atmospheric_light[c]

    # Clip to valid range
    recovered = np.clip(recovered, 0.0, 1.0)

    return recovered


def dark_channel_prior_correction(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t_min: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """Full Dark Channel Prior underwater color correction pipeline.

    Steps:
        1. Compute dark channel of the input image.
        2. Estimate atmospheric light from brightest dark-channel pixels.
        3. Estimate initial transmission map.
        4. Refine transmission with guided filtering (preserves edges).
        5. Recover the scene radiance (dehazed image).

    Args:
        image: Input BGR image as uint8 (0–255) or float64 (0–1).
        patch_size: Dark channel patch size (odd integer).
        omega: Haze removal strength [0, 1].
        t_min: Minimum transmission value.
        guided_radius: Guided filter radius.
        guided_eps: Guided filter regularization.

    Returns:
        Color-corrected image, uint8 (0–255).
    """
    # Convert to float64 [0, 1] if needed
    if image.dtype == np.uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)

    # Step 1: Dark channel
    dark_ch = compute_dark_channel(img_float, patch_size)

    # Step 2: Atmospheric light estimation
    atm_light = estimate_atmospheric_light(img_float, dark_ch)

    # Step 3: Initial transmission estimate
    transmission = estimate_transmission(img_float, atm_light, patch_size, omega)

    # Step 4: Refine transmission with guided filter
    gray_guide = cv2.cvtColor(img_float.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    transmission_refined = guided_filter(gray_guide, transmission, guided_radius, guided_eps)
    transmission_refined = np.clip(transmission_refined, 0.0, 1.0)

    # Step 5: Recover scene
    recovered = recover_scene(img_float, transmission_refined, atm_light, t_min)

    # Convert back to uint8
    result = (recovered * 255.0).astype(np.uint8)

    logger.debug(
        "DCP correction applied: atm_light=%s, mean_transmission=%.3f",
        atm_light,
        transmission_refined.mean(),
    )

    return result


# =============================================================================
# CLAHE — Contrast Limited Adaptive Histogram Equalization
# =============================================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB color space.

    CLAHE divides the image into tiles, computes a histogram for each tile,
    clips the histogram at `clip_limit` to prevent over-amplification of noise,
    redistributes the clipped pixels uniformly, and then performs histogram
    equalization. Bilinear interpolation between tiles eliminates boundary artifacts.

    We apply CLAHE only to the L (lightness) channel in LAB space to enhance
    contrast without distorting colors — critical for underwater images where
    color information carries species-identification cues.

    Reference: Zuiderveld, K. (1994).
    "Contrast Limited Adaptive Histogram Equalization."
    In Graphics Gems IV, Academic Press, pp. 474–485.

    Implementation notes:
        - We use OpenCV's createCLAHE for the core histogram clipping and
          equalization, but the color-space conversion and L-channel isolation
          are done manually to maintain transparency.
        - The L channel in LAB is 0–255 for 8-bit images.

    Args:
        image: Input BGR image, uint8 (0–255).
        clip_limit: Contrast limiting threshold. Higher = more contrast.
        tile_grid_size: Number of tiles (rows, cols) for adaptive processing.

    Returns:
        CLAHE-enhanced BGR image, uint8 (0–255).
    """
    # Convert BGR → LAB color space
    # L: Lightness (0–255 for uint8), A: green–red, B: blue–yellow
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object and apply to L channel only
    # This is the core algorithm:
    #   1. Divide L into tile_grid_size tiles
    #   2. For each tile, compute histogram (256 bins for uint8)
    #   3. Clip histogram: any bin exceeding clip_limit * (tile_pixels / 256)
    #      has its excess redistributed uniformly across all bins
    #   4. Compute CDF from the clipped histogram → mapping function
    #   5. Apply mapping to equalize the tile
    #   6. Bilinear interpolation at tile boundaries for smooth transitions
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge enhanced L with original A, B channels (preserving color)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # Convert LAB → BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    logger.debug(
        "CLAHE applied: clip_limit=%.1f, tile_grid_size=%s, L_mean %.1f → %.1f",
        clip_limit,
        tile_grid_size,
        l_channel.mean(),
        l_enhanced.mean(),
    )

    return result


# =============================================================================
# Full Preprocessing Pipeline
# =============================================================================

def preprocess_image(
    image: np.ndarray,
    cfg: DictConfig,
    apply_dark_channel: bool = True,
    apply_clahe_flag: bool = True,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """Full preprocessing pipeline: DCP → CLAHE → resize.

    Args:
        image: Input BGR image, uint8 (0–255).
        cfg: Project configuration.
        apply_dark_channel: Whether to apply Dark Channel Prior correction.
        apply_clahe_flag: Whether to apply CLAHE.
        target_size: Target image size (square). If None, uses config value.

    Returns:
        Preprocessed BGR image, uint8 (0–255).
    """
    result = image.copy()

    # Step 1: Dark Channel Prior color correction
    if apply_dark_channel:
        dcp_cfg = cfg.preprocessing.dark_channel
        result = dark_channel_prior_correction(
            result,
            patch_size=dcp_cfg.patch_size,
            omega=dcp_cfg.omega,
            t_min=dcp_cfg.t_min,
            guided_radius=dcp_cfg.guided_filter_radius,
            guided_eps=dcp_cfg.guided_filter_eps,
        )

    # Step 2: CLAHE enhancement
    if apply_clahe_flag:
        clahe_cfg = cfg.preprocessing.clahe
        tile_size = tuple(clahe_cfg.tile_grid_size)
        result = apply_clahe(
            result,
            clip_limit=clahe_cfg.clip_limit,
            tile_grid_size=tile_size,
        )

    # Step 3: Resize
    if target_size is None:
        target_size = cfg.get("_effective_image_size", cfg.data.image_size_gpu)
    result = cv2.resize(result, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return result


# =============================================================================
# Image Quality Metrics
# =============================================================================

def compute_entropy(image: np.ndarray) -> float:
    """Compute Shannon entropy of a grayscale image.

    Entropy = -Σ p(i) · log2(p(i)) for each intensity level i.
    Higher entropy indicates more information content (less uniform).

    Args:
        image: Grayscale image, uint8 (0–255).

    Returns:
        Entropy value in bits.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute normalized histogram (probability distribution)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()

    # Remove zero bins to avoid log(0)
    hist = hist[hist > 0]

    # Shannon entropy: H = -Σ p · log2(p)
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def compute_contrast(image: np.ndarray) -> float:
    """Compute RMS contrast of a grayscale image.

    RMS contrast = std(pixel_intensities) / mean(pixel_intensities).
    Higher values indicate more contrast.

    Args:
        image: Grayscale or BGR image, uint8 (0–255).

    Returns:
        RMS contrast value.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_f = image.astype(np.float64)
    mean_val = image_f.mean()
    if mean_val < 1e-8:
        return 0.0
    return float(image_f.std() / mean_val)


def compute_colorfulness(image: np.ndarray) -> float:
    """Compute the colorfulness metric of an image.

    Based on: Hasler, D. & Süsstrunk, S. (2003).
    "Measuring Colourfulness in Natural Images." SPIE.

    Colorfulness = sqrt(σ_rgyb² + μ_rgyb²)
    where rgyb channels capture opponent color information.

    Args:
        image: BGR image, uint8 (0–255).

    Returns:
        Colorfulness score (higher = more colorful).
    """
    B, G, R = image[:, :, 0].astype(np.float64), image[:, :, 1].astype(np.float64), image[:, :, 2].astype(np.float64)

    # Opponent color channels
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    # Mean and standard deviation of the opponent channels
    rg_mean, rg_std = rg.mean(), rg.std()
    yb_mean, yb_std = yb.mean(), yb.std()

    # Combined std and mean
    std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
    mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)

    colorfulness = std_root + 0.3 * mean_root

    return float(colorfulness)


def compute_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute a suite of image quality metrics.

    Args:
        image: BGR image, uint8 (0–255).

    Returns:
        Dictionary with entropy, contrast, and colorfulness scores.
    """
    return {
        "entropy": compute_entropy(image),
        "contrast": compute_contrast(image),
        "colorfulness": compute_colorfulness(image),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 5),
) -> None:
    """Create a side-by-side comparison grid of images with quality metrics.

    Args:
        images: List of BGR images to compare.
        titles: List of titles for each image.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        # Convert BGR → RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Add quality metrics as text below the image
        metrics = compute_image_quality_metrics(img)
        metric_text = (
            f"Entropy: {metrics['entropy']:.2f} | "
            f"Contrast: {metrics['contrast']:.2f} | "
            f"Color: {metrics['colorfulness']:.1f}"
        )
        ax.set_xlabel(metric_text, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Comparison grid saved to %s", save_path)
    plt.close()


def generate_preprocessing_demo(
    image_paths: List[str],
    cfg: DictConfig,
    output_dir: str = "outputs/figures",
) -> Dict[str, Dict[str, float]]:
    """Generate before/after comparison grids for a set of images.

    Creates a grid showing: Original → DCP Only → CLAHE Only → DCP + CLAHE
    for each input image, with quality metrics.

    Args:
        image_paths: List of paths to input images.
        cfg: Project configuration.
        output_dir: Directory to save comparison figures.

    Returns:
        Dictionary mapping image names to their quality metrics for each variant.
    """
    all_metrics = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_name = Path(img_path).stem
        original = cv2.imread(str(img_path))
        if original is None:
            logger.warning("Could not read image: %s", img_path)
            continue

        # Generate all 4 variants for the ablation comparison
        variants = {
            "Original": original,
            "DCP Only": preprocess_image(original, cfg, apply_dark_channel=True, apply_clahe_flag=False,
                                         target_size=original.shape[1]),
            "CLAHE Only": preprocess_image(original, cfg, apply_dark_channel=False, apply_clahe_flag=True,
                                           target_size=original.shape[1]),
            "DCP + CLAHE": preprocess_image(original, cfg, apply_dark_channel=True, apply_clahe_flag=True,
                                            target_size=original.shape[1]),
        }

        images = list(variants.values())
        titles = list(variants.keys())

        save_path = str(Path(output_dir) / f"preprocessing_comparison_{img_name}.png")
        create_comparison_grid(images, titles, save_path=save_path)

        # Collect metrics
        all_metrics[img_name] = {
            name: compute_image_quality_metrics(img) for name, img in variants.items()
        }

    return all_metrics


if __name__ == "__main__":
    from src.utils import setup_logging, load_config

    setup_logging()
    logger.info("Preprocessing module self-test")

    # Create a small synthetic test image (simulated underwater with blue cast)
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Simulate underwater blue-green cast
    test_img[:, :, 0] = 150  # Blue channel high
    test_img[:, :, 1] = 100  # Green channel medium
    test_img[:, :, 2] = 60   # Red channel low (absorbed underwater)
    # Add some texture
    for i in range(0, 200, 20):
        cv2.rectangle(test_img, (i, i), (i + 15, i + 15), (80, 120, 40), -1)

    # Test DCP
    corrected = dark_channel_prior_correction(test_img)
    logger.info("DCP output shape: %s, dtype: %s", corrected.shape, corrected.dtype)

    # Test CLAHE
    enhanced = apply_clahe(test_img)
    logger.info("CLAHE output shape: %s, dtype: %s", enhanced.shape, enhanced.dtype)

    # Test metrics
    metrics = compute_image_quality_metrics(test_img)
    logger.info("Quality metrics: %s", metrics)

    logger.info("Preprocessing module self-test PASSED")

