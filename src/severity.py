"""
Coral bleaching severity scoring module (heuristic).

Extends a binary healthy/bleached coral classifier to four severity levels
using Grad-CAM attention maps and pale-pixel analysis:

    0 — Healthy
    1 — Mild Bleaching
    2 — Moderate Bleaching
    3 — Severe Bleaching

**IMPORTANT**: All severity outputs are *heuristic estimates* derived from
the fraction of pale/white pixels inside the Grad-CAM–highlighted region.
They are NOT ground-truth labels and should be interpreted as indicative
guidance rather than definitive diagnoses.

Algorithm overview:
    1. Run the coral image through the classifier to get a prediction.
    2. Generate a Grad-CAM heatmap for the predicted class, highlighting
       the most salient region the model attends to.
    3. Threshold the heatmap to create a binary attention mask.
    4. Within that mask, detect "pale/white" pixels using HSV color space:
       a pixel is pale when its Value channel is high (V >= min_value)
       AND its Saturation channel is low (S <= max_saturation).
    5. Compute a bleaching score = pale_pixels_in_region / total_pixels_in_region.
    6. Map the score to one of four severity levels using configurable thresholds.

References:
    - Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from
      Deep Networks via Gradient-based Localization." ICCV.
    - Hughes, T. P. et al. (2018). "Global warming transforms coral reef
      assemblages." Nature, 556, 492–496.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.gradcam import GradCAM, overlay_heatmap
from src.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

# Severity level labels, indexed 0–3
SEVERITY_LABELS: List[str] = [
    "Healthy",
    "Mild Bleaching",
    "Moderate Bleaching",
    "Severe Bleaching",
]

# Standard disclaimer appended to every assessment
_HEURISTIC_NOTE: str = (
    "Severity is a heuristic estimate based on the fraction of pale/white "
    "pixels within the Grad-CAM attention region. It is NOT a ground-truth "
    "label and should be used for indicative guidance only."
)


# =============================================================================
# Pale-Pixel Detection
# =============================================================================

def detect_pale_pixels(
    image: np.ndarray,
    min_value: int = 180,
    max_saturation: int = 50,
) -> np.ndarray:
    """Detect pale/white pixels in a BGR image using HSV thresholds.

    A pixel is considered "pale" (indicative of coral bleaching) when it
    has high brightness and low saturation — i.e. it is close to white or
    very light grey rather than a vivid coral colour.

    In HSV space:
        - V (Value) measures brightness: V >= min_value  → bright
        - S (Saturation) measures colour purity: S <= max_saturation → washed-out

    Args:
        image: Input BGR image, uint8, shape (H, W, 3).
        min_value: Minimum V-channel value (0–255) for a pixel to be
            considered bright enough.  Default 180.
        max_saturation: Maximum S-channel value (0–255) for a pixel to be
            considered desaturated enough.  Default 50.

    Returns:
        Binary mask of pale pixels, uint8, shape (H, W).
        255 for pale pixels, 0 otherwise.
    """
    # Convert BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_channel, v_channel = cv2.split(hsv)

    # A pixel is pale when V is high AND S is low
    pale_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pale_mask[(v_channel >= min_value) & (s_channel <= max_saturation)] = 255

    logger.debug(
        "Pale pixel detection: min_value=%d, max_saturation=%d, "
        "pale_fraction=%.4f",
        min_value,
        max_saturation,
        pale_mask.sum() / (255.0 * pale_mask.size),
    )

    return pale_mask


# =============================================================================
# Bleaching Score Computation
# =============================================================================

def compute_bleaching_score(
    image: np.ndarray,
    heatmap: np.ndarray,
    attention_threshold: float = 0.5,
    min_value: int = 180,
    max_saturation: int = 50,
) -> float:
    """Compute a bleaching score from the Grad-CAM–highlighted region.

    The score is the fraction of pale/white pixels within the region of the
    image that the classifier considers most salient (i.e. where the
    Grad-CAM heatmap exceeds ``attention_threshold``).

    Mathematically:
        bleaching_score = |{p ∈ R : pale(p)}| / |R|

    where R is the attention region and pale(p) is the pale-pixel predicate.

    Edge case: if the attention region R is empty (no heatmap pixel exceeds
    the threshold), the *entire image* is used as the fallback region.

    Args:
        image: Input BGR image, uint8, shape (H, W, 3).
        heatmap: Grad-CAM heatmap, float32/float64, values in [0, 1],
            shape (H, W).  Must be the same spatial size as ``image``.
        attention_threshold: Heatmap value above which a pixel is considered
            part of the salient region.  Default 0.5.
        min_value: Forwarded to :func:`detect_pale_pixels`.
        max_saturation: Forwarded to :func:`detect_pale_pixels`.

    Returns:
        Bleaching score in [0.0, 1.0].  Higher means more bleaching.
    """
    h_img, w_img = image.shape[:2]
    h_map, w_map = heatmap.shape[:2]

    # Resize heatmap to image dimensions if they differ
    if (h_map, w_map) != (h_img, w_img):
        heatmap = cv2.resize(
            heatmap.astype(np.float32), (w_img, h_img),
            interpolation=cv2.INTER_LINEAR,
        )

    # Binary attention mask: pixels where the model "looks"
    attention_mask = (heatmap > attention_threshold).astype(np.uint8)

    total_attention_pixels = int(attention_mask.sum())

    # Fallback: if attention region is empty, use the entire image
    if total_attention_pixels == 0:
        logger.warning(
            "Empty attention region (threshold=%.2f). "
            "Falling back to whole-image analysis.",
            attention_threshold,
        )
        attention_mask = np.ones((h_img, w_img), dtype=np.uint8)
        total_attention_pixels = h_img * w_img

    # Detect pale pixels in the full image
    pale_mask = detect_pale_pixels(image, min_value, max_saturation)

    # Count pale pixels that overlap with the attention region
    # pale_mask is 0/255; convert to 0/1 for counting
    pale_in_region = int(
        np.count_nonzero((pale_mask > 0) & (attention_mask > 0))
    )

    # Bleaching score = fraction of pale pixels in the salient region
    bleaching_score = pale_in_region / total_attention_pixels

    logger.debug(
        "Bleaching score: %.4f  (pale_in_region=%d, total_region=%d)",
        bleaching_score,
        pale_in_region,
        total_attention_pixels,
    )

    return float(bleaching_score)


# =============================================================================
# Severity Classification
# =============================================================================

def classify_severity(
    bleaching_score: float,
    threshold_healthy: float = 0.15,
    threshold_mild: float = 0.35,
    threshold_moderate: float = 0.60,
) -> Tuple[str, int]:
    """Map a bleaching score to a severity level.

    Severity ladder (configurable via thresholds):
        score < threshold_healthy   → "Healthy"           (index 0)
        score < threshold_mild      → "Mild Bleaching"    (index 1)
        score < threshold_moderate  → "Moderate Bleaching" (index 2)
        score >= threshold_moderate → "Severe Bleaching"  (index 3)

    Args:
        bleaching_score: Float in [0, 1] from :func:`compute_bleaching_score`.
        threshold_healthy: Upper bound for "Healthy".
        threshold_mild: Upper bound for "Mild Bleaching".
        threshold_moderate: Upper bound for "Moderate Bleaching".

    Returns:
        Tuple of (severity_label, severity_index).
    """
    if bleaching_score < threshold_healthy:
        severity_index = 0
    elif bleaching_score < threshold_mild:
        severity_index = 1
    elif bleaching_score < threshold_moderate:
        severity_index = 2
    else:
        severity_index = 3

    severity_label = SEVERITY_LABELS[severity_index]

    logger.debug(
        "Severity classification: score=%.4f → %s (index=%d)",
        bleaching_score,
        severity_label,
        severity_index,
    )

    return severity_label, severity_index


# =============================================================================
# Severity Assessor (Full Pipeline)
# =============================================================================

class SeverityAssessor:
    """End-to-end coral bleaching severity assessment pipeline.

    Orchestrates: preprocessing → model inference → Grad-CAM → pale-pixel
    analysis → severity classification.

    All severity outputs carry a disclaimer noting they are heuristic
    estimates, not ground-truth labels.

    Attributes:
        model: Trained coral classifier (PyTorch ``nn.Module``).
        cfg: Project configuration (``OmegaConf.DictConfig``).
        device: Torch device for inference.
        gradcam: :class:`GradCAM` wrapper around the model.
        class_names: Ordered list of class names from the model.
    """

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        """Initialise the severity assessor.

        Args:
            model: Trained PyTorch classifier.  Must expose a backbone with
                named layers (e.g. ResNet ``layer4``) and a final ``fc`` head.
            cfg: Full project configuration.  Expected keys:
                - ``cfg.severity.thresholds.{healthy, mild, moderate}``
                - ``cfg.severity.pale_pixel.{min_value, max_saturation}``
                - ``cfg.gradcam.target_layer``
                - ``cfg.gradcam.alpha``
        """
        self.cfg = cfg
        self.model = model
        self.device = torch.device(cfg.get("_device", "cpu"))
        self.model.to(self.device)
        self.model.eval()

        # Severity thresholds from config
        sev_cfg = cfg.severity.thresholds
        self._threshold_healthy: float = float(sev_cfg.healthy)
        self._threshold_mild: float = float(sev_cfg.mild)
        self._threshold_moderate: float = float(sev_cfg.moderate)

        # Pale-pixel HSV thresholds from config
        pale_cfg = cfg.severity.pale_pixel
        self._pale_min_value: int = int(pale_cfg.min_value)
        self._pale_max_saturation: int = int(pale_cfg.max_saturation)

        # Grad-CAM setup
        target_layer_name: str = cfg.gradcam.target_layer
        self.gradcam = GradCAM(model, target_layer_name)
        self._gradcam_alpha: float = float(cfg.gradcam.alpha)

        # Class names (healthy / bleached for the coral binary model)
        self.class_names: List[str] = getattr(
            model, "class_names", ["healthy", "bleached"]
        )

        logger.info(
            "SeverityAssessor initialised: thresholds=(%.2f, %.2f, %.2f), "
            "pale_pixel=(V>=%d, S<=%d), target_layer=%s",
            self._threshold_healthy,
            self._threshold_mild,
            self._threshold_moderate,
            self._pale_min_value,
            self._pale_max_saturation,
            target_layer_name,
        )

    # --------------------------------------------------------------------- #
    # Single-image assessment
    # --------------------------------------------------------------------- #

    def assess(self, image: np.ndarray) -> Dict[str, Any]:
        """Run the full severity assessment pipeline on a single image.

        Steps:
            1. Preprocess the BGR image (DCP + CLAHE + resize + normalise).
            2. Forward pass through the classifier to obtain prediction and
               confidence.
            3. Generate a Grad-CAM heatmap for the predicted class.
            4. Compute the bleaching score from pale pixels in the
               attention region.
            5. Classify the bleaching score into a severity level.

        Args:
            image: Raw BGR image, uint8, shape (H, W, 3).

        Returns:
            Dictionary with keys:
                - ``prediction`` (str): Predicted class label.
                - ``confidence`` (float): Softmax probability of the
                  predicted class.
                - ``bleaching_score`` (float): Fraction of pale pixels in
                  the attention region, range [0, 1].
                - ``severity`` (str): Severity level label.
                - ``severity_index`` (int): Severity level index (0–3).
                - ``heatmap`` (np.ndarray): Raw Grad-CAM heatmap, float,
                  shape (H_cam, W_cam).
                - ``overlay`` (np.ndarray): Heatmap overlaid on the
                  original image, uint8, shape (H, W, 3).
                - ``pale_mask`` (np.ndarray): Binary mask of detected pale
                  pixels, uint8, shape (H, W).
                - ``note`` (str): Heuristic disclaimer.
        """
        # --- 1. Preprocess ------------------------------------------------
        preprocessed = preprocess_image(
            image, self.cfg, apply_dark_channel=True, apply_clahe_flag=True,
        )

        # Build tensor: (H, W, 3) BGR uint8 → (1, 3, H, W) RGB float
        img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        # Apply ImageNet normalisation
        norm_mean = torch.tensor(
            self.cfg.preprocessing.normalize.mean, dtype=torch.float32,
        ).view(3, 1, 1)
        norm_std = torch.tensor(
            self.cfg.preprocessing.normalize.std, dtype=torch.float32,
        ).view(3, 1, 1)
        img_tensor = (img_tensor - norm_mean) / norm_std
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # --- 2. Model inference -------------------------------------------
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        pred_idx_int: int = int(pred_idx.item())
        confidence_float: float = float(confidence.item())
        prediction_label: str = (
            self.class_names[pred_idx_int]
            if pred_idx_int < len(self.class_names)
            else f"class_{pred_idx_int}"
        )

        # --- 3. Grad-CAM heatmap -----------------------------------------
        heatmap = self.gradcam.generate(img_tensor, target_class=pred_idx_int)
        overlay = overlay_heatmap(
            image, heatmap, alpha=self._gradcam_alpha,
        )

        # --- 4. Bleaching score -------------------------------------------
        bleaching_score = compute_bleaching_score(
            image,
            heatmap,
            attention_threshold=0.5,
            min_value=self._pale_min_value,
            max_saturation=self._pale_max_saturation,
        )

        # Pale mask for visualisation (on the original-resolution image)
        pale_mask = detect_pale_pixels(
            image,
            min_value=self._pale_min_value,
            max_saturation=self._pale_max_saturation,
        )

        # --- 5. Severity classification -----------------------------------
        severity_label, severity_index = classify_severity(
            bleaching_score,
            threshold_healthy=self._threshold_healthy,
            threshold_mild=self._threshold_mild,
            threshold_moderate=self._threshold_moderate,
        )

        result: Dict[str, Any] = {
            "prediction": prediction_label,
            "confidence": confidence_float,
            "bleaching_score": bleaching_score,
            "severity": severity_label,
            "severity_index": severity_index,
            "heatmap": heatmap,
            "overlay": overlay,
            "pale_mask": pale_mask,
            "note": _HEURISTIC_NOTE,
        }

        logger.info(
            "Assessment: prediction=%s (%.2f%%), bleaching_score=%.4f, "
            "severity=%s",
            prediction_label,
            confidence_float * 100,
            bleaching_score,
            severity_label,
        )

        return result

    # --------------------------------------------------------------------- #
    # Batch assessment
    # --------------------------------------------------------------------- #

    def assess_batch(
        self, image_dir: str, extensions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Assess all images in a directory and return a summary DataFrame.

        Args:
            image_dir: Path to directory containing coral images.
            extensions: Allowed file extensions (case-insensitive).
                Defaults to ``[".jpg", ".jpeg", ".png", ".bmp", ".tif"]``.

        Returns:
            ``pandas.DataFrame`` with one row per image and columns:
            ``filename``, ``prediction``, ``confidence``,
            ``bleaching_score``, ``severity``, ``severity_index``, ``note``.
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]

        image_dir_path = Path(image_dir)
        if not image_dir_path.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir_path}"
            )

        image_files = sorted(
            p for p in image_dir_path.iterdir()
            if p.suffix.lower() in extensions
        )

        if not image_files:
            logger.warning("No images found in %s", image_dir_path)
            return pd.DataFrame()

        logger.info(
            "Batch assessment: %d images in %s", len(image_files), image_dir,
        )

        records: List[Dict[str, Any]] = []
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            result = self.assess(image)
            records.append({
                "filename": img_path.name,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "bleaching_score": result["bleaching_score"],
                "severity": result["severity"],
                "severity_index": result["severity_index"],
                "note": result["note"],
            })

        df = pd.DataFrame(records)
        logger.info("Batch assessment complete: %d images processed", len(df))
        return df


# =============================================================================
# Reef Health Index
# =============================================================================

def compute_reef_health_index(
    assessments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute a Reef Health Index (RHI) from a collection of assessments.

    The RHI is a weighted average that maps each severity level to a health
    contribution:

        RHI = (N_healthy × 1.0 + N_mild × 0.66 + N_moderate × 0.33
               + N_severe × 0.0) / N_total

    Scale: 0–1, where 1.0 = perfectly healthy reef, 0.0 = fully bleached.

    Note: the RHI is a heuristic summary statistic.  It is NOT a validated
    ecological index.

    Args:
        assessments: List of assessment dictionaries, each containing at
            least a ``severity_index`` key (0–3).

    Returns:
        Dictionary with:
            - ``rhi`` (float): Reef Health Index in [0, 1].
            - ``severity_distribution`` (dict): Counts for each severity
              level, e.g. ``{"Healthy": 5, "Mild Bleaching": 3, ...}``.
            - ``total_assessed`` (int): Total number of images assessed.
            - ``note`` (str): Heuristic disclaimer.
    """
    if not assessments:
        logger.warning("No assessments provided for RHI computation.")
        return {
            "rhi": float("nan"),
            "severity_distribution": {label: 0 for label in SEVERITY_LABELS},
            "total_assessed": 0,
            "note": _HEURISTIC_NOTE,
        }

    # Health contribution weights for severity indices 0–3
    # Healthy = full contribution (1.0), Severe = zero contribution (0.0)
    health_weights = [1.0, 0.66, 0.33, 0.0]

    severity_counts = {label: 0 for label in SEVERITY_LABELS}
    weighted_sum = 0.0

    for assessment in assessments:
        idx = int(assessment["severity_index"])
        severity_counts[SEVERITY_LABELS[idx]] += 1
        weighted_sum += health_weights[idx]

    n_total = len(assessments)
    rhi = weighted_sum / n_total

    logger.info(
        "Reef Health Index: %.4f  (n=%d, distribution=%s)",
        rhi,
        n_total,
        severity_counts,
    )

    return {
        "rhi": float(rhi),
        "severity_distribution": severity_counts,
        "total_assessed": n_total,
        "note": _HEURISTIC_NOTE,
    }


# =============================================================================
# Visualisation
# =============================================================================

# Colour map for severity level annotations (BGR)
_SEVERITY_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),      # Green  — Healthy
    1: (0, 200, 255),     # Orange — Mild
    2: (0, 100, 255),     # Dark orange — Moderate
    3: (0, 0, 255),       # Red — Severe
}


def visualize_severity(
    image: np.ndarray,
    heatmap: np.ndarray,
    pale_mask: np.ndarray,
    severity: str,
    bleaching_score: float,
    output_path: str,
    figsize: Tuple[int, int] = (20, 5),
) -> None:
    """Create a four-panel severity visualisation figure.

    Panels (left → right):
        1. **Original** — the raw input image.
        2. **Grad-CAM** — heatmap overlaid on the image.
        3. **Pale Pixel Mask** — binary mask of detected pale pixels.
        4. **Annotated** — original image with severity label and score.

    Args:
        image: Original BGR image, uint8, shape (H, W, 3).
        heatmap: Grad-CAM heatmap, float, shape (H_cam, W_cam), values
            in [0, 1].
        pale_mask: Binary pale-pixel mask, uint8, shape (H, W), values
            0 or 255.
        severity: Severity level label string.
        bleaching_score: Bleaching score in [0, 1].
        output_path: File path to save the figure (e.g. ``"out/sev.png"``).
        figsize: Matplotlib figure size (width, height).
    """
    h, w = image.shape[:2]

    # Resize heatmap to image dimensions if needed
    if heatmap.shape[:2] != (h, w):
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        heatmap_resized = heatmap

    # Convert BGR → RGB for matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Panel 2: Grad-CAM overlay ---
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    gradcam_overlay = cv2.addWeighted(image, 0.5, heatmap_coloured, 0.5, 0)
    gradcam_overlay_rgb = cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)

    # --- Panel 4: Annotated image ---
    severity_index = SEVERITY_LABELS.index(severity)
    colour_bgr = _SEVERITY_COLORS[severity_index]
    # Convert BGR to RGB for matplotlib
    colour_rgb = (colour_bgr[2] / 255, colour_bgr[1] / 255, colour_bgr[0] / 255)

    annotated = image.copy()
    # Draw a coloured border to indicate severity
    border_thickness = max(3, min(h, w) // 50)
    cv2.rectangle(
        annotated, (0, 0), (w - 1, h - 1), colour_bgr, border_thickness,
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # --- Build figure ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    panel_data = [
        (image_rgb, "Original"),
        (gradcam_overlay_rgb, "Grad-CAM Heatmap"),
        (pale_mask, "Pale Pixel Mask"),
        (annotated_rgb, f"{severity}\n(score: {bleaching_score:.3f})"),
    ]

    for ax, (data, title) in zip(axes, panel_data):
        if data.ndim == 2:
            ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=12, fontweight="bold", color=colour_rgb if title.startswith(severity) else "black")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Coral Bleaching Severity Assessment (heuristic)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Severity visualisation saved to %s", output_path)


# =============================================================================
# Main — Self-Test / Demo
# =============================================================================

if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    setup_logging()
    logger.info("Severity module self-test (standalone functions only)")

    # --- Test detect_pale_pixels ------------------------------------------
    # Create a synthetic image: left half = vivid coral, right half = bleached
    test_img = np.zeros((200, 400, 3), dtype=np.uint8)
    # Left: saturated coral colour (HSV ≈ H=10, S=200, V=180)
    test_img[:, :200] = [100, 80, 200]  # BGR
    # Right: pale/white (HSV ≈ H=0, S=10, V=240)
    test_img[:, 200:] = [240, 235, 245]  # BGR ~ white/pale

    pale_mask = detect_pale_pixels(test_img, min_value=180, max_saturation=50)
    pale_frac_left = pale_mask[:, :200].sum() / (255.0 * 200 * 200)
    pale_frac_right = pale_mask[:, 200:].sum() / (255.0 * 200 * 200)
    logger.info(
        "Pale pixel fractions — left (coral): %.4f, right (bleached): %.4f",
        pale_frac_left,
        pale_frac_right,
    )
    assert pale_frac_left < 0.1, "Coral half should have few pale pixels"
    assert pale_frac_right > 0.5, "Bleached half should have many pale pixels"

    # --- Test compute_bleaching_score with synthetic heatmap ---------------
    # Heatmap that focuses entirely on the right (bleached) half
    heatmap_right = np.zeros((200, 400), dtype=np.float32)
    heatmap_right[:, 200:] = 0.9
    score_right = compute_bleaching_score(
        test_img, heatmap_right, attention_threshold=0.5,
    )
    logger.info("Bleaching score (attention on bleached half): %.4f", score_right)
    assert score_right > 0.5, "Score should be high when focused on bleached area"

    # Heatmap that focuses entirely on the left (healthy) half
    heatmap_left = np.zeros((200, 400), dtype=np.float32)
    heatmap_left[:, :200] = 0.9
    score_left = compute_bleaching_score(
        test_img, heatmap_left, attention_threshold=0.5,
    )
    logger.info("Bleaching score (attention on healthy half): %.4f", score_left)
    assert score_left < 0.15, "Score should be low when focused on healthy area"

    # --- Test classify_severity -------------------------------------------
    for test_score, expected_idx in [
        (0.05, 0), (0.25, 1), (0.45, 2), (0.80, 3),
    ]:
        label, idx = classify_severity(test_score)
        assert idx == expected_idx, (
            f"Score {test_score} should map to index {expected_idx}, "
            f"got {idx} ({label})"
        )
        logger.info("Score %.2f → %s (index %d) ✓", test_score, label, idx)

    # --- Test compute_reef_health_index -----------------------------------
    fake_assessments = [
        {"severity_index": 0},
        {"severity_index": 0},
        {"severity_index": 1},
        {"severity_index": 2},
        {"severity_index": 3},
    ]
    rhi_result = compute_reef_health_index(fake_assessments)
    # Expected: (1.0 + 1.0 + 0.66 + 0.33 + 0.0) / 5 = 0.598
    logger.info("RHI: %.4f (expected ≈ 0.598)", rhi_result["rhi"])
    assert abs(rhi_result["rhi"] - 0.598) < 0.01, (
        f"RHI should be ≈ 0.598, got {rhi_result['rhi']}"
    )

    logger.info("Severity module self-test PASSED ✓")
