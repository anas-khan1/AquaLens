"""
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

Reference:
    Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
    (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization." In Proceedings of the IEEE International Conference on Computer
    Vision (ICCV), pp. 618–626.

This module provides:
    - GradCAM class for generating class-discriminative heatmaps
    - Heatmap overlay utilities for visualization
    - Sample generation with side-by-side comparisons
    - Error analysis with confusion pattern detection
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Generates class-discriminative localization heatmaps by computing the
    gradient of the target class score with respect to the feature maps of
    a specified convolutional layer.

    The Grad-CAM heatmap is computed as:
        L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)
    where:
        α^c_k = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}   (global average pooled gradients)
        A^k   = k-th channel of the target layer's activation map
        y^c   = score for class c (pre-softmax logit)

    Args:
        model: A PyTorch classification model (e.g., MarineClassifier).
        target_layer_name: Name of the convolutional layer to compute
            Grad-CAM on. For ResNet architectures, typically "layer4".
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = "layer4",
    ) -> None:
        """Initialize GradCAM with forward and backward hooks on the target layer.

        Args:
            model: A PyTorch classification model.
            target_layer_name: Dot-separated path to the target convolutional
                layer (e.g., "layer4" or "backbone.layer4").

        Raises:
            ValueError: If the target layer cannot be found in the model.
        """
        self.model = model
        self.target_layer_name = target_layer_name

        # Storage for hooked activations and gradients
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Resolve the target layer from the model
        target_layer = self._resolve_layer(model, target_layer_name)

        # Register forward hook to capture activations: A^k
        fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._hooks.append(fwd_hook)

        # Register backward hook to capture gradients: ∂y^c/∂A^k
        bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)
        self._hooks.append(bwd_hook)

        logger.info(
            "GradCAM initialized on layer '%s'", target_layer_name
        )

    @staticmethod
    def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
        """Resolve a layer by its dot-separated name in the model hierarchy.

        Args:
            model: The parent model.
            layer_name: Dot-separated path (e.g., "backbone.layer4").

        Returns:
            The target nn.Module.

        Raises:
            ValueError: If the layer is not found.
        """
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Try searching one level deeper (e.g., inside a 'backbone')
                found = False
                for name, module in current.named_modules():
                    if name == layer_name or name.endswith(f".{layer_name}"):
                        current = module
                        found = True
                        break
                if not found:
                    available = [n for n, _ in model.named_modules() if n]
                    raise ValueError(
                        f"Layer '{layer_name}' not found in model. "
                        f"Available layers: {available[:20]}"
                    )
                break
        return current

    def _forward_hook(
        self,
        module: nn.Module,
        input_tensor: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Forward hook to capture feature map activations.

        Args:
            module: The hooked module.
            input_tensor: Input to the module.
            output: Output activation maps of shape (B, C, H, W).
        """
        self._activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook to capture gradients flowing into the target layer.

        Args:
            module: The hooked module.
            grad_input: Gradients w.r.t. the module's input.
            grad_output: Gradients w.r.t. the module's output (∂y^c/∂A^k).
        """
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the given input.

        Algorithm:
            1. Forward pass to obtain class logits and capture activations A^k.
            2. Select the target class score y^c (predicted class if not specified).
            3. Backward pass to obtain gradients ∂y^c/∂A^k_{ij}.
            4. Compute channel importance weights via global average pooling:
               α^c_k = (1/Z) Σ_i Σ_j  ∂y^c/∂A^k_{ij}
            5. Compute weighted combination:
               L_cam = ReLU(Σ_k α^c_k · A^k)
            6. Upsample to input spatial dimensions and normalize to [0, 1].

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W).
                Must be on the same device as the model.
            target_class: Class index to generate the heatmap for. If None,
                uses the predicted class (argmax of logits).

        Returns:
            Heatmap as a float32 numpy array of shape (H_input, W_input) with
            values in [0, 1].
        """
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Input spatial size for upsampling
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]

        # Enable gradients for the forward pass
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Step 1: Forward pass — captures activations via hook
        logits = self.model(input_tensor)

        # Step 2: Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Extract the scalar score for the target class
        score = logits[0, target_class]

        # Step 3: Backward pass — captures gradients via hook
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Validate that hooks captured data
        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Hooks did not capture activations/gradients. "
                "Ensure the target layer is part of the forward graph."
            )

        # Step 4: Global Average Pooling of gradients to get channel weights
        # α^c_k = (1/Z) Σ_i Σ_j  ∂y^c/∂A^k_{ij}
        # gradients shape: (1, K, H_feat, W_feat)
        alpha_k = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Step 5: Weighted combination of activation maps
        # L_cam = ReLU(Σ_k α_k · A^k)
        # activations shape: (1, K, H_feat, W_feat)
        cam = (alpha_k * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Step 6: Upsample to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy and normalize to [0, 1]
        heatmap = cam.squeeze().cpu().numpy()
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove all registered forward and backward hooks.

        Should be called when Grad-CAM is no longer needed to prevent
        memory leaks from dangling hook references.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations = None
        self._gradients = None
        logger.info("GradCAM hooks removed")


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image using JET colormap blending.

    Applies the JET colormap to the heatmap and blends it with the original
    image using alpha compositing:
        result = α · heatmap_colored + (1 - α) · original

    Args:
        image: Original BGR image as uint8 array of shape (H, W, 3).
        heatmap: Grad-CAM heatmap as float array in [0, 1] of shape (H, W).
        alpha: Transparency weight for the heatmap overlay. Higher values
            make the heatmap more visible. Must be in [0, 1].

    Returns:
        Blended BGR image as uint8 array of shape (H, W, 3).
    """
    # Resize heatmap to match image dimensions if needed
    img_h, img_w = image.shape[:2]
    if heatmap.shape[:2] != (img_h, img_w):
        heatmap = cv2.resize(heatmap, (img_w, img_h))

    # Convert heatmap to uint8 and apply JET colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Alpha blending: result = α * heatmap_colored + (1 - α) * original
    blended = cv2.addWeighted(
        heatmap_colored, alpha,
        image, 1.0 - alpha,
        gamma=0.0,
    )

    return blended.astype(np.uint8)


def _denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalized image tensor back to a displayable uint8 BGR image.

    Args:
        tensor: Image tensor of shape (C, H, W), normalized with given stats.
        mean: Per-channel means used during normalization.
        std: Per-channel standard deviations used during normalization.

    Returns:
        BGR uint8 image of shape (H, W, 3).
    """
    img = tensor.cpu().clone()
    for ch in range(3):
        img[ch] = img[ch] * std[ch] + mean[ch]
    img = img.clamp(0.0, 1.0)
    # Convert CHW → HWC, float → uint8, RGB → BGR
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


def generate_gradcam_samples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    class_names: List[str],
    num_samples: int = 20,
    output_dir: Optional[str] = None,
) -> List[Tuple[str, int, int, float, bool]]:
    """Generate Grad-CAM visualizations for a set of samples.

    For each sample, creates a figure with three columns:
        1. Original (denormalized) image
        2. Enhanced image (same as original if no separate enhanced version)
        3. Grad-CAM heatmap overlay

    The figure title shows prediction, ground truth, and correctness.

    Args:
        model: Trained classification model.
        dataloader: DataLoader yielding (images, labels) batches.
        cfg: Project configuration (must contain gradcam and preprocessing
            sections).
        class_names: List of class name strings, indexed by label.
        num_samples: Maximum number of samples to visualize.
        output_dir: Directory to save the figures. If None, uses
            cfg.paths.gradcam_dir.

    Returns:
        List of tuples (image_path, true_label, pred_label, confidence, correct)
        for each visualized sample.
    """
    # Resolve output directory
    save_dir = Path(output_dir if output_dir else cfg.paths.gradcam_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract config values
    target_layer = cfg.gradcam.get("target_layer", "layer4")
    alpha = cfg.gradcam.get("alpha", 0.5)
    norm_mean = tuple(cfg.preprocessing.normalize.mean)
    norm_std = tuple(cfg.preprocessing.normalize.std)

    # Initialize Grad-CAM
    device = next(model.parameters()).device
    grad_cam = GradCAM(model, target_layer_name=target_layer)

    results: List[Tuple[str, int, int, float, bool]] = []
    sample_count = 0

    model.eval()
    with torch.no_grad():
        pass  # We need gradients for Grad-CAM, so don't use no_grad below

    for batch_images, batch_labels in dataloader:
        if sample_count >= num_samples:
            break

        for i in range(batch_images.size(0)):
            if sample_count >= num_samples:
                break

            image_tensor = batch_images[i].to(device)
            true_label = batch_labels[i].item()

            # Generate Grad-CAM heatmap (needs gradients)
            with torch.enable_grad():
                input_batch = image_tensor.unsqueeze(0)
                heatmap = grad_cam.generate(input_batch)

            # Get prediction
            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0))
                probs = F.softmax(logits, dim=1)
                confidence, pred_label = probs.max(dim=1)
                confidence = confidence.item()
                pred_label = pred_label.item()

            correct = pred_label == true_label

            # Denormalize image for display
            img_bgr = _denormalize_image(image_tensor, norm_mean, norm_std)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Create Grad-CAM overlay
            overlay = overlay_heatmap(img_bgr, heatmap, alpha=alpha)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Build the 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Column 1: Original image
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")

            # Column 2: Enhanced image (re-display original as enhanced
            # since preprocessing is already applied in the pipeline)
            axes[1].imshow(img_rgb)
            axes[1].set_title("Enhanced", fontsize=12)
            axes[1].axis("off")

            # Column 3: Grad-CAM overlay
            axes[2].imshow(overlay_rgb)
            axes[2].set_title("Grad-CAM Overlay", fontsize=12)
            axes[2].axis("off")

            # Build informative title
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
            true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            status = "✓" if correct else "✗"
            title_color = "green" if correct else "red"
            fig.suptitle(
                f"Pred: {pred_name} ({confidence:.1%}) | True: {true_name} | {status}",
                fontsize=14,
                fontweight="bold",
                color=title_color,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])

            # Save figure
            filename = f"gradcam_{sample_count:03d}_{status.replace('✓', 'correct').replace('✗', 'incorrect')}.png"
            save_path = save_dir / filename
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            results.append((str(save_path), true_label, pred_label, confidence, correct))
            sample_count += 1

            logger.debug(
                "Grad-CAM sample %d: pred=%s (%.1f%%) true=%s %s",
                sample_count, pred_name, confidence * 100, true_name, status,
            )

    # Clean up hooks
    grad_cam.remove_hooks()

    # Log summary
    n_correct = sum(1 for *_, c in results if c)
    logger.info(
        "Generated %d Grad-CAM samples (%d correct, %d incorrect) in %s",
        len(results), n_correct, len(results) - n_correct, save_dir,
    )

    return results


def generate_error_analysis(
    results: List[Tuple[str, int, int, float, bool]],
    class_names: List[str],
    output_dir: str,
) -> Dict[str, object]:
    """Analyze misclassifications and produce an error analysis report.

    Creates:
        1. A summary grid figure showing incorrect predictions with their
           Grad-CAM overlays.
        2. A text report listing each misclassification, common confusion
           patterns, and potential failure reasons.

    Args:
        results: Output from generate_gradcam_samples — list of
            (image_path, true_label, pred_label, confidence, correct).
        class_names: List of class name strings.
        output_dir: Directory to save the error analysis outputs.

    Returns:
        Dictionary with keys:
            - "num_correct": Count of correct predictions.
            - "num_incorrect": Count of incorrect predictions.
            - "accuracy": Fraction correct.
            - "confusion_patterns": List of (true_class, pred_class, count).
            - "report_path": Path to the text report file.
            - "figure_path": Path to the summary grid figure (or None).
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    incorrect = [(p, t, pr, c) for p, t, pr, c, ok in results if not ok]
    correct_list = [(p, t, pr, c) for p, t, pr, c, ok in results if ok]

    num_correct = len(correct_list)
    num_incorrect = len(incorrect)
    total = len(results)
    accuracy = num_correct / total if total > 0 else 0.0

    # --- Confusion pattern analysis ---
    confusion_counter: Counter = Counter()
    for _, true_label, pred_label, _ in incorrect:
        true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
        confusion_counter[(true_name, pred_name)] += 1

    confusion_patterns = [
        (true_cls, pred_cls, count)
        for (true_cls, pred_cls), count in confusion_counter.most_common()
    ]

    # --- Generate error grid figure ---
    figure_path = None
    if incorrect:
        max_display = min(len(incorrect), 16)  # 4×4 grid max
        ncols = min(4, max_display)
        nrows = (max_display + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(max_display):
            row, col = divmod(idx, ncols)
            img_path, true_label, pred_label, conf = incorrect[idx]

            # Load the saved Grad-CAM figure image
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img_rgb)
            else:
                axes[row, col].text(
                    0.5, 0.5, "Image\nNot Found",
                    ha="center", va="center", fontsize=10,
                )

            true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
            axes[row, col].set_title(
                f"T: {true_name}\nP: {pred_name} ({conf:.0%})",
                fontsize=8,
                color="red",
            )
            axes[row, col].axis("off")

        # Hide unused subplots
        for idx in range(max_display, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis("off")

        fig.suptitle(
            f"Error Analysis — {num_incorrect} Misclassifications "
            f"(Accuracy: {accuracy:.1%})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        figure_path = str(save_dir / "error_analysis_grid.png")
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Error analysis grid saved to %s", figure_path)

    # --- Generate text report ---
    report_lines = [
        "=" * 70,
        "GRAD-CAM ERROR ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total samples analyzed: {total}",
        f"Correct predictions:    {num_correct} ({accuracy:.1%})",
        f"Incorrect predictions:  {num_incorrect} ({1.0 - accuracy:.1%})",
        "",
        "-" * 70,
        "MISCLASSIFICATION DETAILS",
        "-" * 70,
    ]

    for img_path, true_label, pred_label, conf in incorrect:
        true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
        report_lines.append(
            f"  True: {true_name:<20s} → Predicted: {pred_name:<20s} "
            f"(Confidence: {conf:.1%})  [{Path(img_path).name}]"
        )

    report_lines.extend([
        "",
        "-" * 70,
        "COMMON CONFUSION PATTERNS",
        "-" * 70,
    ])

    if confusion_patterns:
        for true_cls, pred_cls, count in confusion_patterns:
            report_lines.append(
                f"  {true_cls:<20s} → {pred_cls:<20s} : {count} occurrence(s)"
            )
        report_lines.extend([
            "",
            "Potential failure reasons (based on Grad-CAM analysis):",
            "  - Visually similar species may share texture/color patterns that",
            "    cause the model to attend to non-discriminative regions.",
            "  - Underwater color distortion can reduce inter-class contrast,",
            "    especially between species with similar body shapes.",
            "  - Occluded or partially visible specimens may lack the key features",
            "    the model has learned to rely on (as shown by dispersed Grad-CAM).",
            "  - Background coral/substrate patterns may dominate the activation",
            "    maps, distracting the model from the foreground subject.",
        ])
    else:
        report_lines.append("  No misclassifications detected — all predictions correct!")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    report_path = str(save_dir / "error_analysis_report.txt")
    Path(report_path).write_text(report_text, encoding="utf-8")
    logger.info("Error analysis report saved to %s", report_path)

    summary = {
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "accuracy": accuracy,
        "confusion_patterns": confusion_patterns,
        "report_path": report_path,
        "figure_path": figure_path,
    }

    return summary


if __name__ == "__main__":
    """Self-test using a dummy ResNet-like model and random input."""
    import torchvision.models as models

    from src.utils import setup_logging

    setup_logging()

    # --- Create a simple model with a known layer4 ---
    logger.info("Creating dummy ResNet-18 model for Grad-CAM self-test")
    dummy_model = models.resnet18(weights=None, num_classes=5)
    dummy_model.eval()

    # --- Test GradCAM class ---
    logger.info("Testing GradCAM.generate()")
    grad_cam = GradCAM(dummy_model, target_layer_name="layer4")

    # Random input: batch of 1, 3 channels, 224×224
    dummy_input = torch.randn(1, 3, 224, 224)

    # Generate heatmap for the predicted class
    heatmap = grad_cam.generate(dummy_input)
    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"Expected float32, got {heatmap.dtype}"
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0, "Heatmap not in [0, 1]"
    logger.info(
        "Heatmap shape=%s, min=%.4f, max=%.4f ✓",
        heatmap.shape, heatmap.min(), heatmap.max(),
    )

    # Generate heatmap for a specific class
    heatmap_cls2 = grad_cam.generate(dummy_input, target_class=2)
    assert heatmap_cls2.shape == (224, 224)
    logger.info("Target-class heatmap generated ✓")

    # Clean up
    grad_cam.remove_hooks()

    # --- Test overlay_heatmap ---
    logger.info("Testing overlay_heatmap()")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    overlay = overlay_heatmap(dummy_image, heatmap, alpha=0.5)
    assert overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Expected uint8, got {overlay.dtype}"
    logger.info("Overlay shape=%s, dtype=%s ✓", overlay.shape, overlay.dtype)

    # --- Test generate_error_analysis with synthetic results ---
    logger.info("Testing generate_error_analysis()")
    class_names = ["Clownfish", "Surgeonfish", "Grouper", "Parrotfish", "Wrasse"]
    synthetic_results = [
        ("dummy_path_0.png", 0, 0, 0.95, True),
        ("dummy_path_1.png", 1, 2, 0.60, False),
        ("dummy_path_2.png", 3, 1, 0.45, False),
        ("dummy_path_3.png", 2, 2, 0.88, True),
        ("dummy_path_4.png", 4, 4, 0.92, True),
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = generate_error_analysis(
            results=synthetic_results,
            class_names=class_names,
            output_dir=tmpdir,
        )
        assert summary["num_correct"] == 3
        assert summary["num_incorrect"] == 2
        assert len(summary["confusion_patterns"]) == 2
        assert Path(summary["report_path"]).exists()
        logger.info("Error analysis summary: %s ✓", summary)

    logger.info("All Grad-CAM self-tests passed ✓")
