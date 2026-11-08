"""Batch inference pipeline for Marine Reef Vision.

Processes a folder of underwater images, applying the full pipeline:
    1. Preprocessing (Dark Channel Prior + CLAHE)
    2. ResNet-34 Classification (Fish species or Coral bleaching)
    3. Coral Bleaching Severity Assessment (Heuristic based on Grad-CAM + HSV)
    4. Batch aggregation & Reef Health Index (RHI) calculation
    5. Output generation (CSV table, JSON summary, optional visualization overlays)

Usage:
    python -m src.batch_inference \\
        --input-dir path/to/images \\
        --task coral \\
        --checkpoint outputs/checkpoints/coral_best.pth \\
        --config config.yaml \\
        --output-csv outputs/batch_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.gradcam import GradCAM, overlay_heatmap
from src.model import MarineClassifier, create_model, load_checkpoint
from src.preprocessing import preprocess_image
from src.severity import SeverityAssessor, compute_reef_health_index
from src.utils import get_device, get_effective_config, load_config, setup_logging

logger = logging.getLogger(__name__)

# Standard supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference on underwater marine and coral images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing images to process, or path to a single image.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["fish", "coral"],
        help="Task type: 'fish' for species identification, 'coral' for reef health.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/batch_inference_results.csv",
        help="Path to save output results CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/reef_health_summary.json",
        help="Path to save aggregate summary JSON.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Whether to save Grad-CAM and enhancement visualization overlays.",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="outputs/batch_visualizations",
        help="Directory to save visual outputs if --save-visualizations is set.",
    )
    return parser.parse_args()


def get_image_files(input_path: Path) -> List[Path]:
    """Gather all image paths from a file or folder recursively."""
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTENSIONS:
            return [input_path]
        return []

    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(input_path.rglob(f"*{ext}"))
        images.extend(input_path.rglob(f"*{ext.upper()}"))
    return sorted(list(set(images)))


def run_batch_inference(
    image_paths: List[Path],
    model: MarineClassifier,
    class_names: List[str],
    task: str,
    cfg: Any,
    device: torch.device,
    save_visualizations: bool = False,
    vis_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute batch inference across images and return results and summary.

    Args:
        image_paths: List of paths to underwater images.
        model: Loaded MarineClassifier model.
        class_names: List of class name labels corresponding to model outputs.
        task: 'fish' or 'coral'.
        cfg: Configuration dictionary.
        device: Torch compute device.
        save_visualizations: Whether to save visual overlays.
        vis_dir: Directory to save visual overlays.

    Returns:
        Tuple of (results_dataframe, summary_dict).
    """
    model.eval()
    results: List[Dict[str, Any]] = []

    if save_visualizations and vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Setup Grad-CAM and SeverityAssessor
    gradcam = GradCAM(model, target_layer_name=cfg.gradcam.target_layer)
    severity_assessor = SeverityAssessor(model, cfg) if task == "coral" else None

    # Normalization constants (ImageNet stats)
    mean = np.array(cfg.preprocessing.normalize.mean, dtype=np.float32)
    std = np.array(cfg.preprocessing.normalize.std, dtype=np.float32)

    image_size = cfg.get("_effective_image_size", cfg.data.image_size_gpu)

    start_time = time.time()
    logger.info("Processing %d images for %s task on %s...", len(image_paths), task, device)

    for idx, img_path in enumerate(image_paths):
        # Read image
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Could not read image: %s. Skipping.", img_path)
            continue

        orig_h, orig_w = bgr.shape[:2]

        # 1. Preprocessing (DCP + CLAHE)
        enhanced_bgr = preprocess_image(
            bgr,
            cfg,
            apply_dark_channel=True,
            apply_clahe_flag=True,
            target_size=image_size,
        )

        # Prepare tensor (RGB normalized)
        rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - mean) / std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        # 2. Forward inference
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"

        record: Dict[str, Any] = {
            "filename": img_path.name,
            "filepath": str(img_path),
            "task": task,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
        }

        # 3. Top-k predictions for fish
        if task == "fish":
            top_k_indices = np.argsort(probs)[::-1][:min(3, len(probs))]
            top3_info = [
                f"{class_names[i]}: {probs[i]:.2%}"
                for i in top_k_indices
            ]
            record["top3_predictions"] = " | ".join(top3_info)

        # 4. Coral Bleaching Severity Assessment
        if task == "coral" and severity_assessor is not None:
            # Generate Grad-CAM heatmap
            heatmap = gradcam.generate(tensor, target_class=pred_idx)
            # Use severity assessor heuristic
            assessment = severity_assessor.assess(enhanced_bgr)
            record["severity"] = assessment["severity"]
            record["severity_index"] = assessment["severity_index"]
            record["bleaching_score"] = round(assessment["bleaching_score"], 4)
            record["is_bleached"] = (assessment["severity_index"] > 0)

            # Optional visualization
            if save_visualizations and vis_dir:
                overlay = overlay_heatmap(enhanced_bgr, heatmap, alpha=cfg.gradcam.alpha)
                # Create side-by-side composite
                bgr_resized = cv2.resize(bgr, (image_size, image_size))
                composite = np.hstack([bgr_resized, enhanced_bgr, overlay])
                out_path = vis_dir / f"{img_path.stem}_analysis.png"
                cv2.imwrite(str(out_path), composite)

        elif task == "fish" and save_visualizations and vis_dir:
            heatmap = gradcam.generate(tensor, target_class=pred_idx)
            overlay = overlay_heatmap(enhanced_bgr, heatmap, alpha=cfg.gradcam.alpha)
            bgr_resized = cv2.resize(bgr, (image_size, image_size))
            composite = np.hstack([bgr_resized, enhanced_bgr, overlay])
            out_path = vis_dir / f"{img_path.stem}_fish_gradcam.png"
            cv2.imwrite(str(out_path), composite)

        results.append(record)

    gradcam.remove_hooks()
    total_time = time.time() - start_time
    avg_latency_ms = (total_time / max(len(results), 1)) * 1000.0

    df_results = pd.DataFrame(results)

    # 5. Aggregate Summary
    summary: Dict[str, Any] = {
        "task": task,
        "total_images_processed": len(results),
        "total_execution_time_sec": round(total_time, 2),
        "avg_latency_ms_per_image": round(avg_latency_ms, 2),
        "device": str(device),
    }

    if task == "fish":
        species_counts = df_results["prediction"].value_counts().to_dict()
        summary["species_distribution"] = species_counts
        summary["predominant_species"] = df_results["prediction"].mode()[0] if not df_results.empty else None

    elif task == "coral":
        rhi_metrics = compute_reef_health_index(results)
        summary["reef_health_index"] = rhi_metrics.get("rhi", 0.0)
        summary["reef_status"] = (
            "Healthy / Low Risk" if summary["reef_health_index"] >= 0.75
            else "Moderate Alert" if summary["reef_health_index"] >= 0.50
            else "Severe Bleaching Crisis"
        )
        summary["severity_distribution"] = rhi_metrics.get("severity_distribution", {})
        summary["bleached_percentage"] = (
            round((df_results["is_bleached"].sum() / len(df_results)) * 100, 2)
            if not df_results.empty else 0.0
        )

    logger.info("Batch inference completed in %.2fs (%.1f ms/image)", total_time, avg_latency_ms)
    return df_results, summary


def main() -> None:
    setup_logging()
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)
    cfg = get_effective_config(cfg)
    device = get_device(cfg)

    # Discover images
    input_path = Path(args.input_dir)
    image_paths = get_image_files(input_path)
    if not image_paths:
        logger.error("No valid images found in: %s", input_path)
        sys.exit(1)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at: %s", ckpt_path)
        sys.exit(1)

    # Determine classes from checkpoint or default
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt.get("class_names", None)
    if not class_names:
        # Fallback default classes
        if args.task == "coral":
            class_names = ["healthy", "bleached"]
        else:
            class_names = [f"Class_{i}" for i in range(10)]

    num_classes = len(class_names)
    model = create_model(args.task, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    logger.info("Model loaded successfully with classes: %s", class_names)

    # Run batch inference
    vis_dir = Path(args.vis_dir) if args.save_visualizations else None
    df_results, summary = run_batch_inference(
        image_paths=image_paths,
        model=model,
        class_names=class_names,
        task=args.task,
        cfg=cfg,
        device=device,
        save_visualizations=args.save_visualizations,
        vis_dir=vis_dir,
    )

    # Save CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    logger.info("Detailed results saved to %s", out_csv)

    # Save JSON summary
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary metrics saved to %s", out_json)

    # Print summary
    print("\n" + "=" * 60)
    print(f"       BATCH INFERENCE SUMMARY ({args.task.upper()})")
    print("=" * 60)
    print(f"Total Images: {summary['total_images_processed']}")
    print(f"Execution Time: {summary['total_execution_time_sec']}s ({summary['avg_latency_ms_per_image']} ms/img)")
    if args.task == "coral":
        print(f"Reef Health Index (RHI): {summary['reef_health_index']:.3f} [{summary['reef_status']}]")
        print(f"Bleached Ratio: {summary['bleached_percentage']}%")
        print("Severity Breakdown:", summary["severity_distribution"])
    else:
        print("Species Breakdown:", summary.get("species_distribution", {}))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

