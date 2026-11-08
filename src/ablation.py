"""Ablation Study Module for Marine Reef Vision.

Systematically evaluates the impact of each preprocessing component:
    Variant 1: Raw Images (no enhancement)
    Variant 2: CLAHE Only (L-channel histogram equalization)
    Variant 3: Dark Channel Prior Only (optical color correction & dehazing)
    Variant 4: Combined DCP + CLAHE (full proposed preprocessing pipeline)

Computes:
    - Classification metrics (Accuracy, Macro-Precision, Macro-Recall, Macro-F1)
    - Quantitative image quality metrics (Shannon Entropy, RMS Contrast, Colorfulness)
    - Visual comparative bar plots saved to outputs/figures/ablation_comparison.png
    - Outputs tabular summary (CSV and JSON)

Usage:
    python -m src.ablation --task coral --cpu-mode
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tabulate import tabulate

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import MarineDataset, create_stratified_splits, get_dataloaders
from src.evaluate import compute_metrics
from src.model import MarineClassifier, create_model
from src.preprocessing import compute_image_quality_metrics, preprocess_image
from src.train import train_model
from src.utils import get_device, get_effective_config, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def evaluate_variant(
    variant_name: str,
    apply_dcp: bool,
    apply_clahe: bool,
    cfg: DictConfig,
    task: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Train and evaluate a specific preprocessing variant.

    Args:
        variant_name: Label for the variant.
        apply_dcp: Whether DCP is active.
        apply_clahe: Whether CLAHE is active.
        cfg: Configuration object.
        task: 'coral' or 'fish'.
        device: Torch compute device.

    Returns:
        Dictionary of performance metrics and image quality metrics.
    """
    logger.info("=" * 60)
    logger.info("Evaluating Ablation Variant: %s (DCP=%s, CLAHE=%s)", variant_name, apply_dcp, apply_clahe)
    logger.info("=" * 60)

    # 1. Get dataloaders with specific preprocessing settings
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        cfg,
        task=task,
        apply_dcp=apply_dcp,
        apply_clahe=apply_clahe,
    )

    num_classes = len(class_names)
    model = create_model(task_name=f"{task}_{variant_name}", num_classes=num_classes, cfg=cfg).to(device)

    # 2. Train model for this variant
    ckpt_path, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        task_name=f"{task}_{variant_name}",
        class_names=class_names,
    )

    # 3. Evaluate on test set
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    # 4. Compute average image quality metrics across test set images
    raw_images_dir = Path(cfg.paths.coral_data if task == "coral" else cfg.paths.fish_data)
    test_files = list(raw_images_dir.rglob("*.png"))[:10]
    entropies, contrasts, colorfulnesses = [], [], []

    for f in test_files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        proc = preprocess_image(
            bgr,
            cfg,
            apply_dark_channel=apply_dcp,
            apply_clahe_flag=apply_clahe,
            target_size=128,
        )
        q = compute_image_quality_metrics(proc)
        entropies.append(q["entropy"])
        contrasts.append(q["contrast"])
        colorfulnesses.append(q["colorfulness"])

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    mean_colorfulness = float(np.mean(colorfulnesses)) if colorfulnesses else 0.0

    result = {
        "variant": variant_name,
        "dcp_enabled": apply_dcp,
        "clahe_enabled": apply_clahe,
        "accuracy": round(metrics["accuracy"], 4),
        "precision_macro": round(metrics["precision_macro"], 4),
        "recall_macro": round(metrics["recall_macro"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
        "mean_entropy": round(mean_entropy, 3),
        "mean_contrast": round(mean_contrast, 3),
        "mean_colorfulness": round(mean_colorfulness, 2),
    }
    return result


def plot_ablation_results(results: List[Dict[str, Any]], save_path: Path) -> None:
    """Create publication-grade ablation comparison charts."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Model Accuracy & F1
    x = np.arange(len(df))
    width = 0.35

    axes[0].bar(x - width / 2, df["accuracy"], width, label="Accuracy", color="#1f77b4", alpha=0.9)
    axes[0].bar(x + width / 2, df["f1_macro"], width, label="Macro F1", color="#ff7f0e", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["variant"], fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Classification Performance by Preprocessing Variant", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Image Quality Metrics (Entropy & RMS Contrast)
    axes[1].bar(x - width / 2, df["mean_entropy"], width, label="Entropy (bits)", color="#2ca02c", alpha=0.9)
    axes[1].bar(x + width / 2, df["mean_contrast"] * 10, width, label="RMS Contrast (×10)", color="#d62728", alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["variant"], fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].set_title("Optical Quality Indicators by Variant", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Ablation study comparison figure saved to %s", save_path)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run underwater preprocessing ablation study.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--task", type=str, default="coral", choices=["coral", "fish"])
    parser.add_argument("--cpu-mode", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cpu_mode:
        cfg.cpu_mode = True
    cfg = get_effective_config(cfg)
    device = get_device(cfg)
    set_seed(cfg.seed)

    variants = [
        ("Raw Images", False, False),
        ("CLAHE Only", False, True),
        ("DCP Only", True, False),
        ("DCP + CLAHE (Full)", True, True),
    ]

    ablation_records = []
    for name, dcp, clahe in variants:
        rec = evaluate_variant(name, dcp, clahe, cfg, args.task, device)
        ablation_records.append(rec)

    # Save CSV
    df = pd.DataFrame(ablation_records)
    out_csv = Path(cfg.paths.output_dir) / f"{args.task}_ablation_table.csv"
    df.to_csv(out_csv, index=False)

    # Save JSON
    out_json = Path(cfg.paths.output_dir) / f"{args.task}_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_records, f, indent=2)

    # Plot
    fig_path = Path(cfg.paths.figures_dir) / f"{args.task}_ablation_comparison.png"
    plot_ablation_results(ablation_records, fig_path)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"       UNDERWATER PREPROCESSING ABLATION STUDY RESULTS ({args.task.upper()})")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
