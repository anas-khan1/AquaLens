"""
Evaluation module for the Marine Reef Vision project.

Provides comprehensive model evaluation including:
    - Multi-class classification metrics (accuracy, precision, recall, F1)
    - Top-k accuracy from predicted probabilities
    - ROC-AUC (one-vs-rest, macro-averaged)
    - Confusion matrix visualization (counts and percentages)
    - Per-class ROC curve plots
    - Training history plots from CSV logs
    - End-to-end evaluation pipeline with CLI entry point
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tabulate import tabulate

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import get_dataloaders
from src.model import create_model, load_checkpoint
from src.utils import (
    get_device,
    get_effective_config,
    load_config,
    save_metrics,
    setup_logging,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.size": 10,
    }
)


# ============================================================================
# Metric Computation
# ============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth label indices of shape ``(N,)``.
        y_pred: Predicted label indices of shape ``(N,)``.
        y_prob: Predicted probabilities / softmax outputs of shape
            ``(N, C)`` where *C* is the number of classes.
        class_names: Ordered list of human-readable class names.

    Returns:
        Dictionary mapping metric names to their computed values.
        Includes accuracy, macro-precision, macro-recall, macro-F1,
        top-1 accuracy, top-3 accuracy, and (when applicable) ROC-AUC.
    """
    num_classes = len(class_names)

    # --- Basic metrics ---
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(macro_f1),
    }

    # --- Top-k accuracy ---
    for k in (1, 3):
        top_k_acc = _top_k_accuracy(y_true, y_prob, k=k)
        metrics[f"top_{k}_accuracy"] = float(top_k_acc)

    # --- ROC-AUC (one-vs-rest, macro) ---
    if num_classes > 2:
        try:
            # Binarize labels for one-vs-rest AUC computation
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            roc_auc = roc_auc_score(
                y_true_bin, y_prob, average="macro", multi_class="ovr"
            )
            metrics["roc_auc_macro"] = float(roc_auc)
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
            metrics["roc_auc_macro"] = None
    else:
        # Binary case — use probability of the positive class
        try:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics["roc_auc"] = float(roc_auc)
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
            metrics["roc_auc"] = None

    # --- Per-class F1 ---
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_class_f1"] = {
        name: float(score) for name, score in zip(class_names, per_class_f1)
    }

    logger.info(
        "Metrics — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
        acc,
        precision,
        recall,
        macro_f1,
    )
    return metrics


def _top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> float:
    """Compute top-k accuracy from probability predictions.

    For each sample the prediction is correct if the ground-truth label
    appears among the *k* highest-probability classes.

    Args:
        y_true: Ground-truth label indices of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float in ``[0, 1]``.
    """
    # argsort descending and take top-k indices per sample
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(correct))


# ============================================================================
# Classification Report
# ============================================================================


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """Generate a per-class classification report.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered list of class names.

    Returns:
        Tuple of (formatted report string, report dictionary).
    """
    report_str: str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict: Dict[str, Any] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    logger.info("Classification report generated for %d classes", len(class_names))
    return report_str, report_dict


# ============================================================================
# Confusion Matrix Plot
# ============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot and save a confusion matrix heatmap with counts and percentages.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered list of class names.
        save_path: File path to save the figure.
        title: Plot title.
        normalize: If True the matrix cells are normalised to percentages
            (row-wise, i.e. per true class).
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

    # Compute row-normalised percentages for annotations
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with no true samples
    cm_pct = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    # Build annotation strings: "count\n(pct%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    # Determine which data to colour-map
    display_data = cm_pct if normalize else cm

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.9), max(6, num_classes * 0.75)))
    sns.heatmap(
        display_data,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


# ============================================================================
# ROC Curve Plot
# ============================================================================


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Plot one-vs-rest ROC curves for each class plus a macro-average.

    Args:
        y_true: Ground-truth label indices of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        class_names: Ordered list of class names.
        save_path: File path to save the figure.
    """
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Handle the binary edge-case produced by label_binarize
    if num_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Per-class ROC
    fpr_dict: Dict[str, np.ndarray] = {}
    tpr_dict: Dict[str, np.ndarray] = {}
    auc_dict: Dict[str, float] = {}

    for i, name in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        fpr_dict[name] = fpr_i
        tpr_dict[name] = tpr_i
        auc_dict[name] = auc_i
        ax.plot(fpr_i, tpr_i, lw=1.5, label=f"{name} (AUC = {auc_i:.3f})")

    # Macro-average ROC: interpolate all curves onto a common FPR grid
    all_fpr = np.unique(np.concatenate([fpr_dict[n] for n in class_names]))
    mean_tpr = np.zeros_like(all_fpr)
    for name in class_names:
        mean_tpr += np.interp(all_fpr, fpr_dict[name], tpr_dict[name])
    mean_tpr /= num_classes

    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr,
        mean_tpr,
        color="navy",
        lw=2.5,
        linestyle="--",
        label=f"Macro-avg (AUC = {macro_auc:.3f})",
    )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


# ============================================================================
# Training History Plot
# ============================================================================


def plot_training_history(
    csv_path: str,
    save_path: str,
) -> None:
    """Plot training history from a CSV log file.

    Expects columns: ``epoch``, ``train_loss``, ``val_loss``,
    ``train_acc``, ``val_acc``, ``lr``.  Missing columns are silently
    skipped.

    Creates a 2×2 subplot grid:
        - (0,0) Loss curves (train / val)
        - (0,1) Accuracy curves (train / val)
        - (1,0) Learning rate schedule
        - (1,1) Combined overview (loss + accuracy on twin axes)

    Args:
        csv_path: Path to the training log CSV.
        save_path: File path to save the figure.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.warning("Training log not found at %s — skipping history plot", csv_path)
        return

    df = pd.read_csv(csv_file)
    logger.info("Loaded training history with %d epochs from %s", len(df), csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    epochs = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)

    # --- (0, 0) Loss ---
    ax = axes[0, 0]
    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"], label="Train Loss", marker=".")
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val Loss", marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (0, 1) Accuracy ---
    ax = axes[0, 1]
    if "train_acc" in df.columns:
        ax.plot(epochs, df["train_acc"], label="Train Acc", marker=".")
    if "val_acc" in df.columns:
        ax.plot(epochs, df["val_acc"], label="Val Acc", marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (1, 0) Learning Rate ---
    ax = axes[1, 0]
    if "lr" in df.columns:
        ax.plot(epochs, df["lr"], label="Learning Rate", color="tab:green", marker=".")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "LR data not available", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (1, 1) Combined overview ---
    ax = axes[1, 1]
    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"], label="Train Loss", color="tab:red", linestyle="--")
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val Loss", color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax.twinx()
    if "train_acc" in df.columns:
        ax2.plot(epochs, df["train_acc"], label="Train Acc", color="tab:blue", linestyle="--")
    if "val_acc" in df.columns:
        ax2.plot(epochs, df["val_acc"], label="Val Acc", color="tab:blue")
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Merge legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=8)
    ax.set_title("Combined Overview")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


# ============================================================================
# End-to-End Model Evaluation
# ============================================================================


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    task_name: str,
    class_names: List[str],
) -> Dict[str, Any]:
    """Run full evaluation on a test set and generate reports + plots.

    Performs inference, computes all metrics, generates a classification
    report, plots confusion matrix and ROC curves, saves metrics to JSON,
    and prints a formatted summary table.

    Args:
        model: Trained PyTorch model (set to eval mode internally).
        test_loader: DataLoader for the test split.
        cfg: Project configuration (OmegaConf DictConfig).
        task_name: Identifier for the evaluation task (e.g. ``"fish"``
            or ``"coral"``).  Used in file names and titles.
        class_names: Ordered list of class names matching label indices.

    Returns:
        Dictionary of all computed metrics.
    """
    device = torch.device(cfg.get("_device", "cpu"))
    model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    logger.info("Running inference on test set (%d batches)…", len(test_loader))

    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    # --- Metrics ---
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics["task"] = task_name
    metrics["num_test_samples"] = len(y_true)
    metrics["num_classes"] = len(class_names)
    metrics["class_names"] = class_names

    # --- Classification report ---
    report_str, report_dict = generate_classification_report(y_true, y_pred, class_names)
    metrics["classification_report"] = report_dict
    logger.info("\n%s", report_str)

    # --- Output paths ---
    figures_dir = Path(cfg.paths.get("figures_dir", "outputs/figures"))
    metrics_path = Path(cfg.paths.get("output_dir", "outputs")) / f"{task_name}_metrics.json"

    # --- Confusion matrix ---
    if cfg.get("evaluation", {}).get("confusion_matrix", True):
        cm_path = figures_dir / f"{task_name}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            save_path=str(cm_path),
            title=f"{task_name.replace('_', ' ').title()} — Confusion Matrix",
            normalize=False,
        )
        # Also save a normalised version
        cm_norm_path = figures_dir / f"{task_name}_confusion_matrix_normalised.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            save_path=str(cm_norm_path),
            title=f"{task_name.replace('_', ' ').title()} — Normalised Confusion Matrix",
            normalize=True,
        )

    # --- ROC curves ---
    if cfg.get("evaluation", {}).get("compute_roc_auc", True):
        roc_path = figures_dir / f"{task_name}_roc_curves.png"
        try:
            plot_roc_curves(y_true, y_prob, class_names, save_path=str(roc_path))
        except Exception as exc:
            logger.warning("ROC curve plotting failed: %s", exc)

    # --- Training history ---
    history_csv = Path(cfg.paths.get("logs_dir", "outputs/logs")) / f"{task_name}_training_log.csv"
    history_fig = figures_dir / f"{task_name}_training_history.png"
    plot_training_history(csv_path=str(history_csv), save_path=str(history_fig))

    # --- Save metrics JSON ---
    save_metrics(metrics, str(metrics_path))

    # --- Print summary table ---
    _print_summary_table(metrics, task_name, class_names)

    return metrics


def _print_summary_table(
    metrics: Dict[str, Any],
    task_name: str,
    class_names: List[str],
) -> None:
    """Print a formatted summary of evaluation metrics using tabulate.

    Args:
        metrics: Computed metrics dictionary.
        task_name: Name of the evaluation task.
        class_names: List of class names.
    """
    header = f"Evaluation Summary — {task_name}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("=" * len(header))

    # Global metrics table
    global_rows = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Precision (macro)", f"{metrics['precision_macro']:.4f}"],
        ["Recall (macro)", f"{metrics['recall_macro']:.4f}"],
        ["F1 (macro)", f"{metrics['f1_macro']:.4f}"],
        ["Top-1 Accuracy", f"{metrics['top_1_accuracy']:.4f}"],
        ["Top-3 Accuracy", f"{metrics['top_3_accuracy']:.4f}"],
    ]
    roc_key = "roc_auc_macro" if "roc_auc_macro" in metrics else "roc_auc"
    if roc_key in metrics and metrics[roc_key] is not None:
        global_rows.append(["ROC-AUC", f"{metrics[roc_key]:.4f}"])

    global_table = tabulate(
        global_rows,
        headers=["Metric", "Value"],
        tablefmt="fancy_grid",
    )
    logger.info("\n%s", global_table)

    # Per-class F1 table
    per_class_f1 = metrics.get("per_class_f1", {})
    if per_class_f1:
        class_rows = [
            [name, f"{per_class_f1.get(name, 0.0):.4f}"] for name in class_names
        ]
        class_table = tabulate(
            class_rows,
            headers=["Class", "F1 Score"],
            tablefmt="fancy_grid",
        )
        logger.info("\nPer-class F1 scores:\n%s", class_table)


# ============================================================================
# CLI Entry Point
# ============================================================================


def run_evaluation() -> Dict[str, Any]:
    """CLI entry point: load checkpoint, prepare test data, and evaluate.

    Parses command-line arguments for ``--config``, ``--task``, and
    ``--checkpoint``, then orchestrates the full evaluation pipeline.

    Returns:
        Dictionary of computed metrics.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Marine Reef Vision model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["fish", "coral"],
        help="Evaluation task name (determines which dataset split to use).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth) file.",
    )
    args = parser.parse_args()

    # --- Setup ---
    setup_logging()
    cfg = load_config(args.config)
    cfg = get_effective_config(cfg)
    device = get_device(cfg)

    logger.info("Starting evaluation for task '%s'", args.task)
    logger.info("Checkpoint: %s", args.checkpoint)

    # --- Data ---
    _, _, test_loader, class_names = get_dataloaders(cfg, task=args.task)
    logger.info(
        "Test set: %d batches, %d classes (%s)",
        len(test_loader),
        len(class_names),
        ", ".join(class_names),
    )

    # --- Model ---
    num_classes = len(class_names)
    model = create_model(task_name=args.task, num_classes=num_classes, cfg=cfg)
    load_checkpoint(checkpoint_path=args.checkpoint, model=model, map_location=device)
    logger.info("Model loaded from checkpoint")

    # --- Evaluate ---
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        task_name=args.task,
        class_names=class_names,
    )

    logger.info("Evaluation complete for task '%s'", args.task)
    return metrics


if __name__ == "__main__":
    run_evaluation()
