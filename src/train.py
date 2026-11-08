"""
Training module for the Marine Reef Vision project.

Provides:
    - EarlyStopping callback to halt training when validation plateaus
    - CSVLogger for per-epoch metric logging
    - Mixed-precision training loop with tqdm progress bars
    - Model checkpointing with best-model tracking
    - Complete CLI entry point for end-to-end training

Typical usage::

    python -m src.train --config config.yaml --task fish
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import get_dataloaders
from src.model import ProgressiveUnfreezer, create_model, save_checkpoint
from src.utils import (
    ensure_dirs,
    get_device,
    get_effective_config,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitors a validation metric and signals when training should stop.

    Supports monitoring ``val_loss`` (lower is better) or ``val_acc``
    (higher is better).

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        monitor: Metric to watch — ``"val_loss"`` or ``"val_acc"``.

    Raises:
        ValueError: If *monitor* is not one of the supported metrics.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
    ) -> None:
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError(
                f"monitor must be 'val_loss' or 'val_acc', got '{monitor}'"
            )
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

        # For val_loss, lower is better; for val_acc, higher is better
        self._is_better = (
            (lambda new, best: new < best - self.min_delta)
            if monitor == "val_loss"
            else (lambda new, best: new > best + self.min_delta)
        )

        logger.info(
            "EarlyStopping initialised: monitor=%s, patience=%d, min_delta=%.4f",
            monitor,
            patience,
            min_delta,
        )

    def step(self, metric: float) -> bool:
        """Check whether training should stop based on the latest metric.

        Args:
            metric: Current value of the monitored metric.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self.best_score is None:
            self.best_score = metric
            logger.debug(
                "EarlyStopping: initial %s=%.6f", self.monitor, metric
            )
            return False

        if self._is_better(metric, self.best_score):
            logger.debug(
                "EarlyStopping: %s improved from %.6f to %.6f",
                self.monitor,
                self.best_score,
                metric,
            )
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs (best %s=%.6f)",
                self.counter,
                self.patience,
                self.monitor,
                self.best_score,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(
                    "EarlyStopping triggered after %d epochs without improvement",
                    self.patience,
                )
        return self.should_stop


# ---------------------------------------------------------------------------
# CSVLogger
# ---------------------------------------------------------------------------


class CSVLogger:
    """Logs per-epoch training metrics to a CSV file.

    Columns written: ``epoch, train_loss, train_acc, val_loss, val_acc,
    lr, time_elapsed``.

    Args:
        log_path: File path for the CSV output.
    """

    _FIELDNAMES: List[str] = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "time_elapsed",
    ]

    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write header
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()

        logger.info("CSVLogger writing to %s", self.log_path)

    def log(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        time_elapsed: float,
    ) -> None:
        """Append a row of metrics for one epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            train_loss: Average training loss.
            train_acc: Training accuracy (fraction 0–1).
            val_loss: Average validation loss.
            val_acc: Validation accuracy (fraction 0–1).
            lr: Current learning rate.
            time_elapsed: Wall-clock seconds elapsed since training start.
        """
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.4f}",
            "lr": f"{lr:.8f}",
            "time_elapsed": f"{time_elapsed:.1f}",
        }
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Training Loop — Single Epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run a single training epoch.

    Args:
        model: The neural network.
        loader: Training DataLoader.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimiser instance.
        device: Compute device.
        epoch: Current epoch index (for display only).
        scaler: Optional GradScaler for mixed-precision training.
        use_amp: Whether to enable automatic mixed precision.

    Returns:
        Dictionary with ``train_loss`` and ``train_acc``.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc=f"  Train Epoch {epoch}",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass — optionally with automatic mixed precision
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass — scale gradients when using AMP
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate statistics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size

        # Update progress bar
        pbar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    train_loss = running_loss / total
    train_acc = correct / total
    return {"train_loss": train_loss, "train_acc": train_acc}


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Run the validation / evaluation loop.

    Args:
        model: The neural network (switched to eval mode internally).
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.
        epoch: Current epoch index (for display only).
        use_amp: Whether to enable automatic mixed precision.

    Returns:
        Dictionary with ``val_loss``, ``val_acc``, ``all_preds``
        (list of int), and ``all_labels`` (list of int).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(
        loader,
        desc=f"  Val   Epoch {epoch}",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    val_loss = running_loss / total
    val_acc = correct / total
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Main Training Orchestrator
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    task_name: str,
    class_names: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, List[float]]]:
    """Full training loop with scheduling, checkpointing, and logging.

    Args:
        model: The neural network to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Effective project configuration (already resolved via
            :func:`src.utils.get_effective_config`).
        task_name: ``"fish"`` or ``"coral"`` — used in checkpoint naming.

    Returns:
        A tuple of *(best_ckpt_path, history)* where *history* maps
        ``"train_loss"``, ``"train_acc"``, ``"val_loss"``, ``"val_acc"``,
        and ``"lr"`` to lists of per-epoch values.
    """
    device = torch.device(cfg._device)
    model = model.to(device)

    epochs = cfg._effective_epochs
    use_amp = cfg._effective_mixed_precision and device.type == "cuda"

    # ---- Optimizer --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    logger.info(
        "Optimiser: AdamW (lr=%.6f, weight_decay=%.4f)",
        cfg.training.learning_rate,
        cfg.training.weight_decay,
    )

    # ---- LR Scheduler -----------------------------------------------------
    scheduler_name = cfg.training.get("scheduler", "onecycle")
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.max_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=cfg.training.get("pct_start", 0.3),
        )
        scheduler_step_per_batch = True
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
        scheduler_step_per_batch = False
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    logger.info("LR scheduler: %s", scheduler_name)

    # ---- Mixed Precision Scaler -------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    logger.info("Mixed precision (AMP): %s", "enabled" if use_amp else "disabled")

    # ---- Progressive Unfreezer --------------------------------------------
    unfreezer = ProgressiveUnfreezer(model, cfg.model.unfreeze_schedule)

    # ---- Loss Function ----------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # ---- Early Stopping ---------------------------------------------------
    es_cfg = cfg.training.early_stopping
    early_stopper = EarlyStopping(
        patience=es_cfg.patience,
        min_delta=es_cfg.min_delta,
        monitor=es_cfg.monitor,
    )

    # ---- CSV Logger -------------------------------------------------------
    csv_log_path = str(
        Path(cfg.paths.logs_dir) / f"{task_name}_training_log.csv"
    )
    csv_logger = CSVLogger(csv_log_path)

    # ---- TensorBoard (optional) -------------------------------------------
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_log_dir = str(Path(cfg.paths.logs_dir) / f"tensorboard_{task_name}")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info("TensorBoard logging to %s", tb_log_dir)
    except ImportError:
        logger.info("TensorBoard not installed — skipping TB logging")

    # ---- Checkpoint directory ---------------------------------------------
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = str(ckpt_dir / f"{task_name}_best.pth")

    # ---- History tracking -------------------------------------------------
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # ---- Training Loop ----------------------------------------------------
    start_time = time.time()
    logger.info(
        "Starting training: %d epochs, device=%s, task=%s",
        epochs,
        device,
        task_name,
    )

    for epoch in range(epochs):
        epoch_start = time.time()

        # Progressive unfreezing — adjust which layers are trainable
        unfreezer.step(epoch)

        # --- Train one epoch -----------------------------------------------
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
        )

        # Step OneCycleLR per-batch has already been handled inside
        # optimizer steps; for CosineAnnealing step once per epoch.
        if not scheduler_step_per_batch:
            scheduler.step()

        # --- Validate ------------------------------------------------------
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_start

        # --- Log to console ------------------------------------------------
        logger.info(
            "Epoch %d/%d | train_loss=%.4f  train_acc=%.4f | "
            "val_loss=%.4f  val_acc=%.4f | lr=%.2e | %.1fs",
            epoch + 1,
            epochs,
            train_metrics["train_loss"],
            train_metrics["train_acc"],
            val_metrics["val_loss"],
            val_metrics["val_acc"],
            current_lr,
            epoch_time,
        )

        # --- Record history ------------------------------------------------
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["lr"].append(current_lr)

        # --- CSV logging ---------------------------------------------------
        csv_logger.log(
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_acc"],
            val_loss=val_metrics["val_loss"],
            val_acc=val_metrics["val_acc"],
            lr=current_lr,
            time_elapsed=elapsed,
        )

        # --- TensorBoard logging -------------------------------------------
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            tb_writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            tb_writer.add_scalar("Accuracy/train", train_metrics["train_acc"], epoch)
            tb_writer.add_scalar("Accuracy/val", val_metrics["val_acc"], epoch)
            tb_writer.add_scalar("LR", current_lr, epoch)

        # --- Checkpoint best model -----------------------------------------
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_metrics["val_loss"],
                val_acc=val_metrics["val_acc"],
                path=best_ckpt_path,
                class_names=class_names,
            )
            logger.info(
                "Best model saved → %s (val_loss=%.6f)",
                best_ckpt_path,
                best_val_loss,
            )

        # --- Early Stopping ------------------------------------------------
        monitor_value = val_metrics[early_stopper.monitor]
        if early_stopper.step(monitor_value):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # ---- Cleanup ----------------------------------------------------------
    if tb_writer is not None:
        tb_writer.close()

    total_time = time.time() - start_time
    logger.info(
        "Training complete in %.1fs (%d epochs). Best val_loss=%.6f",
        total_time,
        epoch + 1,
        best_val_loss,
    )

    return best_ckpt_path, history


# ---------------------------------------------------------------------------
# End-to-End Pipeline
# ---------------------------------------------------------------------------


def run_training(
    config_path: str = "config.yaml",
    task: str = "fish",
    cpu_mode: bool = False,
) -> None:
    """Complete training pipeline: config → data → model → train → save.

    Args:
        config_path: Path to the YAML configuration file.
        task: Task identifier — ``"fish"`` or ``"coral"``.
        cpu_mode: If ``True``, force CPU-friendly settings regardless of
            GPU availability.
    """
    # ---- Setup ------------------------------------------------------------
    setup_logging()
    cfg = load_config(config_path)

    if cpu_mode:
        cfg.cpu_mode = True

    cfg = get_effective_config(cfg)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("Marine Reef Vision — Training Pipeline")
    logger.info("Task: %s | Config: %s", task, config_path)
    logger.info("=" * 60)

    # ---- Data -------------------------------------------------------------
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg, task=task)
    num_classes = len(class_names)
    logger.info(
        "Data loaded: %d classes, %d train batches, %d val batches",
        num_classes,
        len(train_loader),
        len(val_loader),
    )

    # ---- Model ------------------------------------------------------------
    model = create_model(
        task_name=task,
        num_classes=num_classes,
        cfg=cfg,
    )

    # ---- Train ------------------------------------------------------------
    best_ckpt, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        task_name=task,
        class_names=class_names,
    )

    logger.info("Best checkpoint: %s", best_ckpt)
    logger.info(
        "Final val_loss=%.4f  val_acc=%.4f",
        history["val_loss"][-1],
        history["val_acc"][-1],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Marine Reef Vision classifier.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["fish", "coral"],
        default="fish",
        help="Training task: 'fish' for species classification, "
        "'coral' for bleaching detection (default: fish).",
    )
    parser.add_argument(
        "--cpu-mode",
        action="store_true",
        default=False,
        help="Force CPU-friendly settings (smaller images, fewer epochs).",
    )
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        task=args.task,
        cpu_mode=args.cpu_mode,
    )
