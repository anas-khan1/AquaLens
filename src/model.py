"""Model module for marine reef classification.

Provides a ResNet-34-based classifier with progressive unfreezing,
checkpoint management, ONNX export, and inference benchmarking.

Typical usage::

    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config.yaml")
    model = create_model(
        task_name="fish",
        num_classes=10,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )
    unfreezer = ProgressiveUnfreezer(model, cfg.model.unfreeze_schedule)
    unfreezer.step(epoch=0)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MarineClassifier
# ---------------------------------------------------------------------------

class MarineClassifier(nn.Module):
    """ResNet-34 backbone with a custom classification head.

    The final fully-connected layer of the pretrained ResNet-34 is replaced
    with ``Dropout(p) → Linear(512, num_classes)`` so that the network can
    be fine-tuned on underwater reef imagery.

    Args:
        num_classes: Number of target classes.
        pretrained: Whether to load ImageNet-pretrained weights for the
            backbone.
        dropout: Dropout probability applied before the final linear layer.

    Attributes:
        backbone: The ResNet-34 feature extractor (everything before the
            original ``fc`` layer).
        fc: The replacement classification head
            (``Sequential[Dropout, Linear]``).
    """

    # Number of features produced by the ResNet-34 average-pool layer.
    _BACKBONE_FEATURES: int = 512

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Load the full ResNet-34 (with or without pretrained weights).
        if pretrained:
            try:
                weights = models.ResNet34_Weights.DEFAULT
                resnet = models.resnet34(weights=weights)
            except Exception as exc:
                logger.warning(
                    "Could not load ImageNet pretrained weights (%s). "
                    "Proceeding with randomly initialized backbone weights.",
                    exc,
                )
                resnet = models.resnet34(weights=None)
        else:
            resnet = models.resnet34(weights=None)

        # Preserve every layer *except* the original fc as the backbone.
        # nn.Sequential preserves the forward-pass order used by ResNet:
        #   conv1 → bn1 → relu → maxpool → layer1‥layer4 → avgpool
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Keep named references to the residual blocks so that
        # ProgressiveUnfreezer (and Grad-CAM) can address them by name.
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Custom classification head.
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self._BACKBONE_FEATURES, num_classes),
        )

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        h = torch.flatten(h, 1)  # (B, 512)
        logits = self.fc(h)
        return logits

    # ---- Grad-CAM helper --------------------------------------------------

    def get_feature_extractor(
        self,
        target_layer: str = "layer4",
    ) -> nn.Module:
        """Return the sub-module up to *target_layer* (inclusive).

        This is useful for Grad-CAM: register a forward hook on the
        returned module's last conv layer to capture feature-map activations.

        Args:
            target_layer: Name of the backbone stage to extract features
                from.  Must be one of ``"layer1"`` … ``"layer4"``.

        Returns:
            The requested ``nn.Module``.

        Raises:
            ValueError: If *target_layer* is not a valid backbone stage.
        """
        valid_layers = {"layer1", "layer2", "layer3", "layer4"}
        if target_layer not in valid_layers:
            raise ValueError(
                f"target_layer must be one of {valid_layers}, "
                f"got '{target_layer}'"
            )
        return getattr(self, target_layer)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    task_name: Optional[str] = "marine",
    num_classes: int = 2,
    pretrained: Optional[bool] = None,
    dropout: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
    **kwargs: Any,
) -> MarineClassifier:
    """Instantiate a :class:`MarineClassifier` and log a parameter summary.

    Supports both explicit argument specification and DictConfig configuration.

    Args:
        task_name: Descriptive label for the task ("fish" or "coral").
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained backbone weights.
        dropout: Dropout probability for the classification head.
        cfg: Optional full experiment configuration (DictConfig).
        **kwargs: Additional parameters.

    Returns:
        A ready-to-train :class:`MarineClassifier`.
    """
    if cfg is not None:
        if pretrained is None:
            pretrained = getattr(cfg.model, "pretrained", True)
        if dropout is None:
            dropout = getattr(cfg.model, "dropout", 0.3)
    if pretrained is None:
        pretrained = True
    if dropout is None:
        dropout = 0.3

    model = MarineClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    # Parameter summary ---------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    logger.info(
        "Created MarineClassifier for task '%s'  |  classes=%d  "
        "pretrained=%s  dropout=%.2f",
        task_name,
        num_classes,
        pretrained,
        dropout,
    )
    logger.info(
        "Parameter summary — total: %s  trainable: %s  frozen: %s",
        f"{total_params:,}",
        f"{trainable_params:,}",
        f"{frozen_params:,}",
    )

    return model


# ---------------------------------------------------------------------------
# Progressive Unfreezing
# ---------------------------------------------------------------------------

class ProgressiveUnfreezer:
    """Gradually unfreeze backbone layers according to an epoch schedule.

    A common transfer-learning strategy is to start training with only the
    classification head unfrozen, then progressively unfreeze deeper layers
    as training proceeds.  This class automates that process.

    The *schedule* maps **epoch numbers** to the list of layer-name prefixes
    that should be **trainable** at that epoch.  The special value ``"all"``
    means every parameter becomes trainable.

    Example YAML::

        unfreeze_schedule:
          0:  ["fc"]
          3:  ["layer4", "fc"]
          6:  ["layer3", "layer4", "fc"]
          10: "all"

    Args:
        model: The :class:`MarineClassifier` (or compatible ``nn.Module``)
            whose parameters will be managed.
        schedule: Mapping of ``{epoch: layer_names | "all"}``.  Keys are
            integers (or integer-convertible strings); values are lists of
            layer-name prefixes or the literal string ``"all"``.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: Union[Dict[int, Any], DictConfig],
    ) -> None:
        self.model = model

        # Normalise the schedule: ensure integer keys, list/str values.
        self.schedule: Dict[int, Union[str, List[str]]] = {}
        raw = OmegaConf.to_container(schedule, resolve=True) \
            if isinstance(schedule, (DictConfig, ListConfig)) else schedule
        for epoch_key, layers in raw.items():
            self.schedule[int(epoch_key)] = layers

        self._current_trainable: Optional[Union[str, List[str]]] = None

        # Start fully frozen.
        self._freeze_all()

    # ---- public API -------------------------------------------------------

    def step(self, epoch: int) -> None:
        """Apply the unfreeze schedule for the given *epoch*.

        If *epoch* is not in the schedule, the model's freeze state is
        left unchanged.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        if epoch not in self.schedule:
            return

        target = self.schedule[epoch]

        # Avoid redundant work when the schedule entry hasn't changed.
        if target == self._current_trainable:
            return

        self._current_trainable = target

        if target == "all":
            self._unfreeze_all()
            logger.info(
                "Epoch %d — unfreezing ALL layers (%s trainable params)",
                epoch,
                f"{self._count_trainable():,}",
            )
            return

        # Freeze everything, then selectively unfreeze.
        self._freeze_all()
        trainable_names: List[str] = (
            [target] if isinstance(target, str) else list(target)
        )
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in trainable_names):
                param.requires_grad = True

        logger.info(
            "Epoch %d — trainable layers: %s  (%s trainable params)",
            epoch,
            trainable_names,
            f"{self._count_trainable():,}",
        )

    # ---- internals --------------------------------------------------------

    def _freeze_all(self) -> None:
        """Set ``requires_grad=False`` on every parameter."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_all(self) -> None:
        """Set ``requires_grad=True`` on every parameter."""
        for param in self.model.parameters():
            param.requires_grad = True

    def _count_trainable(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    filename: str = "checkpoint.pth",
    path: Optional[Union[str, Path]] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    class_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Path:
    """Persist a training checkpoint to disk.

    Supports either (checkpoint_dir, filename) or a direct path argument.
    """
    if path is not None:
        ckpt_path = Path(path)
    elif checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / filename
    else:
        ckpt_path = Path("outputs/checkpoints") / filename

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    metric_val = best_metric if best_metric is not None else (val_loss if val_loss is not None else 0.0)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
        "epoch": epoch,
        "best_metric": metric_val,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "class_names": class_names,
    }
    if cfg is not None:
        try:
            payload["config"] = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            payload["config"] = str(cfg)

    torch.save(payload, ckpt_path)
    logger.info("Checkpoint saved → %s  (epoch %d, metric=%.4f)", ckpt_path, epoch, metric_val)
    return ckpt_path


def load_checkpoint(
    checkpoint_path: Union[str, Path, nn.Module],
    model: Optional[Union[str, Path, nn.Module]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Any:
    """Restore a model (and optionally an optimizer) from a checkpoint.

    Supports either load_checkpoint(path, model) or load_checkpoint(model, path).
    """
    if isinstance(checkpoint_path, nn.Module):
        model, checkpoint_path = checkpoint_path, model

    if device is not None:
        map_location = device
    if map_location is None:
        map_location = torch.device("cpu")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload: Dict[str, Any] = torch.load(
        ckpt_path, map_location=map_location, weights_only=False,
    )
    if model is not None:
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        model.to(map_location)
        logger.info("Model state restored from %s", ckpt_path)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        logger.info("Optimizer state restored from %s", ckpt_path)

    res = {
        "epoch": payload.get("epoch", 0),
        "best_metric": payload.get("best_metric", 0.0),
        "config": payload.get("config", {}),
        "class_names": payload.get("class_names", []),
    }
    # Return model if caller expects it
    class ModelDictResult(dict):
        def __init__(self, d, mod):
            super().__init__(d)
            self._model = mod

        def __call__(self, *args, **kw):
            return self._model(*args, **kw)

        def __getattr__(self, name):
            if name in self:
                return self[name]
            return getattr(self._model, name)

    return ModelDictResult(res, model) if model is not None else res


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: nn.Module,
    dummy_input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    output_path: Union[str, Path] = "model.onnx",
    opset_version: int = 17,
) -> Path:
    """Export the model to ONNX format and validate the graph.

    Args:
        model: The trained (or untrained) model to export.
        dummy_input_shape: Shape of the dummy tensor fed through the model
            during tracing.  Defaults to ``(1, 3, 224, 224)``.
        output_path: Destination file path for the ``.onnx`` model.
        opset_version: ONNX opset version to target.

    Returns:
        The resolved ``Path`` of the exported ONNX file.

    Raises:
        ImportError: If the ``onnx`` package is not installed.
        onnx.checker.ValidationError: If the exported graph is invalid.
    """
    try:
        import onnx          # noqa: F811 — lazy import
        import onnx.checker
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with:  pip install onnx"
        ) from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*dummy_input_shape)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Validate the exported graph.
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)

    logger.info(
        "ONNX model exported and validated → %s  (opset %d)",
        out,
        opset_version,
    )
    return out


# ---------------------------------------------------------------------------
# Inference benchmarking
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    n_iterations: int = 100,
    n_warmup: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Measure model inference latency on CPU and (optionally) GPU.

    Runs *n_warmup* warm-up iterations followed by *n_iterations* timed
    iterations.  Reports per-device mean and standard deviation of the
    forward-pass latency in **milliseconds**.

    Args:
        model: The model to benchmark.
        input_shape: Shape of the random input tensor.
        n_iterations: Number of timed iterations per device.
        n_warmup: Number of untimed warm-up iterations per device.

    Returns:
        Dictionary of the form::

            {
                "cpu": {"mean_ms": float, "std_ms": float},
                "cuda": {"mean_ms": float, "std_ms": float},  # if available
            }
    """
    results: Dict[str, Dict[str, float]] = {}

    devices: List[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        dev_name = str(device)
        m = model.to(device)
        m.eval()

        x = torch.randn(*input_shape, device=device)

        # Warm-up (not timed).
        with torch.no_grad():
            for _ in range(n_warmup):
                m(x)

        # Synchronise CUDA before starting the timer.
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(n_iterations):
                t0 = time.perf_counter()
                m(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # seconds → ms

        lat_tensor = torch.tensor(latencies)
        mean_ms = float(lat_tensor.mean())
        std_ms = float(lat_tensor.std())

        results[dev_name] = {"mean_ms": mean_ms, "std_ms": std_ms}
        logger.info(
            "Benchmark [%s]  mean=%.2f ms  std=%.2f ms  (%d iters)",
            dev_name,
            mean_ms,
            std_ms,
            n_iterations,
        )

    return results


# ---------------------------------------------------------------------------
# CLI entry-point (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # Quick smoke test: create a model, run a dummy forward pass,
    # apply the unfreezer for epoch 0, and print the output shape.
    dummy_schedule: Dict[int, Any] = {
        0: ["fc"],
        3: ["layer4", "fc"],
        6: ["layer3", "layer4", "fc"],
        10: "all",
    }

    model = create_model(
        task_name="fish",
        num_classes=10,
        pretrained=False,
        dropout=0.3,
    )

    unfreezer = ProgressiveUnfreezer(model, dummy_schedule)
    unfreezer.step(epoch=0)

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    logger.info("Output shape: %s", logits.shape)

    feat_ext = model.get_feature_extractor("layer4")
    logger.info("Feature extractor type: %s", type(feat_ext).__name__)
