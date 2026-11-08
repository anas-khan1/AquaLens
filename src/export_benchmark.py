"""Export trained PyTorch models to ONNX and benchmark inference latency.

Generates:
    - outputs/checkpoints/coral_model.onnx
    - outputs/checkpoints/fish_model.onnx
    - outputs/benchmark_results.json
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model import benchmark_inference, create_model, load_checkpoint
from src.utils import get_device, get_effective_config, load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = load_config("config.yaml")
    cfg = get_effective_config(cfg)
    device = get_device(cfg)

    results = {}
    checkpoints = [
        ("coral", "outputs/checkpoints/coral_best.pth", 2),
        ("fish", "outputs/checkpoints/fish_best.pth", 5),
    ]

    image_size = cfg.get("_effective_image_size", 128)

    for task, ckpt_path, num_classes in checkpoints:
        if not Path(ckpt_path).exists():
            logger.warning("Checkpoint %s does not exist, skipping.", ckpt_path)
            continue

        model = create_model(task_name=task, num_classes=num_classes, cfg=cfg)
        load_checkpoint(ckpt_path, model=model, map_location=device)
        model.eval()

        # Benchmark PyTorch Inference Latency
        bench = benchmark_inference(
            model=model,
            input_shape=(1, 3, image_size, image_size),
            n_warmup=10,
            n_iterations=50,
        )
        results[f"{task}_pytorch"] = bench
        cpu_metrics = bench.get("cpu", {})
        mean_ms = cpu_metrics.get("mean_ms", 0.0)
        std_ms = cpu_metrics.get("std_ms", 0.0)
        fps = 1000.0 / max(mean_ms, 1e-4)
        logger.info(
            "%s PyTorch Latency (CPU): %.2f ± %.2f ms (throughput: %.1f FPS)",
            task.upper(),
            mean_ms,
            std_ms,
            fps,
        )

        # ONNX export
        onnx_path = Path("outputs/checkpoints") / f"{task}_model.onnx"
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                opset_version=14,
            )
            logger.info("Successfully exported %s to ONNX: %s", task, onnx_path)
            results[f"{task}_onnx_exported"] = str(onnx_path)
        except Exception as exc:
            logger.warning("ONNX export for %s failed: %s", task, exc)

    # Save benchmark
    out_json = Path("outputs/benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark results saved to %s", out_json)


if __name__ == "__main__":
    main()
