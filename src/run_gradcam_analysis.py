"""Run Grad-CAM and Error Analysis for trained models.

Generates:
    - Grad-CAM heatmap samples for correct and incorrect predictions
    - Error analysis diagnostic report (outputs/error_analysis_{task}.txt)
    - Error analysis grid figure (outputs/figures/{task}_error_analysis.png)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import get_dataloaders
from src.gradcam import generate_error_analysis, generate_gradcam_samples
from src.model import create_model, load_checkpoint
from src.utils import get_device, get_effective_config, load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps and error analysis.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default="fish", choices=["fish", "coral"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = get_effective_config(cfg)
    device = get_device(cfg)

    # Load data
    _, _, test_loader, class_names = get_dataloaders(cfg, task=args.task)

    # Load model
    model = create_model(task_name=args.task, num_classes=len(class_names), cfg=cfg)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    out_gradcam_dir = Path(cfg.paths.gradcam_dir) / args.task
    out_gradcam_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Grad-CAM samples for task: %s...", args.task)
    samples = generate_gradcam_samples(
        model=model,
        dataloader=test_loader,
        cfg=cfg,
        class_names=class_names,
        num_samples=args.num_samples,
        output_dir=str(out_gradcam_dir),
    )

    # Error analysis
    error_summary = generate_error_analysis(
        results=samples,
        class_names=class_names,
        output_dir=cfg.paths.figures_dir,
    )
    logger.info("Error analysis completed: %s", error_summary)


if __name__ == "__main__":
    main()

