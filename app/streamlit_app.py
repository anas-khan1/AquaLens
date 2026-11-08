"""AquaLens — Underwater Marine Species & Coral Reef Health Visual Assessment.

Multi-Page Interactive Streamlit Application:
    Page 1: 🐟 Marine Fish Species Recognition
    Page 2: 🪸 Coral Reef Bleaching & Health Assessment
    Page 3: ⚙️ Optical Enhancement Studio (DCP + CLAHE live sandbox)
    Page 4: 📖 Problem Statement & Colab Training Guide
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path regardless of execution directory
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Safe writable matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Import internal AquaLens modules
from src.gradcam import GradCAM, overlay_heatmap
from src.model import MarineClassifier, create_model
from src.preprocessing import (
    apply_clahe,
    compute_image_quality_metrics,
    dark_channel_prior_correction,
    preprocess_image,
)
from src.severity import SeverityAssessor, classify_severity, compute_reef_health_index
from src.utils import get_device, get_effective_config, load_config

# Set page configuration
st.set_page_config(
    page_title="AquaLens — Marine Vision AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .brand-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0277bd;
        margin-bottom: 0.1rem;
    }
    .brand-sub {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 1.5rem;
    }
    .status-box-pending {
        background-color: #fff9db;
        border: 1px solid #ffe066;
        border-left: 5px solid #f59f00;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .status-box-ready {
        background-color: #ebfbee;
        border: 1px solid #b2f2bb;
        border-left: 5px solid #37b24d;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .badge-healthy {
        background-color: #28a745; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-mild {
        background-color: #ffc107; color: black; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-moderate {
        background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    .badge-severe {
        background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app_config() -> Any:
    """Load Master YAML Configuration."""
    cfg = load_config("config.yaml")
    return get_effective_config(cfg)


@st.cache_resource
def load_task_model(
    task: str, ckpt_path: str, num_classes: int = 5
) -> Tuple[Optional[MarineClassifier], List[str], str]:
    """Load PyTorch checkpoint safely, searching candidate directories."""
    candidates = [
        Path(ckpt_path),
        _PROJECT_ROOT / ckpt_path,
        _PROJECT_ROOT / "outputs" / "checkpoints" / f"{task}_best.pth",
        _PROJECT_ROOT / "outputs" / "checkpoints" / f"best_{task}.pth",
        Path("outputs") / "checkpoints" / f"{task}_best.pth",
        Path("outputs") / "checkpoints" / f"best_{task}.pth",
    ]
    found_path: Optional[Path] = None
    for c in candidates:
        if c.exists() and c.is_file() and c.stat().st_size > 1024:
            found_path = c
            break

    if found_path is None:
        return None, [], "No checkpoint found on disk."

    device = torch.device("cpu")
    try:
        ckpt = torch.load(found_path, map_location=device)
        class_names = ckpt.get("class_names", [])
        if not class_names:
            class_names = (
                ["bleached", "healthy"]
                if task == "coral"
                else [f"Species_{i}" for i in range(num_classes)]
            )

        model = create_model(task, num_classes=len(class_names), pretrained=False).to(device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model, class_names, str(found_path)
    except Exception as exc:
        return None, [], f"Error loading checkpoint ({exc})"


def enhance_frame(
    bgr: np.ndarray,
    enable_dcp: bool = True,
    dcp_omega: float = 0.95,
    dcp_patch: int = 15,
    enable_clahe: bool = True,
    clahe_clip: float = 2.5,
    clahe_tile: int = 8,
    target_size: int = 224,
) -> np.ndarray:
    """Apply native OpenCV/NumPy optical restoration (DCP + CLAHE)."""
    res = bgr.copy()
    if enable_dcp:
        res = dark_channel_prior_correction(
            res,
            patch_size=dcp_patch,
            omega=dcp_omega,
            t_min=0.1,
            guided_radius=30,
            guided_eps=1e-3,
        )
    if enable_clahe:
        res = apply_clahe(
            res,
            clip_limit=clahe_clip,
            tile_grid_size=(clahe_tile, clahe_tile),
        )
    res = cv2.resize(res, (target_size, target_size))
    return res


def render_sidebar(cfg: Any) -> Tuple[str, Dict[str, Any]]:
    """Render global sidebar navigation and shared optical controls."""
    st.sidebar.markdown("## 🌊 **AquaLens AI**")
    st.sidebar.caption("Underwater Marine Species & Coral Health")

    page = st.sidebar.radio(
        "Select Application View:",
        options=[
            "🐟 Marine Fish Species Recognition",
            "🪸 Coral Reef Bleaching & Health",
            "⚙️ Optical Enhancement Studio (DCP + CLAHE)",
            "📖 Problem Statement & Colab Guide",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Optical Parameters")
    enable_dcp = st.sidebar.checkbox("Enable Dark Channel Prior (DCP)", value=True)
    dcp_omega = st.sidebar.slider("DCP Dehaze Strength (ω)", 0.2, 1.0, 0.95, 0.05)
    dcp_patch = st.sidebar.slider("DCP Patch Size", 5, 25, 15, 2)

    enable_clahe = st.sidebar.checkbox("Enable L-channel CLAHE", value=True)
    clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 6.0, 2.5, 0.5)
    clahe_tile = st.sidebar.slider("CLAHE Grid Size", 4, 16, 8, 2)

    st.sidebar.markdown("---")
    st.sidebar.caption("AquaLens v2.0 • ResNet-34 • Grad-CAM")

    optical_settings = {
        "enable_dcp": enable_dcp,
        "dcp_omega": dcp_omega,
        "dcp_patch": dcp_patch,
        "enable_clahe": enable_clahe,
        "clahe_clip": clahe_clip,
        "clahe_tile": clahe_tile,
    }
    return page, optical_settings


# =============================================================================
# PAGE 1: Marine Fish Species Recognition
# =============================================================================
def render_fish_page(cfg: Any, optical_settings: Dict[str, Any]) -> None:
    st.markdown('<div class="brand-header">🐟 Marine Fish Species Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Classify marine teleost fish species despite optical turbidity and underwater color attenuation with Grad-CAM explainability.</div>',
        unsafe_allow_html=True,
    )

    # Check model availability
    fish_ckpt_default = "outputs/checkpoints/fish_best.pth"
    model, class_names, status_msg = load_task_model("fish", fish_ckpt_default, num_classes=5)

    if model is not None:
        st.markdown(
            f'<div class="status-box-ready">✅ <b>Fish Model Active:</b> Loaded weights from <code>{status_msg}</code> ({len(class_names)} species recognized).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-box-pending">
                ⚙️ <b>Fish Model Status: Ready for Training.</b><br/>
                No active <code>fish_best.pth</code> was found in <code>outputs/checkpoints/</code>.
                You can still test <b>optical dehazing & CLAHE enhancement in real time</b> below!
                Once you train your ResNet-34 on Google Colab with your new real sea animal dataset,
                place the <code>fish_best.pth</code> file into <code>outputs/checkpoints/</code> and species classification + Grad-CAM will activate automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_single, tab_batch = st.tabs(["🔬 Single Fish Frame Analysis", "📁 Survey Batch Processor"])

    # --- TAB 1: Single Image ---
    with tab_single:
        col_upload, col_preview = st.columns([1, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload an underwater fish photograph (JPG / PNG):",
                type=["jpg", "jpeg", "png"],
                key="fish_uploader",
            )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Applying Optical Dark Channel Prior & CIELAB CLAHE..."):
                enhanced_bgr = enhance_frame(
                    orig_bgr,
                    enable_dcp=optical_settings["enable_dcp"],
                    dcp_omega=optical_settings["dcp_omega"],
                    dcp_patch=optical_settings["dcp_patch"],
                    enable_clahe=optical_settings["enable_clahe"],
                    clahe_clip=optical_settings["clahe_clip"],
                    clahe_tile=optical_settings["clahe_tile"],
                    target_size=224,
                )

            # Display side-by-side
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Raw Degraded Image")
                st.image(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                q_raw = compute_image_quality_metrics(orig_bgr)
                st.caption(f"Entropy: {q_raw['entropy']:.2f} bits | Contrast: {q_raw['contrast']:.3f} | Colorfulness: {q_raw['colorfulness']:.1f}")

            with c2:
                st.subheader("Restored (DCP + CLAHE)")
                st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                q_enh = compute_image_quality_metrics(enhanced_bgr)
                st.caption(f"Entropy: {q_enh['entropy']:.2f} bits (Δ {q_enh['entropy'] - q_raw['entropy']:+.2f}) | "
                           f"Contrast: {q_enh['contrast']:.3f} | "
                           f"Colorfulness: {q_enh['colorfulness']:.1f}")

            st.markdown("---")

            # Classification & Explainability
            if model is not None:
                if st.button("🚀 Identify Fish Species & Generate Grad-CAM", type="primary", key="btn_fish_infer"):
                    with st.spinner("Computing ResNet-34 Feature Activations & Grad-CAM..."):
                        mean = np.array(cfg.preprocessing.normalize.mean, dtype=np.float32)
                        std = np.array(cfg.preprocessing.normalize.std, dtype=np.float32)
                        rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        norm = (rgb - mean) / std
                        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(tensor)
                            probs = F.softmax(logits, dim=1).squeeze(0).numpy()

                        top_indices = np.argsort(probs)[::-1][: min(5, len(class_names))]
                        top_species = class_names[top_indices[0]]
                        top_conf = float(probs[top_indices[0]])

                        # Grad-CAM
                        gradcam = GradCAM(model, target_layer_name="layer4")
                        cam_mask = gradcam.generate(tensor, target_class=top_indices[0])
                        gradcam.remove_hooks()
                        heatmap_overlay = overlay_heatmap(enhanced_bgr, cam_mask, alpha=0.5)

                        st.success(f"**Identified Species:** `{top_species}` (Confidence: {top_conf:.1%})")

                        res_c1, res_c2 = st.columns([1, 1])
                        with res_c1:
                            st.subheader("Top Predictions")
                            for idx in top_indices:
                                sp_name = class_names[idx]
                                sp_prob = float(probs[idx])
                                st.write(f"**{sp_name}** ({sp_prob:.1%})")
                                st.progress(sp_prob)

                        with res_c2:
                            st.subheader("Grad-CAM Morphological Focus")
                            st.image(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.caption("Warm regions (red/yellow) indicate anatomical traits (fin structure, stripes) used by ResNet-34.")
            else:
                st.info("💡 **Model Inference Notice:** Once your newly trained `fish_best.pth` model is placed in `outputs/checkpoints/`, full species classification and morphological Grad-CAM heatmaps will appear here.")

    # --- TAB 2: Batch Processing ---
    with tab_batch:
        st.subheader("Batch Survey Video / Image Analysis")
        st.write("Upload multiple underwater frames from a dive survey to classify species population distribution.")
        batch_files = st.file_uploader(
            "Select multiple fish images:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="fish_batch_uploader",
        )
        if batch_files:
            st.info(f"{len(batch_files)} images staged for batch inference.")
            if model is not None:
                if st.button("⚡ Process All Survey Frames", key="btn_fish_batch"):
                    records = []
                    prog = st.progress(0)
                    for i, bf in enumerate(batch_files):
                        b_bytes = np.asarray(bytearray(bf.read()), dtype=np.uint8)
                        b_bgr = cv2.imdecode(b_bytes, cv2.IMREAD_COLOR)
                        if b_bgr is None:
                            continue
                        enh = enhance_frame(b_bgr, **optical_settings, target_size=224)
                        mean = np.array(cfg.preprocessing.normalize.mean, dtype=np.float32)
                        std = np.array(cfg.preprocessing.normalize.std, dtype=np.float32)
                        rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        norm = (rgb - mean) / std
                        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(tensor)
                            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                        p_idx = int(np.argmax(probs))
                        records.append({
                            "Filename": bf.name,
                            "Predicted Species": class_names[p_idx],
                            "Confidence": f"{probs[p_idx]:.1%}",
                        })
                        prog.progress((i + 1) / len(batch_files))
                    df_res = pd.DataFrame(records)
                    st.dataframe(df_res, use_container_width=True)
                    st.bar_chart(df_res["Predicted Species"].value_counts())
            else:
                st.warning("Model checkpoint required for batch species inference. Please train and place `fish_best.pth`.")


# =============================================================================
# PAGE 2: Coral Reef Bleaching & Health Assessment
# =============================================================================
def render_coral_page(cfg: Any, optical_settings: Dict[str, Any]) -> None:
    st.markdown('<div class="brand-header">🪸 Coral Reef Bleaching & Health Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Detect coral reef bleaching, assess 4-tier ecological severity (Healthy, Mild, Moderate, Severe), and calculate the Reef Health Index (RHI).</div>',
        unsafe_allow_html=True,
    )

    coral_ckpt_default = "outputs/checkpoints/coral_best.pth"
    model, class_names, status_msg = load_task_model("coral", coral_ckpt_default, num_classes=2)

    if model is not None:
        st.markdown(
            f'<div class="status-box-ready">✅ <b>Coral Model Active:</b> Loaded weights from <code>{status_msg}</code> (Binary health classification).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-box-pending">
                ⚙️ <b>Coral Model Status: Ready for Training.</b><br/>
                No active <code>coral_best.pth</code> was found in <code>outputs/checkpoints/</code>.
                You can test the <b>optical dehazing and CIELAB CLAHE algorithms</b> below!
                Once you train your ResNet-34 on coral bleaching images on Google Colab,
                place <code>coral_best.pth</code> in <code>outputs/checkpoints/</code> to activate automated bleaching severity scoring and Grad-CAM localization.
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_single, tab_batch = st.tabs(["🔬 Benthic Coral Frame Analysis", "📊 Transect Survey & Reef Health Index (RHI)"])

    with tab_single:
        uploaded_coral = st.file_uploader(
            "Upload an underwater coral colony or benthic transect image (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            key="coral_uploader",
        )

        if uploaded_coral is not None:
            file_bytes = np.asarray(bytearray(uploaded_coral.read()), dtype=np.uint8)
            orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Applying Optical Restoration Pipeline..."):
                enhanced_bgr = enhance_frame(
                    orig_bgr,
                    enable_dcp=optical_settings["enable_dcp"],
                    dcp_omega=optical_settings["dcp_omega"],
                    dcp_patch=optical_settings["dcp_patch"],
                    enable_clahe=optical_settings["enable_clahe"],
                    clahe_clip=optical_settings["clahe_clip"],
                    clahe_tile=optical_settings["clahe_tile"],
                    target_size=224,
                )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Raw Degraded Substrate")
                st.image(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                q_raw = compute_image_quality_metrics(orig_bgr)
                st.caption(f"Entropy: {q_raw['entropy']:.2f} bits | Contrast: {q_raw['contrast']:.3f} | Colorfulness: {q_raw['colorfulness']:.1f}")

            with c2:
                st.subheader("Restored Coral Benthos (DCP + CLAHE)")
                st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                q_enh = compute_image_quality_metrics(enhanced_bgr)
                st.caption(f"Entropy: {q_enh['entropy']:.2f} bits (Δ {q_enh['entropy'] - q_raw['entropy']:+.2f}) | "
                           f"Contrast: {q_enh['contrast']:.3f} | "
                           f"Colorfulness: {q_enh['colorfulness']:.1f}")

            st.markdown("---")

            if model is not None:
                if st.button("🚀 Assess Coral Health & Bleaching Severity", type="primary", key="btn_coral_infer"):
                    with st.spinner("Assessing bleaching severity via Grad-CAM & HSV Pale-Pixel Heuristic..."):
                        assessor = SeverityAssessor(model, cfg)
                        assessment = assessor.assess(enhanced_bgr)

                        severity = assessment["severity"]
                        bleach_score = assessment["bleaching_score"]
                        confidence = assessment["confidence"]
                        cam_mask = assessment["cam_mask"]
                        overlay = assessment["overlay"]

                        # Severity badge
                        badge_class = {
                            "Healthy": "badge-healthy",
                            "Mild Bleaching": "badge-mild",
                            "Moderate Bleaching": "badge-moderate",
                            "Severe Bleaching": "badge-severe",
                        }.get(severity, "badge-healthy")

                        st.markdown(
                            f"### Result: <span class='{badge_class}'>{severity.upper()}</span> &nbsp; (Bleaching Score: <b>{bleach_score:.1%}</b>)",
                            unsafe_allow_html=True,
                        )

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.subheader("Ecological Diagnosis")
                            st.metric("Primary Prediction", assessment["prediction"].capitalize())
                            st.metric("Model Confidence", f"{confidence:.1%}")
                            st.metric("Bleached Coral Fraction", f"{bleach_score:.1%}")
                            st.caption(
                                "Severity Tiers: Healthy (<15%), Mild (15-35%), Moderate (35-60%), Severe (≥60%)."
                            )

                        with col_m2:
                            st.subheader("Grad-CAM Bleaching Localization")
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.caption("Visualizes bleached corallite heads and calcium carbonate skeletal exposure.")
            else:
                st.info("💡 **Model Inference Notice:** Once your newly trained `coral_best.pth` model is placed in `outputs/checkpoints/`, automated bleaching severity scoring and Grad-CAM heatmaps will appear here.")

    with tab_batch:
        st.subheader("Transect Survey Assessment & Reef Health Index (RHI)")
        st.write("Upload a transect survey batch of coral images to calculate ecosystem-level Reef Health Index (RHI).")
        coral_batch_files = st.file_uploader(
            "Upload multiple coral survey frames:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="coral_batch_uploader",
        )
        if coral_batch_files and model is not None:
            if st.button("⚡ Calculate Reef Health Index Across Batch", key="btn_coral_batch"):
                assessor = SeverityAssessor(model, cfg)
                records = []
                prog = st.progress(0)
                for i, bf in enumerate(coral_batch_files):
                    b_bytes = np.asarray(bytearray(bf.read()), dtype=np.uint8)
                    b_bgr = cv2.imdecode(b_bytes, cv2.IMREAD_COLOR)
                    if b_bgr is None:
                        continue
                    enh = enhance_frame(b_bgr, **optical_settings, target_size=224)
                    res = assessor.assess(enh)
                    records.append({
                        "Filename": bf.name,
                        "Diagnosis": res["prediction"],
                        "Severity": res["severity"],
                        "Bleaching Score": f"{res['bleaching_score']:.1%}",
                        "raw_score": res["bleaching_score"],
                    })
                    prog.progress((i + 1) / len(coral_batch_files))

                df_coral = pd.DataFrame(records)
                scores = [r["raw_score"] for r in records]
                rhi = compute_reef_health_index(scores)

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Survey Frames", len(df_coral))
                m2.metric("Reef Health Index (RHI)", f"{rhi:.3f}", help="Scale: 0.0 (Dead) to 1.0 (Pristine)")
                m3.metric("Healthy Colony Ratio", f"{(df_coral['Severity'] == 'Healthy').mean():.1%}")

                st.subheader("Bleaching Severity Distribution")
                st.bar_chart(df_coral["Severity"].value_counts())
                st.dataframe(df_coral.drop(columns=["raw_score"]), use_container_width=True)


# =============================================================================
# PAGE 3: Optical Enhancement Studio
# =============================================================================
def render_studio_page(cfg: Any, optical_settings: Dict[str, Any]) -> None:
    st.markdown('<div class="brand-header">⚙️ Optical Enhancement Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Test and fine-tune native Dark Channel Prior dehazing, Guided Filter transmission refinement, and CIELAB CLAHE parameters with live quality metrics.</div>',
        unsafe_allow_html=True,
    )

    studio_file = st.file_uploader(
        "Upload any underwater image to test optical dehazing & color restoration:",
        type=["jpg", "jpeg", "png"],
        key="studio_uploader",
    )

    if studio_file is not None:
        file_bytes = np.asarray(bytearray(studio_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 4-Way Comparison Grid
        with st.spinner("Generating 4-way optical enhancement comparison..."):
            dcp_only = dark_channel_prior_correction(
                bgr.copy(),
                patch_size=optical_settings["dcp_patch"],
                omega=optical_settings["dcp_omega"],
            )
            clahe_only = apply_clahe(
                bgr.copy(),
                clip_limit=optical_settings["clahe_clip"],
                tile_grid_size=(optical_settings["clahe_tile"], optical_settings["clahe_tile"]),
            )
            full_enh = enhance_frame(bgr.copy(), **optical_settings, target_size=bgr.shape[1])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Raw Underwater Image")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            q1 = compute_image_quality_metrics(bgr)
            st.caption(f"Entropy: {q1['entropy']:.2f} | Contrast: {q1['contrast']:.3f} | Colorfulness: {q1['colorfulness']:.1f}")

        with col2:
            st.subheader("2. CLAHE Only (L-Channel)")
            st.image(cv2.cvtColor(clahe_only, cv2.COLOR_BGR2RGB), use_container_width=True)
            q2 = compute_image_quality_metrics(clahe_only)
            st.caption(f"Entropy: {q2['entropy']:.2f} | Contrast: {q2['contrast']:.3f} | Colorfulness: {q2['colorfulness']:.1f}")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. Dark Channel Prior Only (DCP)")
            st.image(cv2.cvtColor(dcp_only, cv2.COLOR_BGR2RGB), use_container_width=True)
            q3 = compute_image_quality_metrics(dcp_only)
            st.caption(f"Entropy: {q3['entropy']:.2f} | Contrast: {q3['contrast']:.3f} | Colorfulness: {q3['colorfulness']:.1f}")

        with col4:
            st.subheader("4. Full Proposed Pipeline (DCP + CLAHE)")
            st.image(cv2.cvtColor(full_enh, cv2.COLOR_BGR2RGB), use_container_width=True)
            q4 = compute_image_quality_metrics(full_enh)
            st.caption(f"Entropy: {q4['entropy']:.2f} | Contrast: {q4['contrast']:.3f} | Colorfulness: {q4['colorfulness']:.1f}")


# =============================================================================
# PAGE 4: Problem Statement & Colab Guide
# =============================================================================
def render_guide_page() -> None:
    st.markdown('<div class="brand-header">📖 Problem Statement & Colab Training Guide</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🎯 Capstone Problem Statement")
    st.markdown(
        """
        > **Underwater Marine Species and Coral Reef Health Visual Assessment**
        > 
        > *Construct an automated underwater computer vision model to detect fish species and identify coral reef bleaching severity despite severe optical water degradations. Students will implement CLAHE and optical dark channel color correction before feeding images into a fine-tuned ResNet-34 classifier on the free Kaggle Marine Species dataset, visualizing bleaching heatmaps via Grad-CAM. 100% free to implement using PyTorch on free Google Colab or laptop CPU.*
        """
    )

    st.subheader("🚀 How to Train Your New Models on Free Google Colab (T4 GPU)")
    st.markdown(
        """
        Since your new sea creature dataset is large, training on your local laptop CPU would take hours.
        Google Colab gives you a **free NVIDIA T4 GPU (16 GB VRAM)** that trains ResNet-34 in **under 5 minutes**!

        ### Step-by-Step Instructions:
        1. Open [Google Colab](https://colab.research.google.com).
        2. Click **Upload** and choose `AquaLens/notebooks/Colab_Run_All.ipynb` from this project.
        3. Go to **Runtime** → **Change runtime type** → select **T4 GPU** (Free).
        4. Upload your dataset folder or ZIP to the Colab files panel.
        5. Run the cells in the notebook. It runs:
           ```bash
           # To train your new Fish model:
           python src/train.py --task fish --config config.yaml

           # To train your Coral model later:
           python src/train.py --task coral --config config.yaml
           ```
        6. Download the resulting `fish_best.pth` (or `coral_best.pth`) from `outputs/checkpoints/` in Colab.
        7. Place it into your local `AquaLens/outputs/checkpoints/` folder.
        8. Refresh this Streamlit app and your model is live!
        """
    )


# =============================================================================
# Main Router
# =============================================================================
def main() -> None:
    cfg = load_app_config()
    page, optical_settings = render_sidebar(cfg)

    if page == "🐟 Marine Fish Species Recognition":
        render_fish_page(cfg, optical_settings)
    elif page == "🪸 Coral Reef Bleaching & Health":
        render_coral_page(cfg, optical_settings)
    elif page == "⚙️ Optical Enhancement Studio (DCP + CLAHE)":
        render_studio_page(cfg, optical_settings)
    elif page == "📖 Problem Statement & Colab Guide":
        render_guide_page()


if __name__ == "__main__":
    main()
