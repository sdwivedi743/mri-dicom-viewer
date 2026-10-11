import io
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from enhance import ai_enhance, classical_enhance
from batch import enhance_series
from export_series import export_enhanced_series
from evaluate import evaluate_images


st.set_page_config(
    page_title="MRI DICOM Enhancement Viewer",
    layout="wide",
)

st.title("MRI DICOM Enhancement Viewer")
st.caption(
    "Research use only. AI enhancement is post-processing "
    "and does not replace diagnostic review."
)

uploaded = st.file_uploader(
    "Upload a DICOM ZIP or select a DICOM folder",
    type=["zip"],
    accept_multiple_files="directory",
)

weights_path = st.text_input(
    "Trained model weights (.pt), optional",
    "enhancement_unet.pt",
)

mode = st.radio(
    "Enhancement mode",
    [
        "AI enhancement",
        "Classical enhancement",
    ],
    horizontal=True,
)


def read_mri_datasets(files):
    datasets = []

    for uploaded_file in files:
        try:
            ds = pydicom.dcmread(
                io.BytesIO(uploaded_file.getvalue())
            )

            if getattr(ds, "Modality", "") == "MR":
                datasets.append(
                    (uploaded_file.name, ds)
                )

        except Exception:
            continue

    return datasets


if uploaded:
    files = (
        uploaded
        if isinstance(uploaded, list)
        else [uploaded]
    )

    datasets = read_mri_datasets(files)

    if not datasets:
        st.error(
            "No readable MRI DICOM files found."
        )
        st.stop()

    datasets.sort(
        key=lambda item: int(
            getattr(item[1], "InstanceNumber", 0) or 0
        )
    )

    selected_mode = (
        "ai"
        if mode.startswith("AI")
        else "classical"
    )

    results = enhance_series(
        datasets,
        weights_path=weights_path,
        mode=selected_mode,
    )

    middle_index = len(results) // 2
    name, ds, original, enhanced, info = results[
        middle_index
    ]

    psnr = peak_signal_noise_ratio(
        original,
        enhanced,
        data_range=1,
    )

    ssim = structural_similarity(
        original,
        enhanced,
        data_range=1,
    )

    st.success(
        f"Loaded {len(datasets)} MRI DICOM slices."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original",
            use_container_width=True,
        )

    with col2:
        st.image(
            enhanced,
            caption=f"Enhanced — {mode}",
            use_container_width=True,
        )

    with col3:
        st.image(
            np.abs(enhanced - original),
            caption="Difference",
            use_container_width=True,
        )

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "PSNR",
            f"{psnr:.2f} dB",
        )

    with metric2:
        st.metric(
            "SSIM",
            f"{ssim:.4f}",
        )

    st.json(info)

    if st.button(
        "Export entire enhanced series as DICOM"
    ):
        exported = export_enhanced_series(
            results,
            output_dir="enhanced_output",
            mode=selected_mode,
        )

        st.success(
            f"Exported {len(exported)} DICOM files."
        )

else:
    st.info(
        "Upload a .zip containing DICOM files, or choose "
        "an uncompressed DICOM folder."
    )