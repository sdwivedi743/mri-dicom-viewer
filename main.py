import base64
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom
import scipy.ndimage as ndimage
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(
    page_title="MRI Workstation",
    page_icon="🧠",
    layout="wide",
)

st.html("""
<style>
    .stApp {
        background: linear-gradient(135deg, #060913 0%, #0b1121 100%);
        color: #f1f5f9;
    }
    .stSidebar {
        background: rgba(15, 23, 42, 0.55);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .viewport-grid {
        display: grid;
        gap: 16px;
    }
    .viewport-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px;
    }
    .viewport-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 6px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .viewport-status {
        color: #94a3b8;
        font-size: 0.72rem;
        font-family: ui-monospace, monospace;
        margin-top: 6px;
    }
    canvas {
        width: 100%;
        height: auto;
        background: #000;
        border-radius: 8px;
        cursor: crosshair;
        display: block;
    }
</style>
""")

st.title("🧠 MRI Diagnostics Workstation")
st.write(
    "Upload a ZIP archive containing DICOM MRI series. "
    "Images are windowed from the original rescaled pixel values; no diagnostic measurement should be used without validation."
)

MAX_SERIES = 6
MAX_SLICES_PER_SERIES = 250


@dataclass
class Slice:
    instance_number: int
    position: tuple[float, float, float] | None
    pixel_array: np.ndarray
    window_center: float
    window_width: float


@dataclass
class Series:
    name: str
    slices: list[Slice] = field(default_factory=list)
    row_spacing: float = 1.0
    column_spacing: float = 1.0


def apply_ai_corrections(pixels: np.ndarray) -> np.ndarray:
    """
    Classical enhancement, not AI: bias-field correction, ringing suppression,
    and conservative Laplacian sharpening.
    """
    img = np.asarray(pixels, dtype=np.float32)

    finite = np.isfinite(img)
    if not finite.all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(np.min(img))
    shifted = img - min_value + 1.0

    log_img = np.log(shifted)
    sigma = max(1.0, max(img.shape) / 6.0)
    low_frequency = ndimage.gaussian_filter(log_img, sigma=sigma)
    bias_corrected = np.exp(log_img - low_frequency)

    smooth = ndimage.gaussian_filter(bias_corrected, sigma=1.2)
    high_frequency = bias_corrected - smooth
    suppressed = bias_corrected - 0.35 * high_frequency

    sharpened = suppressed - 0.20 * ndimage.laplace(suppressed)
    return np.clip(sharpened, 0.0, None).astype(np.float32)


def _safe_float(value, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _get_window_values(dcm, pixels: np.ndarray) -> tuple[float, float]:
    default_center = float(np.median(pixels))
    default_width = max(1.0, float(np.percentile(pixels, 99) - np.percentile(pixels, 1)))

    try:
        center = dcm.WindowCenter
        width = dcm.WindowWidth
    except AttributeError:
        return default_center, default_width

    if isinstance(center, (list, tuple, pydicom.multival.MultiValue)):
        center = center[0] if len(center) else default_center
    if isinstance(width, (list, tuple, pydicom.multival.MultiValue)):
        width = width[0] if len(width) else default_width

    center = _safe_float(center, default_center)
    width = _safe_float(width, default_width)
    return center, max(1.0, width)


def _slice_sort_key(slice_item: Slice) -> tuple:
    if slice_item.position is not None:
        return (0, slice_item.position[2], slice_item.position[1], slice_item.position[0])
    return (1, slice_item.instance_number, 0.0, 0.0)


@st.cache_data(show_spinner="Parsing DICOM series...")
def parse_dicom_zip(file_bytes: bytes) -> tuple[dict[str, Series], list[str], dict[str, str]]:
    """
    Parse a ZIP into series. Returns series, warnings, and patient metadata.
    Caching prevents repeated ZIP extraction and DICOM parsing on every rerun.
    """
    series: dict[str, Series] = {}
    warnings: list[str] = []
    patient = {
        "name": "Anonymous",
        "id": "N/A",
        "date": "N/A",
        "manufacturer": "N/A",
    }

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            path_parts = [part for part in info.filename.split("/") if part]
            if not path_parts or path_parts[0] == "__MACOSX__":
                continue

            try:
                with archive.open(info) as file_handle:
                    data = file_handle.read()

                dcm = pydicom.dcmread(io.BytesIO(data), force=True)
                if not hasattr(dcm, "pixel_array"):
                    continue

                pixels = dcm.pixel_array.astype(np.float32)

                slope = _safe_float(getattr(dcm, "RescaleSlope", 1.0), 1.0)
                intercept = _safe_float(getattr(dcm, "RescaleIntercept", 0.0), 0.0)
                pixels = pixels * slope + intercept

                if not np.isfinite(pixels).all():
                    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)

                series_name = str(
                    getattr(dcm, "SeriesDescription", "")
                    or getattr(dcm, "ProtocolName", "")
                    or f"Sequence {getattr(dcm, 'SeriesNumber', 'Unknown')}"
                ).strip()

                try:
                    instance_number = int(dcm.InstanceNumber)
                except (AttributeError, TypeError, ValueError):
                    instance_number = len(series.get(series_name, Series(series_name)).slices)

                position = None
                if hasattr(dcm, "ImagePositionPatient"):
                    try:
                        position = tuple(float(value) for value in dcm.ImagePositionPatient)
                    except (TypeError, ValueError):
                        position = None

                window_center, window_width = _get_window_values(dcm, pixels)

                slice_item = Slice(
                    instance_number=instance_number,
                    position=position,
                    pixel_array=pixels,
                    window_center=window_center,
                    window_width=window_width,
                )

                if series_name not in series:
                    spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
                    row_spacing = _safe_float(spacing[0], 1.0)
                    column_spacing = _safe_float(spacing[1], 1.0)
                    series[series_name] = Series(
                        name=series_name,
                        row_spacing=max(row_spacing, 1e-6),
                        column_spacing=max(column_spacing, 1e-6),
                    )

                series[series_name].slices.append(slice_item)

                if patient["name"] == "Anonymous" and hasattr(dcm, "PatientName"):
                    patient["name"] = str(dcm.PatientName) or "Anonymous"
                if patient["id"] == "N/A" and hasattr(dcm, "PatientID"):
                    patient["id"] = str(dcm.PatientID) or "N/A"
                if patient["date"] == "N/A" and hasattr(dcm, "StudyDate"):
                    patient["date"] = str(dcm.StudyDate) or "N/A"
                if patient["manufacturer"] == "N/A" and hasattr(dcm, "Manufacturer"):
                    patient["manufacturer"] = str(dcm.Manufacturer) or "N/A"

            except Exception as exc:
                warnings.append(f"{info.filename}: {type(exc).__name__}: {exc}")

    for item in series.values():
        item.slices.sort(key=_slice_sort_key)

    return series, warnings, patient


def encode_float32_array(array: np.ndarray) -> str:
    """Encode a little-endian float32 array for safe decoding in JavaScript."""
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def build_viewer_payload(
    selected_series: dict[str, Series],
    ai_mode: bool,
) -> tuple[list[dict], list[str]]:
    payload = []
    warnings: list[str] = []

    for name, series_item in selected_series.items():
        if len(series_item.slices) > MAX_SLICES_PER_SERIES:
            warnings.append(
                f"{name}: showing first {MAX_SLICES_PER_SERIES} of "
                f"{len(series_item.slices)} slices to limit browser memory use."
            )

        encoded_slices = []
        for slice_item in series_item.slices[:MAX_SLICES_PER_SERIES]:
            pixels = apply_ai_corrections(slice_item.pixel_array) if ai_mode else slice_item.pixel_array
            encoded_slices.append(
                {
                    "pixels": encode_float32_array(pixels),
                    "rows": int(pixels.shape[0]),
                    "cols": int(pixels.shape[1]),
                    "windowCenter": float(slice_item.window_center),
                    "windowWidth": float(slice_item.window_width),
                }
            )

        payload.append(
            {
                "title": name,
                "rowSpacing": float(series_item.row_spacing),
                "columnSpacing": float(series_item.column_spacing),
                "slices": encoded_slices,
            }
        )

    return payload, warnings


def render_viewer(payload: list[dict]) -> None:
    if not payload:
        st.warning("No series selected.")
        return

    columns = min(3, len(payload))
    rows = math.ceil(len(payload) / columns)

    panels_html = ""
    for index in range(len(payload)):
        panels_html += f"""
        <div class="viewport-panel">
            <div class="viewport-title" id="title-{index}"></div>
            <canvas id="canvas-{index}" width="512" height="512"></canvas>
            <div class="viewport-status" id="status-{index}"></div>
        </div>
        """

    # Escape "</" so a malicious DICOM string cannot terminate the JSON block.
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""
<div class="viewport-grid" style="grid-template-columns: repeat({columns}, minmax(0, 1fr));">
    {panels_html}
</div>

<script type="application/json" id="workstation-data">{safe_json}</script>
<script>
(() => {{
    const dataElement = document.getElementById("workstation-data");
    const workstationData = JSON.parse(dataElement.textContent);
    const panelCount = workstationData.length;
    const currentIndices = new Array(panelCount).fill(0);
    const windowCenters = [];
    const windowWidths = [];
    const caliperState = {{ active: false, panel: null, startX: 0, startY: 0, endX: 0, endY: 0 }};
    const dragState = {{ active: false, panel: null, lastX: 0, lastY: 0 }};

    workstationData.forEach((series, index) => {{
        const first = series.slices[0];
        windowCenters[index] = first ? first.windowCenter : 0;
        windowWidths[index] = first ? first.windowWidth : 1;
    }});

    function decodePixels(base64) {{
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{
            bytes[i] = binary.charCodeAt(i);
        }}
        return new Float32Array(bytes.buffer);
    }}

    function renderPanel(index) {{
        const series = workstationData[index];
        const sliceIndex = currentIndices[index];
        const slice = series.slices[sliceIndex];
        const canvas = document.getElementById("canvas-" + index);
        const context = canvas.getContext("2d");

        if (!series || !slice || !context) {{
            return;
        }}

        const titleElement = document.getElementById("title-" + index);
        const statusElement = document.getElementById("status-" + index);
        if (titleElement) {{
            titleElement.textContent = series.title;
        }}

        const pixels = decodePixels(slice.pixels);
        const imageData = context.createImageData(slice.cols, slice.rows);
        const output = imageData.data;

        const center = windowCenters[index];
        const width = Math.max(1, windowWidths[index]);
        const lower = center - width / 2;
        const upper = center + width / 2;
        const scale = 255 / width;

        for (let i = 0; i < pixels.length; i++) {{
            const value = Math.min(upper, Math.max(lower, pixels[i]));
            const gray = Math.max(0, Math.min(255, Math.round((value - lower) * scale)));
            const offset = i * 4;
            output[offset] = gray;
            output[offset + 1] = gray;
            output[offset + 2] = gray;
            output[offset + 3] = 255;
        }}

        canvas.width = slice.cols;
        canvas.height = slice.rows;
        context.putImageData(imageData, 0, 0);

        if (caliperState.active && caliperState.panel === index) {{
            context.strokeStyle = "#22d3ee";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(caliperState.startX, caliperState.startY);
            context.lineTo(caliperState.endX, caliperState.endY);
            context.stroke();

            const dxPixels = caliperState.endX - caliperState.startX;
            const dyPixels = caliperState.endY - caliperState.startY;
            const dxMm = dxPixels * series.columnSpacing;
            const dyMm = dyPixels * series.rowSpacing;
            const distanceMm = Math.sqrt(dxMm * dxMm + dyMm * dyMm);

            context.font = "bold 14px ui-monospace, monospace";
            context.fillStyle = "#22d3ee";
            context.shadowColor = "#000";
            context.shadowBlur = 4;
            context.fillText(distanceMm.toFixed(2) + " mm", caliperState.endX + 8, caliperState.endY - 8);
        }}

        if (statusElement) {{
            statusElement.textContent =
                "Slice " + (sliceIndex + 1) + "/" + series.slices.length +
                " | W " + Math.round(width) +
                " | L " + Math.round(center);
        }}
    }}

    function canvasCoordinates(event, canvas) {{
        const rect = canvas.getBoundingClientRect();
        return [
            (event.clientX - rect.left) * (canvas.width / rect.width),
            (event.clientY - rect.top) * (canvas.height / rect.height),
        ];
    }}

    workstationData.forEach((series, index) => {{
        const canvas = document.getElementById("canvas-" + index);
        if (!canvas) {{
            return;
        }}

        canvas.addEventListener("wheel", (event) => {{
            event.preventDefault();
            const direction = event.deltaY > 0 ? 1 : -1;
            const count = series.slices.length;
            currentIndices[index] = (currentIndices[index] + direction + count) % count;
            renderPanel(index);
        }}, {{ passive: false }});

        canvas.addEventListener("mousedown", (event) => {{
            const [x, y] = canvasCoordinates(event, canvas);

            if (event.shiftKey) {{
                caliperState.active = true;
                caliperState.panel = index;
                caliperState.startX = x;
                caliperState.startY = y;
                caliperState.endX = x;
                caliperState.endY = y;
            }} else {{
                dragState.active = true;
                dragState.panel = index;
                dragState.lastX = event.clientX;
                dragState.lastY = event.clientY;
                caliperState.active = false;
            }}
        }});
    }});

    window.addEventListener("mousemove", (event) => {{
        if (caliperState.active && caliperState.panel !== null) {{
            const canvas = document.getElementById("canvas-" + caliperState.panel);
            if (!canvas) return;
            const [x, y] = canvasCoordinates(event, canvas);
            caliperState.endX = x;
            caliperState.endY = y;
            renderPanel(caliperState.panel);
        }} else if (dragState.active && dragState.panel !== null) {{
            const deltaX = event.clientX - dragState.lastX;
            const deltaY = event.clientY - dragState.lastY;
            dragState.lastX = event.clientX;
            dragState.lastY = event.clientY;

            windowWidths[dragState.panel] = Math.max(1, windowWidths[dragState.panel] + deltaX * 2);
            windowCenters[dragState.panel] = windowCenters[dragState.panel] - deltaY * 2;
            renderPanel(dragState.panel);
        }}
    }});

    window.addEventListener("mouseup", () => {{
        dragState.active = false;
    }});

    for (let index = 0; index < panelCount; index++) {{
        renderPanel(index);
    }}
}})();
</script>
"""

    components.html(html, height=max(420, 430 * rows), scrolling=False)


uploaded_file = st.sidebar.file_uploader("Upload Patient ZIP", type=["zip"])
ai_mode = st.sidebar.toggle(
    "Enhancement mode",
    value=False,
    help="Applies classical bias-field, ringing-suppression, and sharpening filters. This is not AI super-resolution.",
)

if uploaded_file is None:
    st.info("Upload a DICOM ZIP archive to begin.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    series_dict, parse_warnings, patient = parse_dicom_zip(file_bytes)
except Exception as exc:
    st.error(f"Could not read the uploaded ZIP: {type(exc).__name__}: {exc}")
    st.stop()

if not series_dict:
    st.error("No readable DICOM images with pixel data were found in this archive.")
    if parse_warnings:
        with st.expander("Parsing warnings"):
            st.write("\n".join(parse_warnings))
    st.stop()

series_names = sorted(series_dict.keys())
selected_names = st.sidebar.multiselect(
    f"Select sequences (up to {MAX_SERIES})",
    options=series_names,
    default=series_names[: min(3, len(series_names))],
)
selected_names = selected_names[:MAX_SERIES]

st.sidebar.markdown("### Patient")
st.sidebar.text(f"Name: {patient['name']}")
st.sidebar.text(f"ID: {patient['id']}")
st.sidebar.text(f"Study date: {patient['date']}")
st.sidebar.text(f"Scanner: {patient['manufacturer']}")

if parse_warnings:
    with st.sidebar.expander(f"Skipped files ({len(parse_warnings)})"):
        st.write("\n".join(parse_warnings))

if not selected_names:
    st.warning("Select at least one sequence to display.")
    st.stop()

selected_series = {name: series_dict[name] for name in selected_names}
payload, payload_warnings = build_viewer_payload(selected_series, ai_mode)

for warning in payload_warnings:
    st.warning(warning)

render_viewer(payload)

st.caption(
    "Controls: scroll to change slice; left-drag to adjust window/level; "
    "Shift+drag to draw a caliper. Measurements use each series’ PixelSpacing "
    "but do not account for oblique orientation or voxel spacing and are not for clinical use."
)