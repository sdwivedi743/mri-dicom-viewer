import base64
import io
import json
import math
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom
import scipy.ndimage as ndimage
import streamlit as st
import streamlit.components.v1 as components
from pydicom.errors import InvalidDicomError

st.set_page_config(page_title="MRI Workstation", page_icon="🧠", layout="wide")

st.title("🧠 MRI Diagnostics Workstation")
st.write("Upload a ZIP archive containing DICOM MRI images.")

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


def safe_float(value, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def get_window_values(dcm, pixels: np.ndarray) -> tuple[float, float]:
    default_center = float(np.median(pixels))
    default_width = max(
        1.0,
        float(np.percentile(pixels, 99) - np.percentile(pixels, 1)),
    )

    center = getattr(dcm, "WindowCenter", default_center)
    width = getattr(dcm, "WindowWidth", default_width)

    if isinstance(center, (list, tuple, pydicom.multival.MultiValue)):
        center = center[0] if len(center) else default_center
    if isinstance(width, (list, tuple, pydicom.multival.MultiValue)):
        width = width[0] if len(width) else default_width

    center = safe_float(center, default_center)
    width = safe_float(width, default_width)
    return center, max(1.0, width)


def read_pixel_array(dcm) -> np.ndarray:
    """
    Read pixel data, decompressing supported compressed transfer syntaxes.
    """
    transfer_syntax = getattr(
        getattr(dcm, "file_meta", None),
        "TransferSyntaxUID",
        None,
    )

    compressed_prefixes = (
        "1.2.840.10008.1.2.4.5",   # JPEG lossy
        "1.2.840.10008.1.2.4.50",  # JPEG lossy
        "1.2.840.10008.1.2.4.51",  # JPEG extended
        "1.2.840.10008.1.2.4.57",  # JPEG lossless
        "1.2.840.10008.1.2.4.70",  # JPEG lossless
        "1.2.840.10008.1.2.4.80",  # JPEG-LS
        "1.2.840.10008.1.2.4.90",  # JPEG 2000
        "1.2.840.10008.1.2.4.91",  # JPEG 2000
    )

    if transfer_syntax is not None and str(transfer_syntax).startswith(compressed_prefixes):
        try:
            dcm.decompress()
        except Exception as exc:
            raise RuntimeError(
                f"Compressed pixel data could not be decoded "
                f"(transfer syntax {transfer_syntax}). Install an applicable "
                f"pydicom image handler: {exc}"
            ) from exc

    return dcm.pixel_array


def normalize_pixels(dcm) -> np.ndarray:
    pixels = read_pixel_array(dcm).astype(np.float32)

    slope = safe_float(getattr(dcm, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(dcm, "RescaleIntercept", 0.0), 0.0)
    pixels = pixels * slope + intercept

    if not np.isfinite(pixels).all():
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)

    return pixels


def slice_sort_key(item: Slice) -> tuple:
    if item.position is not None:
        return (0, item.position[2], item.position[1], item.position[0])
    return (1, item.instance_number, 0.0, 0.0)


def add_slice(
    series_dict: dict[str, Series],
    dcm,
    pixels: np.ndarray,
    warnings: list[str],
    filename: str,
) -> None:
    series_name = str(
        getattr(dcm, "SeriesDescription", "")
        or getattr(dcm, "ProtocolName", "")
        or f"Sequence {getattr(dcm, 'SeriesNumber', 'Unknown')}"
    ).strip() or "Unnamed sequence"

    try:
        instance_number = int(dcm.InstanceNumber)
    except (AttributeError, TypeError, ValueError):
        instance_number = len(series_dict.get(series_name, Series(series_name)).slices)

    position = None
    if hasattr(dcm, "ImagePositionPatient"):
        try:
            position = tuple(float(value) for value in dcm.ImagePositionPatient)
        except (TypeError, ValueError):
            position = None

    center, width = get_window_values(dcm, pixels)

    if pixels.ndim == 2:
        frames = [pixels]
    elif pixels.ndim == 3:
        frames = [pixels[frame] for frame in range(pixels.shape[0])]
    else:
        warnings.append(f"{filename}: unsupported pixel array shape {pixels.shape}")
        return

    if series_name not in series_dict:
        spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
        row_spacing = safe_float(spacing[0], 1.0)
        column_spacing = safe_float(spacing[1], 1.0)

        series_dict[series_name] = Series(
            name=series_name,
            row_spacing=max(row_spacing, 1e-6),
            column_spacing=max(column_spacing, 1e-6),
        )

    for frame_index, frame in enumerate(frames):
        series_dict[series_name].slices.append(
            Slice(
                instance_number=instance_number + frame_index,
                position=position,
                pixel_array=frame,
                window_center=center,
                window_width=width,
            )
        )


def extract_patient_metadata(dcm, patient: dict) -> None:
    if patient["name"] == "Unknown" and hasattr(dcm, "PatientName"):
        patient["name"] = str(dcm.PatientName) or "Unknown"
    if patient["id"] == "N/A" and hasattr(dcm, "PatientID"):
        patient["id"] = str(dcm.PatientID) or "N/A"
    if patient["date"] == "N/A" and hasattr(dcm, "StudyDate"):
        patient["date"] = str(dcm.StudyDate) or "N/A"
    if patient["manufacturer"] == "N/A" and hasattr(dcm, "Manufacturer"):
        patient["manufacturer"] = str(dcm.Manufacturer) or "N/A"


def process_dicom_bytes(
    data: bytes,
    filename: str,
    series_dict: dict[str, Series],
    warnings: list[str],
    patient: dict,
) -> None:
    try:
        dcm = pydicom.dcmread(io.BytesIO(data), force=True)

        if not hasattr(dcm, "PixelData"):
            warnings.append(f"{filename}: no PixelData element")
            return

        pixels = normalize_pixels(dcm)
        add_slice(series_dict, dcm, pixels, warnings, filename)
        extract_patient_metadata(dcm, patient)

    except InvalidDicomError as exc:
        warnings.append(f"{filename}: not recognized as DICOM — {exc}")
    except NotImplementedError as exc:
        warnings.append(
            f"{filename}: unsupported transfer syntax or pixel format — {exc}"
        )
    except Exception as exc:
        warnings.append(
            f"{filename}: {type(exc).__name__}: {exc}"
        )


@st.cache_data(show_spinner="Extracting and parsing DICOM files...")
def parse_archive(file_bytes: bytes):
    series_dict: dict[str, Series] = {}
    warnings: list[str] = []
    archive_entries: list[str] = []
    patient = {
        "name": "Unknown",
        "id": "N/A",
        "date": "N/A",
        "manufacturer": "N/A",
    }

    def process_zip_bytes(zip_bytes: bytes, label: str) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue

                    entry_name = f"{label}/{info.filename}"
                    archive_entries.append(entry_name)

                    path_parts = [p for p in info.filename.split("/") if p]
                    if not path_parts or path_parts[0] == "__MACOSX__":
                        continue

                    lower_name = info.filename.lower()

                    with archive.open(info) as handle:
                        data = handle.read()

                    if lower_name.endswith(".zip"):
                        process_zip_bytes(data, entry_name)
                    else:
                        process_dicom_bytes(
                            data,
                            entry_name,
                            series_dict,
                            warnings,
                            patient,
                        )

        except zipfile.BadZipFile as exc:
            warnings.append(f"{label}: invalid ZIP — {exc}")

    process_zip_bytes(file_bytes, "uploaded.zip")

    for series_item in series_dict.values():
        series_item.slices.sort(key=slice_sort_key)

    return series_dict, warnings, archive_entries, patient


def apply_enhancement(pixels: np.ndarray) -> np.ndarray:
    image = np.asarray(pixels, dtype=np.float32)
    shifted = image - float(np.min(image)) + 1.0

    log_image = np.log(shifted)
    sigma = max(1.0, max(image.shape) / 6.0)
    low_frequency = ndimage.gaussian_filter(log_image, sigma=sigma)
    corrected = np.exp(log_image - low_frequency)

    smooth = ndimage.gaussian_filter(corrected, sigma=1.2)
    suppressed = corrected - 0.35 * (corrected - smooth)
    enhanced = suppressed - 0.20 * ndimage.laplace(suppressed)

    return np.clip(enhanced, 0.0, None).astype(np.float32)


def encode_float32(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def build_payload(selected: dict[str, Series], enhancement: bool):
    payload = []
    warnings = []

    for name, series_item in selected.items():
        if len(series_item.slices) > MAX_SLICES_PER_SERIES:
            warnings.append(
                f"{name}: displaying first {MAX_SLICES_PER_SERIES} of "
                f"{len(series_item.slices)} slices."
            )

        slices = []
        for item in series_item.slices[:MAX_SLICES_PER_SERIES]:
            pixels = apply_enhancement(item.pixel_array) if enhancement else item.pixel_array
            slices.append(
                {
                    "pixels": encode_float32(pixels),
                    "rows": int(pixels.shape[0]),
                    "cols": int(pixels.shape[1]),
                    "windowCenter": float(item.window_center),
                    "windowWidth": float(item.window_width),
                }
            )

        payload.append(
            {
                "title": name,
                "rowSpacing": float(series_item.row_spacing),
                "columnSpacing": float(series_item.column_spacing),
                "slices": slices,
            }
        )

    return payload, warnings


def render_viewer(payload: list[dict]) -> None:
    columns = min(3, len(payload))
    rows = math.ceil(len(payload) / columns)

    panels = ""
    for index in range(len(payload)):
        panels += f"""
        <div class="viewport-panel">
            <div class="viewport-title" id="title-{index}"></div>
            <canvas id="canvas-{index}"></canvas>
            <div class="viewport-status" id="status-{index}"></div>
        </div>
        """

    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""
<div class="viewport-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr));">
{panels}
</div>

<script type="application/json" id="workstation-data">{safe_json}</script>
<script>
(() => {{
    const workstationData = JSON.parse(
        document.getElementById("workstation-data").textContent
    );
    const current = new Array(workstationData.length).fill(0);
    const centers = [];
    const widths = [];
    const caliper = {{active:false, panel:null, x1:0, y1:0, x2:0, y2:0}};
    const drag = {{active:false, panel:null, lastX:0, lastY:0}};

    workstationData.forEach((series, i) => {{
        centers[i] = series.slices[0]?.windowCenter ?? 0;
        widths[i] = series.slices[0]?.windowWidth ?? 1;
    }});

    function decode(base64) {{
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return new Float32Array(bytes.buffer);
    }}

    function render(index) {{
        const series = workstationData[index];
        const slice = series.slices[current[index]];
        const canvas = document.getElementById("canvas-" + index);
        const context = canvas.getContext("2d");
        if (!slice || !context) return;

        document.getElementById("title-" + index).textContent = series.title;

        const values = decode(slice.pixels);
        canvas.width = slice.cols;
        canvas.height = slice.rows;

        const image = context.createImageData(slice.cols, slice.rows);
        const out = image.data;
        const center = centers[index];
        const width = Math.max(1, widths[index]);
        const low = center - width / 2;
        const scale = 255 / width;

        for (let i = 0; i < values.length; i++) {{
            const value = Math.min(center + width / 2, Math.max(low, values[i]));
            const gray = Math.max(0, Math.min(255, Math.round((value - low) * scale)));
            const o = i * 4;
            out[o] = out[o + 1] = out[o + 2] = gray;
            out[o + 3] = 255;
        }}

        context.putImageData(image, 0, 0);

        if (caliper.active && caliper.panel === index) {{
            context.strokeStyle = "#22d3ee";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(caliper.x1, caliper.y1);
            context.lineTo(caliper.x2, caliper.y2);
            context.stroke();

            const dx = (caliper.x2 - caliper.x1) * series.columnSpacing;
            const dy = (caliper.y2 - caliper.y1) * series.rowSpacing;
            const distance = Math.sqrt(dx * dx + dy * dy);

            context.font = "bold 14px monospace";
            context.fillStyle = "#22d3ee";
            context.shadowColor = "#000";
            context.shadowBlur = 4;
            context.fillText(distance.toFixed(2) + " mm", caliper.x2 + 8, caliper.y2 - 8);
        }}

        document.getElementById("status-" + index).textContent =
            `Slice ${{current[index] + 1}}/${{series.slices.length}} | ` +
            `W ${{Math.round(width)}} | L ${{Math.round(center)}}`;
    }}

    function coords(event, canvas) {{
        const rect = canvas.getBoundingClientRect();
        return [
            (event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height
        ];
    }}

    workstationData.forEach((series, index) => {{
        const canvas = document.getElementById("canvas-" + index);
        if (!canvas) return;

        canvas.addEventListener("wheel", event => {{
            event.preventDefault();
            const count = series.slices.length;
            current[index] = (current[index] + (event.deltaY > 0 ? 1 : -1) + count) % count;
            render(index);
        }}, {{passive:false}});

        canvas.addEventListener("mousedown", event => {{
            const [x, y] = coords(event, canvas);

            if (event.shiftKey) {{
                Object.assign(caliper, {{
                    active:true, panel:index, x1:x, y1:y, x2:x, y2:y
                }});
            }} else {{
                Object.assign(drag, {{
                    active:true, panel:index,
                    lastX:event.clientX, lastY:event.clientY
                }});
                caliper.active = false;
            }}
        }});
    }});

    window.addEventListener("mousemove", event => {{
        if (caliper.active && caliper.panel !== null) {{
            const canvas = document.getElementById("canvas-" + caliper.panel);
            if (!canvas) return;
            [caliper.x2, caliper.y2] = coords(event, canvas);
            render(caliper.panel);
        }} else if (drag.active && drag.panel !== null) {{
            widths[drag.panel] = Math.max(
                1,
                widths[drag.panel] + (event.clientX - drag.lastX) * 2
            );
            centers[drag.panel] -= (event.clientY - drag.lastY) * 2;
            drag.lastX = event.clientX;
            drag.lastY = event.clientY;
            render(drag.panel);
        }}
    }});

    window.addEventListener("mouseup", () => {{
        drag.active = false;
    }});

    for (let i = 0; i < workstationData.length; i++) render(i);
}})();
</script>
"""

    components.html(html, height=max(430, 440 * rows), scrolling=False)


uploaded_file = st.sidebar.file_uploader("Upload DICOM ZIP", type=["zip"])
enhancement = st.sidebar.toggle("Enhancement mode", value=False)

if uploaded_file is None:
    st.info("Upload a ZIP archive containing DICOM files.")
    st.stop()

try:
    series_dict, warnings, entries, patient = parse_archive(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Failed to read archive: {type(exc).__name__}: {exc}")
    st.stop()

with st.expander("Archive diagnosis", expanded=not series_dict):
    st.write(f"**Entries found:** {len(entries)}")
    if entries:
        st.write("\n".join(entries[:100]))
    st.write(f"**Warnings:** {len(warnings)}")
    if warnings:
        st.write("\n".join(warnings[:200]))

if not series_dict:
    st.error(
        "No readable DICOM images with usable pixel data were found. "
        "Open **Archive diagnosis** above: it lists every archived file and the exact reason each one was skipped."
    )

    st.markdown("### Common causes")
    st.markdown(
        "- The files use a compressed transfer syntax that pydicom cannot decode in this environment.\n"
        "- The archive contains another ZIP inside it; this version now checks nested ZIPs.\n"
        "- The files are not actually DICOM, despite having a `.dcm` extension.\n"
        "- Pixel data cannot be accessed because required DICOM tags or image handlers are missing."
    )
    st.stop()

series_names = sorted(series_dict)
selected_names = st.sidebar.multiselect(
    f"Select sequences (maximum {MAX_SERIES})",
    series_names,
    default=series_names[: min(3, len(series_names))],
)
selected_names = selected_names[:MAX_SERIES]

st.sidebar.markdown("### Patient")
st.sidebar.text(f"Name: {patient['name']}")
st.sidebar.text(f"ID: {patient['id']}")
st.sidebar.text(f"Study date: {patient['date']}")
st.sidebar.text(f"Scanner: {patient['manufacturer']}")

if warnings:
    with st.sidebar.expander(f"Warnings ({len(warnings)})"):
        st.write("\n".join(warnings))

if not selected_names:
    st.warning("Select at least one sequence.")
    st.stop()

payload, payload_warnings = build_payload(
    {name: series_dict[name] for name in selected_names},
    enhancement,
)

for warning in payload_warnings:
    st.warning(warning)

render_viewer(payload)

st.caption(
    "Scroll: change slice. Left-drag: window/level. Shift+drag: caliper. "
    "Measurements use PixelSpacing only and are not clinically validated."
)