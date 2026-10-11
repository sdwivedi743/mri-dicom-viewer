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

st.set_page_config(
    page_title="NeuroVue MRI Workstation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_SERIES = 6
MAX_SLICES_PER_SERIES = 250

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

      .stApp, .stSidebar {
          background: #070b14 !important;
          color: #e6edf7 !important;
          font-family: Inter, system-ui, sans-serif !important;
      }

      section[data-testid="stSidebar"] {
          border-right: 1px solid #16223a;
          background: #0a1020 !important;
      }

      .block-container { padding-top: 1.2rem !important; }

      h1, h2, h3, .stMarkdown, .stText, .stCaption {
          color: #e6edf7 !important;
      }

      .nv-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          padding: 14px 18px;
          border: 1px solid #17233c;
          border-radius: 16px;
          background: linear-gradient(180deg, #0d1526 0%, #080d19 100%);
          margin-bottom: 18px;
      }

      .nv-brand {
          display: flex;
          align-items: center;
          gap: 12px;
      }

      .nv-logo {
          width: 40px;
          height: 40px;
          display: grid;
          place-items: center;
          border-radius: 12px;
          background: linear-gradient(135deg, #22d3ee, #6366f1);
          font-size: 20px;
      }

      .nv-title {
          font-size: 1.15rem;
          font-weight: 700;
          letter-spacing: -0.02em;
      }

      .nv-subtitle {
          color: #7d8db1;
          font-size: 0.78rem;
          margin-top: 2px;
      }

      .nv-pill {
          border: 1px solid #1d2b47;
          background: #0b1322;
          color: #9fb2d4;
          border-radius: 999px;
          padding: 6px 12px;
          font-size: 0.74rem;
          font-family: 'JetBrains Mono', monospace;
      }

      .nv-empty {
          border: 1px dashed #243352;
          border-radius: 18px;
          padding: 48px 32px;
          text-align: center;
          background: #080e1a;
      }

      .nv-empty h2 {
          margin: 0 0 8px 0 !important;
          font-size: 1.35rem !important;
      }

      .nv-empty p {
          color: #8296b8 !important;
          margin: 0 !important;
      }

      .stTabs [data-baseweb="tab-list"] {
          gap: 6px;
          border-bottom: 1px solid #17233c;
      }

      .stTabs [data-baseweb="tab"] {
          background: transparent;
          border-radius: 10px 10px 0 0;
          color: #8296b8 !important;
          font-weight: 600;
      }

      .stTabs [aria-selected="true"] {
          color: #67e8f9 !important;
          border-bottom: 2px solid #22d3ee;
      }

      div[data-testid="stFileUploader"] {
          border: 1px dashed #243352 !important;
          border-radius: 14px !important;
          background: #0a1120 !important;
      }

      .stButton > button {
          width: 100%;
          border-radius: 10px;
          border: 1px solid #1d2b47 !important;
          background: #0b1322 !important;
          color: #dbe6f7 !important;
          font-weight: 600;
      }

      .stButton > button:hover {
          border-color: #22d3ee !important;
          color: #67e8f9 !important;
      }

      .stSlider > div > div > div {
          color: #9fb2d4;
      }

      .nv-footer {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          z-index: 999;
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 18px;
          background: rgba(7, 11, 20, 0.92);
          border-top: 1px solid #16223a;
          color: #7d8db1;
          font-size: 0.72rem;
          font-family: 'JetBrains Mono', monospace;
          backdrop-filter: blur(10px);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    transfer_syntax = str(
        getattr(getattr(dcm, "file_meta", None), "TransferSyntaxUID", "")
    )

    compressed_prefixes = (
        "1.2.840.10008.1.2.4.50",
        "1.2.840.10008.1.2.4.51",
        "1.2.840.10008.1.2.4.57",
        "1.2.840.10008.1.2.4.70",
        "1.2.840.10008.1.2.4.80",
        "1.2.840.10008.1.2.4.90",
        "1.2.840.10008.1.2.4.91",
    )

    if transfer_syntax.startswith(compressed_prefixes):
        try:
            dcm.decompress()
        except Exception as exc:
            raise RuntimeError(
                f"Unsupported compressed transfer syntax {transfer_syntax}: {exc}"
            ) from exc

    return dcm.pixel_array


def normalize_pixels(dcm) -> np.ndarray:
    pixels = read_pixel_array(dcm).astype(np.float32)

    slope = safe_float(getattr(dcm, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(dcm, "RescaleIntercept", 0.0), 0.0)
    pixels = pixels * slope + intercept

    return np.nan_to_num(
        pixels,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


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
        or f"Series {getattr(dcm, 'SeriesNumber', '?')}"
    ).strip() or "Unnamed series"

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
        frames = [pixels[i] for i in range(pixels.shape[0])]
    else:
        warnings.append(f"{filename}: unsupported pixel shape {pixels.shape}")
        return

    if series_name not in series_dict:
        spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
        series_dict[series_name] = Series(
            name=series_name,
            row_spacing=max(safe_float(spacing[0], 1.0), 1e-6),
            column_spacing=max(safe_float(spacing[1], 1.0), 1e-6),
        )

    for offset, frame in enumerate(frames):
        series_dict[series_name].slices.append(
            Slice(
                instance_number=instance_number + offset,
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


def process_dicom(
    data: bytes,
    filename: str,
    series_dict: dict[str, Series],
    warnings: list[str],
    patient: dict,
) -> None:
    try:
        dcm = pydicom.dcmread(io.BytesIO(data), force=True)

        if not hasattr(dcm, "PixelData"):
            warnings.append(f"{filename}: no PixelData")
            return

        pixels = normalize_pixels(dcm)
        add_slice(series_dict, dcm, pixels, warnings, filename)
        extract_patient_metadata(dcm, patient)

    except InvalidDicomError as exc:
        warnings.append(f"{filename}: not DICOM — {exc}")
    except Exception as exc:
        warnings.append(f"{filename}: {type(exc).__name__}: {exc}")


@st.cache_data(show_spinner="Preparing imaging series...")
def parse_archive(file_bytes: bytes):
    series_dict: dict[str, Series] = {}
    warnings: list[str] = []
    entries: list[str] = []
    patient = {
        "name": "Unknown",
        "id": "N/A",
        "date": "N/A",
        "manufacturer": "N/A",
    }

    def walk_zip(zip_bytes: bytes, label: str) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue

                    path = f"{label}/{info.filename}"
                    entries.append(path)

                    parts = [part for part in info.filename.split("/") if part]
                    if not parts or parts[0] == "__MACOSX__":
                        continue

                    with archive.open(info) as handle:
                        data = handle.read()

                    if info.filename.lower().endswith(".zip"):
                        walk_zip(data, path)
                    else:
                        process_dicom(data, path, series_dict, warnings, patient)

        except zipfile.BadZipFile as exc:
            warnings.append(f"{label}: invalid ZIP — {exc}")

    walk_zip(file_bytes, "upload")

    for item in series_dict.values():
        item.slices.sort(key=slice_sort_key)

    return series_dict, warnings, entries, patient


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
                f"{name}: first {MAX_SLICES_PER_SERIES} of "
                f"{len(series_item.slices)} slices loaded."
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


def render_workspace(payload: list[dict], layout: str) -> None:
    if not payload:
        st.warning("Select at least one series from the study rail.")
        return

    if layout == "Single focus":
        columns = 1
    elif layout == "Dual compare":
        columns = min(2, len(payload))
    else:
        columns = min(3, len(payload))

    rows = math.ceil(len(payload) / columns)

    panels = ""
    for index in range(len(payload)):
        panels += f"""
        <div class="viewport-panel">
          <div class="panel-head">
            <span id="title-{index}" class="panel-title"></span>
            <span id="badge-{index}" class="panel-badge"></span>
          </div>
          <canvas id="canvas-{index}"></canvas>
          <div class="panel-tools">
            <button data-action="prev" data-panel="{index}">◀ Prev</button>
            <button data-action="next" data-panel="{index}">Next ▶</button>
            <button data-action="reset" data-panel="{index}">Reset W/L</button>
            <button data-action="caliper" data-panel="{index}">Caliper</button>
          </div>
          <div id="status-{index}" class="panel-status"></div>
        </div>
        """

    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""
<div class="viewer-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr));">
{panels}
</div>

<script type="application/json" id="workspace-data">{safe_json}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("workspace-data").textContent);
  const current = new Array(data.length).fill(0);
  const centers = [];
  const widths = [];
  const defaults = [];
  const caliper = {{enabled:false, panel:null, x1:0, y1:0, x2:0, y2:0}};
  const drag = {{active:false, panel:null, lastX:0, lastY:0}};

  data.forEach((series, i) => {{
    centers[i] = series.slices[0]?.windowCenter ?? 0;
    widths[i] = series.slices[0]?.windowWidth ?? 1;
    defaults[i] = [centers[i], widths[i]];
  }});

  function decode(b64) {{
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }}

  function render(index) {{
    const series = data[index];
    const slice = series.slices[current[index]];
    const canvas = document.getElementById("canvas-" + index);
    const context = canvas.getContext("2d");
    if (!slice || !context) return;

    document.getElementById("title-" + index).textContent = series.title;
    document.getElementById("badge-" + index).textContent =
      `${{current[index] + 1}} / ${{series.slices.length}}`;

    const pixels = decode(slice.pixels);
    canvas.width = slice.cols;
    canvas.height = slice.rows;

    const image = context.createImageData(slice.cols, slice.rows);
    const output = image.data;
    const center = centers[index];
    const width = Math.max(1, widths[index]);
    const low = center - width / 2;
    const scale = 255 / width;

    for (let i = 0; i < pixels.length; i++) {{
      const value = Math.min(center + width / 2, Math.max(low, pixels[i]));
      const gray = Math.max(0, Math.min(255, Math.round((value - low) * scale)));
      const o = i * 4;
      output[o] = output[o + 1] = output[o + 2] = gray;
      output[o + 3] = 255;
    }}

    context.putImageData(image, 0, 0);

    if (caliper.enabled && caliper.panel === index) {{
      context.strokeStyle = "#22d3ee";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(caliper.x1, caliper.y1);
      context.lineTo(caliper.x2, caliper.y2);
      context.stroke();

      const dx = (caliper.x2 - caliper.x1) * series.columnSpacing;
      const dy = (caliper.y2 - caliper.y1) * series.rowSpacing;
      const distance = Math.sqrt(dx * dx + dy * dy);

      context.font = "600 14px 'JetBrains Mono', monospace";
      context.fillStyle = "#67e8f9";
      context.shadowColor = "#000";
      context.shadowBlur = 4;
      context.fillText(distance.toFixed(2) + " mm", caliper.x2 + 8, caliper.y2 - 8);
    }}

    document.getElementById("status-" + index).textContent =
      `W ${{Math.round(width)}} · L ${{Math.round(center)}} · ` +
      `${{slice.cols}}×${{slice.rows}} px`;
  }}

  function coords(event, canvas) {{
    const rect = canvas.getBoundingClientRect();
    return [
      (event.clientX - rect.left) * canvas.width / rect.width,
      (event.clientY - rect.top) * canvas.height / rect.height
    ];
  }}

  data.forEach((series, index) => {{
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

      if (caliper.enabled && caliper.panel === index) {{
        caliper.x1 = x;
        caliper.y1 = y;
        caliper.x2 = x;
        caliper.y2 = y;
      }} else {{
        drag.active = true;
        drag.panel = index;
        drag.lastX = event.clientX;
        drag.lastY = event.clientY;
      }}
    }});
  }});

  window.addEventListener("mousemove", event => {{
    if (drag.active && drag.panel !== null) {{
      widths[drag.panel] = Math.max(
        1,
        widths[drag.panel] + (event.clientX - drag.lastX) * 2
      );
      centers[drag.panel] -= (event.clientY - drag.lastY) * 2;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      render(drag.panel);
    }} else if (caliper.enabled && caliper.panel !== null) {{
      const canvas = document.getElementById("canvas-" + caliper.panel);
      if (!canvas) return;
      [caliper.x2, caliper.y2] = coords(event, canvas);
      render(caliper.panel);
    }}
  }});

  window.addEventListener("mouseup", () => {{
    drag.active = false;
  }});

  document.querySelectorAll("button[data-action]").forEach(button => {{
    button.addEventListener("click", () => {{
      const index = Number(button.dataset.panel);
      const action = button.dataset.action;

      if (action === "prev") {{
        current[index] = (current[index] - 1 + data[index].slices.length) %
          data[index].slices.length;
      }} else if (action === "next") {{
        current[index] = (current[index] + 1) % data[index].slices.length;
      }} else if (action === "reset") {{
        centers[index] = defaults[index][0];
        widths[index] = defaults[index][1];
      }} else if (action === "caliper") {{
        caliper.enabled = !(caliper.enabled && caliper.panel === index);
        caliper.panel = index;
      }}

      render(index);
    }});
  }});

  for (let i = 0; i < data.length; i++) render(i);
}})();
</script>
"""

    st.markdown(
        """
        <style>
          .viewer-grid {
            display: grid;
            gap: 16px;
          }
          .viewport-panel {
            border: 1px solid #17233c;
            border-radius: 16px;
            background: #0a1120;
            padding: 12px;
          }
          .panel-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
          }
          .panel-title {
            color: #67e8f9;
            font-weight: 700;
            font-size: 0.82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .panel-badge {
            color: #9fb2d4;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            border: 1px solid #1d2b47;
            padding: 2px 8px;
            border-radius: 999px;
          }
          canvas {
            width: 100%;
            height: auto;
            display: block;
            background: #000;
            border-radius: 10px;
            cursor: crosshair;
          }
          .panel-tools {
            display: flex;
            gap: 6px;
            margin-top: 8px;
            flex-wrap: wrap;
          }
          .panel-tools button {
            border: 1px solid #1d2b47;
            background: #0b1322;
            color: #9fb2d4;
            border-radius: 8px;
            font-size: 0.68rem;
            padding: 5px 9px;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
          }
          .panel-tools button:hover {
            color: #67e8f9;
            border-color: #22d3ee;
          }
          .panel-status {
            margin-top: 7px;
            color: #7d8db1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(html, height=max(520, 560 * rows), scrolling=False)


def render_topbar(patient: dict, series_count: int, slice_count: int) -> None:
    st.markdown(
        f"""
        <div class="nv-topbar">
          <div class="nv-brand">
            <div class="nv-logo">🧠</div>
            <div>
              <div class="nv-title">NeuroVue MRI Workstation</div>
              <div class="nv-subtitle">Local DICOM review · no cloud upload</div>
            </div>
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="nv-pill">{series_count} series</span>
            <span class="nv-pill">{slice_count} slices</span>
            <span class="nv-pill">{patient['name']}</span>
            <span class="nv-pill">ID {patient['id']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="nv-empty">
          <h2>Load a DICOM study</h2>
          <p>
            Upload a ZIP archive containing MRI DICOM files. NeuroVue parses it locally,
            groups images into series, and provides slice navigation, window/level control,
            and a caliper.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


uploaded_file = st.sidebar.file_uploader(
    "DICOM study archive",
    type=["zip"],
    help="Upload a .zip containing DICOM files. Nested ZIP archives are also supported.",
)

if uploaded_file is None:
    render_empty_state()
    st.stop()

try:
    series_dict, warnings, entries, patient = parse_archive(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not read archive: {type(exc).__name__}: {exc}")
    st.stop()

if not series_dict:
    st.error("No readable DICOM pixel data was found.")
    with st.expander("Archive diagnosis", expanded=True):
        st.write(f"**Files inspected:** {len(entries)}")
        st.write("\n".join(entries[:100]) or "No files found.")
        st.write(f"**Warnings:** {len(warnings)}")
        st.write("\n".join(warnings[:200]) or "No detailed warnings.")
    st.stop()

series_names = sorted(series_dict)
total_slices = sum(len(item.slices) for item in series_dict.values())

render_topbar(patient, len(series_names), total_slices)

left, right = st.columns([1, 4], gap="large")

with left:
    st.markdown("### Study rail")

    chosen = st.multiselect(
        "Active series",
        series_names,
        default=series_names[: min(2, len(series_names))],
        max_selections=MAX_SERIES,
        label_visibility="collapsed",
    )

    st.markdown("### Display")

    layout = st.radio(
        "Layout",
        ["Single focus", "Dual compare", "Grid"],
        horizontal=True,
        label_visibility="collapsed",
    )

    enhancement = st.toggle(
        "Enhancement",
        value=False,
        help="Classical bias-field and sharpening filters. Not AI super-resolution.",
    )

    st.markdown("### Study")
    st.caption(f"Date: {patient['date']}")
    st.caption(f"Scanner: {patient['manufacturer']}")

    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
            st.write("\n".join(warnings))

with right:
    if not chosen:
        st.info("Choose one or more series from the study rail.")
    else:
        tabs = st.tabs(["Viewer", "Series details", "Diagnostics"])

        with tabs[0]:
            payload, payload_warnings = build_payload(
                {name: series_dict[name] for name in chosen[:MAX_SERIES]},
                enhancement,
            )
            for warning in payload_warnings:
                st.warning(warning)
            render_workspace(payload, layout)

        with tabs[1]:
            for name in chosen[:MAX_SERIES]:
                item = series_dict[name]
                st.markdown(f"#### {name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Slices", len(item.slices))
                c2.metric("Row spacing", f"{item.row_spacing:.3f} mm")
                c3.metric("Column spacing", f"{item.column_spacing:.3f} mm")

        with tabs[2]:
            st.write(f"Files inspected: **{len(entries)}**")
            st.write(f"Readable series: **{len(series_dict)}**")
            st.write(f"Warnings: **{len(warnings)}**")
            if warnings:
                st.write("\n".join(warnings))

st.markdown(
    """
    <div class="nv-footer">
      <span>NeuroVue · local review only</span>
      <span>Scroll: slice · Drag: window/level · Caliper: measurement</span>
      <span>Not for clinical diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)