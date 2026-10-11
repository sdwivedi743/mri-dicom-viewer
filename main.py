import io
import math
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom
import scipy.ndimage as ndimage
import streamlit as st
from pydicom.errors import InvalidDicomError

st.set_page_config(
    page_title="NeuroVue MRI Workstation",
    page_icon="🧠",
    layout="wide",
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

    return np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)


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
        instance_number = len(
            series_dict.get(series_name, Series(series_name)).slices
        )

    position = None

    if hasattr(dcm, "ImagePositionPatient"):
        try:
            position = tuple(
                float(value) for value in dcm.ImagePositionPatient
            )
        except (TypeError, ValueError):
            position = None

    center, width = get_window_values(dcm, pixels)

    if pixels.ndim == 2:
        frames = [pixels]
    elif pixels.ndim == 3:
        frames = [pixels[index] for index in range(pixels.shape[0])]
    else:
        warnings.append(
            f"{filename}: unsupported pixel-array shape {pixels.shape}"
        )
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


@st.cache_data(show_spinner="Reading DICOM archive...")
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

                    parts = [
                        part
                        for part in info.filename.split("/")
                        if part
                    ]

                    if not parts or parts[0] == "__MACOSX__":
                        continue

                    with archive.open(info) as handle:
                        data = handle.read()

                    if info.filename.lower().endswith(".zip"):
                        walk_zip(data, path)
                    else:
                        process_dicom(
                            data,
                            path,
                            series_dict,
                            warnings,
                            patient,
                        )

        except zipfile.BadZipFile as exc:
            warnings.append(f"{label}: invalid ZIP — {exc}")

    walk_zip(file_bytes, "upload")

    for series_item in series_dict.values():
        series_item.slices.sort(key=slice_sort_key)

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


def window_image(
    pixels: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    width = max(1.0, float(width))
    lower = center - width / 2
    upper = center + width / 2

    windowed = np.clip(pixels, lower, upper)
    windowed = (windowed - lower) / width * 255.0

    return windowed.astype(np.uint8)


def render_series_viewer(
    series_item: Series,
    enhancement: bool,
) -> None:
    if not series_item.slices:
        st.warning("This series contains no readable slices.")
        return

    slice_count = len(series_item.slices)
    first = series_item.slices[0]

    st.markdown(f"#### {series_item.name}")

    left, right = st.columns([1, 2])

    with left:
        slice_index = st.slider(
            "Slice",
            min_value=0,
            max_value=slice_count - 1,
            value=0,
        )

    with right:
        minimum_center = float(np.min(first.pixel_array))
        maximum_center = float(np.max(first.pixel_array))

        if maximum_center <= minimum_center:
            maximum_center = minimum_center + 1.0

        window_width = st.slider(
            "Window width",
            min_value=1.0,
            max_value=max(
                100.0,
                float(np.ptp(first.pixel_array)) * 2.0,
            ),
            value=float(first.window_width),
        )

        window_center = st.slider(
            "Window center",
            min_value=minimum_center,
            max_value=maximum_center,
            value=float(first.window_center),
        )

    current = series_item.slices[slice_index]
    pixels = current.pixel_array

    if enhancement:
        pixels = apply_enhancement(pixels)

    display = window_image(
        pixels,
        window_center,
        window_width,
    )

    st.image(
        display,
        use_container_width=True,
        caption=(
            f"Slice {slice_index + 1} / {slice_count} · "
            f"W {window_width:.0f} · L {window_center:.0f}"
        ),
    )

    st.caption(
        f"Image: {pixels.shape[1]} × {pixels.shape[0]} px · "
        f"Row spacing: {series_item.row_spacing:.3f} mm · "
        f"Column spacing: {series_item.column_spacing:.3f} mm"
    )


st.title("🧠 NeuroVue MRI Workstation")
st.caption("Local DICOM review · not for clinical diagnosis")

uploaded_file = st.sidebar.file_uploader(
    "DICOM study archive",
    type=["zip"],
)

if uploaded_file is None:
    st.info("Upload a ZIP archive containing DICOM MRI files.")
    st.stop()

try:
    series_dict, warnings, entries, patient = parse_archive(
        uploaded_file.getvalue()
    )
except Exception as exc:
    st.error(f"Could not read archive: {type(exc).__name__}: {exc}")
    st.stop()

if not series_dict:
    st.error("No readable DICOM pixel data was found.")

    with st.expander("Archive diagnosis", expanded=True):
        st.write(f"Files inspected: {len(entries)}")
        st.write("\n".join(entries[:100]) or "No files found.")
        st.write(f"Warnings: {len(warnings)}")
        st.write("\n".join(warnings[:200]) or "No detailed warnings.")

    st.stop()

series_names = sorted(series_dict)
total_slices = sum(len(item.slices) for item in series_dict.values())

st.success(
    f"Loaded {len(series_dict)} series containing {total_slices} slices."
)

selected_names = st.sidebar.multiselect(
    f"Active series (maximum {MAX_SERIES})",
    series_names,
    default=series_names[: min(2, len(series_names))],
    max_selections=MAX_SERIES,
)

layout = st.sidebar.radio(
    "Layout",
    ["Single focus", "Dual compare", "Grid"],
    horizontal=True,
)

enhancement = st.sidebar.toggle(
    "Enhancement",
    value=False,
    help="Classical filtering; not AI super-resolution.",
)

st.sidebar.markdown("### Patient")
st.sidebar.text(f"Name: {patient['name']}")
st.sidebar.text(f"ID: {patient['id']}")
st.sidebar.text(f"Study date: {patient['date']}")
st.sidebar.text(f"Scanner: {patient['manufacturer']}")

if warnings:
    with st.sidebar.expander(f"Warnings ({len(warnings)})"):
        st.write("\n".join(warnings))

if not selected_names:
    st.warning("Select at least one series.")
    st.stop()

active_names = selected_names[:MAX_SERIES]

viewer_tab, details_tab, diagnostics_tab = st.tabs(
    ["Viewer", "Series details", "Diagnostics"]
)

with viewer_tab:
    if len(active_names) == 1 or layout == "Single focus":
        if len(active_names) == 1:
            focused_name = active_names[0]
        else:
            focused_name = st.selectbox(
                "Focused series",
                active_names,
            )

        render_series_viewer(
            series_dict[focused_name],
            enhancement,
        )

    else:
        column_count = (
            2
            if layout == "Dual compare"
            else min(3, len(active_names))
        )

        for start in range(0, len(active_names), column_count):
            row_names = active_names[start:start + column_count]
            columns = st.columns(len(row_names))

            for column, name in zip(columns, row_names):
                with column:
                    render_series_viewer(
                        series_dict[name],
                        enhancement,
                    )

with details_tab:
    for name in active_names:
        item = series_dict[name]
        st.markdown(f"#### {name}")

        first_column, second_column, third_column = st.columns(3)
        first_column.metric("Slices", len(item.slices))
        second_column.metric(
            "Row spacing",
            f"{item.row_spacing:.3f} mm",
        )
        third_column.metric(
            "Column spacing",
            f"{item.column_spacing:.3f} mm",
        )

with diagnostics_tab:
    st.write(f"Files inspected: **{len(entries)}**")
    st.write(f"Readable series: **{len(series_dict)}**")
    st.write(f"Warnings: **{len(warnings)}**")

    if warnings:
        st.write("\n".join(warnings))