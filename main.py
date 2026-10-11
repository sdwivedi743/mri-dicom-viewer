import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import zipfile
import io

st.set_page_config(page_title="Advanced MRI DICOM Viewer", page_icon="🧠", layout="centered")
st.title("🧠 Advanced MRI DICOM Viewer")
st.write("Upload a `.zip` folder containing your MRI scan series.")

uploaded_file = st.file_uploader("Drop your ZIP folder here")

if uploaded_file is not None:
    # Dictionary structure: { "Series Name": [ (InstanceNumber, dicom_dataset), ... ] }
    series_groups = {}

    if zipfile.is_zipfile(uploaded_file):
        with st.spinner("Unzipping and sorting MRI sequences... Please wait."):
            with zipfile.ZipFile(uploaded_file) as z:
                for file_info in z.infolist():
                    if not file_info.is_dir():
                        try:
                            with z.open(file_info) as f:
                                bytes_data = f.read()
                                dcm = pydicom.dcmread(io.BytesIO(bytes_data), force=True)
                                
                                if hasattr(dcm, 'pixel_array'):
                                    # Group by Series Description (e.g., T2, FLAIR). Fallback to Series Instance UID if blank.
                                    series_name = getattr(dcm, 'SeriesDescription', None)
                                    if not series_name or str(series_name).strip() == "":
                                        series_name = f"Series {getattr(dcm, 'SeriesNumber', 'Unknown')}"
                                    
                                    # Get slice sorting number
                                    instance_num = int(getattr(dcm, 'InstanceNumber', 0))
                                    
                                    if series_name not in series_groups:
                                        series_groups[series_name] = []
                                    series_groups[series_name].append((instance_num, dcm))
                        except Exception:
                            continue
    else:
        st.error("Please upload a folder packed as a .zip file.")

    if series_groups:
        st.success(f"Successfully loaded {len(series_groups)} MRI sequence(s)!")
        
        # 1. FREELY SELECT THE MRI SEQUENCE / SERIES
        selected_series = st.selectbox("🎯 Select MRI Sequence:", list(series_groups.keys()))
        
        # Sort the slices of the chosen sequence in correct order (1, 2, 3...)
        current_series_slices = sorted(series_groups[selected_series], key=lambda x: x[0])
        total_slices = len(current_series_slices)
        
        # 2. SCROLL THROUGH IMAGES (SLICE SELECTOR)
        if total_slices > 1:
            slice_index = st.slider("🎞️ Scroll Through Slices:", 1, total_slices, 1) - 1
        else:
            slice_index = 0
            st.info("This sequence only contains a single image slice.")
            
        # Get the specific slice dataset chosen by the sliders
        _, dicom_data = current_series_slices[slice_index]
        pixel_array = dicom_data.pixel_array.astype(float)
        
        # 3. CONTRAST ADJUSTMENT SLIDERS
        st.subheader("🎛️ Image Controls")
        col_c1, col_c2 = st.columns(2)
        
        # Use medical Windowing defaults if available, otherwise calculate fallback ranges
        default_window = float(dicom_data.WindowWidth[0] if isinstance(getattr(dicom_data, 'WindowWidth', None), (list, pydicom.multival.MultiValue)) else getattr(dicom_data, 'WindowWidth', pixel_array.max() - pixel_array.min()))
        default_level = float(dicom_data.WindowCenter[0] if isinstance(getattr(dicom_data, 'WindowCenter', None), (list, pydicom.multival.MultiValue)) else getattr(dicom_data, 'WindowCenter', (pixel_array.max() + pixel_array.min()) / 2))
        
        with col_c1:
            window_width = st.slider("Contrast (Window Width):", 1.0, float(pixel_array.max()*1.5), float(default_window))
        with col_c2:
            window_level = st.slider("Brightness (Window Level):", float(pixel_array.min()), float(pixel_array.max()), float(default_level))
            
        # Apply the Contrast Window/Level calculations
        lower_bound = window_level - (window_width / 2.0)
        upper_bound = window_level + (window_width / 2.0)
        
        # Clip values outside our window and scale to normal image range (0 - 255)
        pixel_array = np.clip(pixel_array, lower_bound, upper_bound)
        if upper_bound != lower_bound:
            scaled_image = ((pixel_array - lower_bound) / (upper_bound - lower_bound)) * 255.0
        else:
            scaled_image = np.zeros(pixel_array.shape)
            
        final_image = Image.fromarray(scaled_image.astype(np.uint8))
        
        # Display the final interactive image
        st.image(final_image, caption=f"{selected_series} — Slice {slice_index + 1} of {total_slices}", use_container_width=True)
        
        # Patient Data Table
        st.subheader("📋 Metadata Info")
        st.text(f"Patient Name: {getattr(dicom_data, 'PatientName', 'Anonymous')}\nModality: {getattr(dicom_data, 'Modality', 'MR')}")
    else:
        if uploaded_file is not None:
            st.error("No valid DICOM sequences with scan images found in this ZIP archive.")