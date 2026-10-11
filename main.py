import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import zipfile
import io

st.set_page_config(page_title="MRI DICOM Viewer", page_icon="🏥", layout="centered")
st.title("🏥 MRI DICOM Viewer")
st.write("Upload an individual DICOM file or a **.zip folder** containing images.")

# Accept any uploaded file format
uploaded_file = st.file_uploader("Choose a DICOM file or a ZIP archive")

if uploaded_file is not None:
    dicom_files = {}

    # Check if the uploaded file is a ZIP folder
    if zipfile.is_zipfile(uploaded_file):
        with zipfile.ZipFile(uploaded_file) as z:
            # Go through every file inside the zip archive
            for file_info in z.infolist():
                # Ignore folder names, look only at files
                if not file_info.is_dir():
                    try:
                        with z.open(file_info) as f:
                            # Read bytes into memory
                            bytes_data = f.read()
                            # Use force=True to read files even if they lack the standard header
                            dcm = pydicom.dcmread(io.BytesIO(bytes_data), force=True)
                            
                            # If it looks like a valid DICOM image with pixel data, save it
                            if hasattr(dcm, 'pixel_array'):
                                dicom_files[file_info.filename] = dcm
                    except Exception:
                        continue
    else:
        # If it's a single file instead of a zip
        try:
            dcm = pydicom.dcmread(uploaded_file, force=True)
            if hasattr(dcm, 'pixel_array'):
                dicom_files[uploaded_file.name] = dcm
        except Exception as e:
            st.error(f"Could not read this file: {e}")

    # Process and display found images
    if dicom_files:
        st.success(f"Successfully found {len(dicom_files)} medical image(s)!")
        
        # Create a dropdown menu if multiple files exist
        selected_filename = st.selectbox("Select an image to view:", list(dicom_files.keys()))
        dicom_data = dicom_files[selected_filename]

        try:
            # Convert raw pixel data to viewable image
            pixel_array = dicom_data.pixel_array.astype(float)
            
            # Safe normalization to prevent math errors
            if pixel_array.max() == pixel_array.min():
                scaled_image = np.zeros(pixel_array.shape)
            else:
                scaled_image = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0
                
            final_image = Image.fromarray(scaled_image.astype(np.uint8))
            
            # Display image
            st.image(final_image, caption=f"Viewing: {selected_filename}", use_container_width=True)
            
            # Display patient details
            st.subheader("📋 Patient & Scan Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Patient Name:** {getattr(dicom_data, 'PatientName', 'Anonymous')}")
                st.write(f"**Patient ID:** {getattr(dicom_data, 'PatientID', 'N/A')}")
            with col2:
                st.write(f"**Modality:** {getattr(dicom_data, 'Modality', 'N/A')}")
                st.write(f"**Study Date:** {getattr(dicom_data, 'StudyDate', 'N/A')}")
                
        except Exception as e:
            st.error(f"Error displaying image data: {e}")
    else:
        st.error("No viewable DICOM images found inside the uploaded file. Ensure they contain pixel data.")