import streamlit as st
import pydicom
from PIL import Image
import numpy as np

# 1. Create a friendly title on your website
st.set_page_config(page_title="MRI DICOM Viewer", page_icon="🏥", layout="centered")
st.title("🏥 MRI DICOM Viewer")
st.write("Upload a `.dcm` file below to view the medical image.")

# 2. Create a file uploader button
uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm"])

if uploaded_file is not None:
    try:
        # 3. Read the medical image using pydicom
        dicom_data = pydicom.dcmread(uploaded_file)
        
        # 4. Convert the raw pixel data into a viewable image format
        pixel_array = dicom_data.pixel_array
        
        # Normalize the pixel data so it displays correctly
        pixel_array = pixel_array.astype(float)
        scaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        final_image = Image.fromarray(scaled_image.astype(np.uint8))
        
        # 5. Display the image on your web page
        st.image(final_image, caption="Uploaded DICOM Image", use_container_width=True)
        
        # Display some information about the patient if available
        st.subheader("📋 Patient & Scan Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Patient Name:** {getattr(dicom_data, 'PatientName', 'Anonymous')}")
            st.write(f"**Patient ID:** {getattr(dicom_data, 'PatientID', 'N/A')}")
        with col2:
            st.write(f"**Modality:** {getattr(dicom_data, 'Modality', 'N/A')}")
            st.write(f"**Study Date:** {getattr(dicom_data, 'StudyDate', 'N/A')}")

    except Exception as e:
        st.error(f"Could not read this DICOM file: {e}")