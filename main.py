import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import zipfile
import io
import json
import base64

st.set_page_config(page_title="Pro MRI DICOM Viewer", page_icon="🧠", layout="wide")

# FIX: Explicitly name the 'body' argument to stop the TypeError crash
st.markdown(
    body="""
    <style>
    .reportview-container { background: #111111; color: white; }
    .sidebar .sidebar-content { background: #1e1e1e; }
    div.stButton > button:first-child { width: 100%; }
    </style>
    """, 
    unsafe_allowed_html=True
)

st.title("🧠 Pro Medical MRI DICOM Viewer")

uploaded_file = st.file_uploader("Upload a .zip folder containing patient MRI series", type=["zip"])

if uploaded_file is not None:
    series_groups = {}

    with st.spinner("Processing MRI datasets... Please wait."):
        with zipfile.ZipFile(uploaded_file) as z:
            for file_info in z.infolist():
                if not file_info.is_dir():
                    try:
                        with z.open(file_info) as f:
                            bytes_data = f.read()
                            dcm = pydicom.dcmread(io.BytesIO(bytes_data), force=True)
                            
                            if hasattr(dcm, 'pixel_array'):
                                series_name = getattr(dcm, 'SeriesDescription', None)
                                if not series_name or str(series_name).strip() == "":
                                    series_name = f"Sequence {getattr(dcm, 'SeriesNumber', 'Unknown')}"
                                
                                instance_num = int(getattr(dcm, 'InstanceNumber', 0))
                                
                                if series_name not in series_groups:
                                    series_groups[series_name] = []
                                series_groups[series_name].append((instance_num, dcm))
                    except Exception:
                        continue

    if series_groups:
        # 1. SIDEBAR SEQUENCE LISTING (Left Hand Side Selection)
        st.sidebar.header("📁 Patient Sequences")
        selected_series = st.sidebar.radio(
            "Select MRI Sequence to View:", 
            list(series_groups.keys())
        )
        
        # Sort slices cleanly
        current_slices = sorted(series_groups[selected_series], key=lambda x: x[0])
        total_slices = len(current_slices)
        
        st.sidebar.metric(label="Total Slices", value=total_slices)
        
        # Extract metadata from the FIRST slice properly
        _, sample_dcm = current_slices[0]
        st.sidebar.subheader("📋 Patient Metadata")
        st.sidebar.write(f"**Name:** {getattr(sample_dcm, 'PatientName', 'Anonymous')}")
        st.sidebar.write(f"**ID:** {getattr(sample_dcm, 'PatientID', 'N/A')}")
        st.sidebar.write(f"**Date:** {getattr(sample_dcm, 'StudyDate', 'N/A')}")
        
        # 2. CONVERT ALL SLICES TO HIGH-SPEED IMAGES FOR THE BROWSER
        encoded_slices = []
        for _, dcm in current_slices:
            pixels = dcm.pixel_array.astype(float)
            
            # Default windowing fallbacks
            dw = float(dcm.WindowWidth if isinstance(getattr(dcm, 'WindowWidth', None), (list, pydicom.multival.MultiValue)) else getattr(dcm, 'WindowWidth', pixels.max() - pixels.min()))
            dl = float(dcm.WindowCenter if isinstance(getattr(dcm, 'WindowCenter', None), (list, pydicom.multival.MultiValue)) else getattr(dcm, 'WindowCenter', (pixels.max() + pixels.min()) / 2))
            
            # Normalize to 0-255 initially
            if pixels.max() != pixels.min():
                norm = ((pixels - pixels.min()) / (pixels.max() - pixels.min())) * 255.0
            else:
                norm = np.zeros(pixels.shape)
                
            img = Image.fromarray(norm.astype(np.uint8))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            encoded_slices.append({
                "base64": f"data:image/jpeg;base64,{img_str}",
                "raw_min": float(pixels.min()),
                "raw_max": float(pixels.max()),
                "default_w": dw,
                "default_l": dl
            })

        # 3. INTERACTIVE MOUSE CONTROLLER ENGINE
        slices_json = json.dumps(encoded_slices)
        
        html_code = f"""
        <div style="background-color: #000; padding: 15px; border-radius: 8px; color: #fff; font-family: sans-serif; text-align: center;">
            <p style="margin: 0 0 10px 0; font-size: 14px; color: #aaa;">
                🖱️ <b>Mouse Wheel:</b> Scroll Slices | ↕️ <b>Left Click & Drag Vertically:</b> Brightness | ↔️ <b>Horizontal Drag:</b> Contrast
            </p>
            <canvas id="dicomCanvas" style="border: 2px solid #333; max-width: 100%; background: #000; cursor: contrast;"></canvas>
            <div id="statusInfo" style="margin-top: 10px; font-weight: bold; font-size: 15px; color: #4CAF50;">Slice: 1 / {total_slices}</div>
        </div>

        <script>
            const slices = {slices_json};
            let currentIdx = 0;
            let total = {total_slices};
            
            const canvas = document.getElementById('dicomCanvas');
            const ctx = canvas.getContext('2d');
            const statusInfo = document.getElementById('statusInfo');
            
            let windowWidth = slices[currentIdx].default_w;
            let windowLevel = slices[currentIdx].default_l;
            let isDragging = false;
            let startX, startY;

            const imgObj = new Image();
            function renderSlice() {{
                imgObj.src = slices[currentIdx].base64;
                imgObj.onload = function() {{
                    canvas.width = imgObj.width || 512;
                    canvas.height = imgObj.height || 512;
                    
                    let brightnessPct = (windowLevel / (slices[currentIdx].raw_max || 255)) * 100;
                    let contrastPct = (255 / (windowWidth || 1)) * 100;
                    
                    ctx.clearRect(0,0,canvas.width,canvas.height);
                    ctx.filter = `brightness(${{brightnessPct + 50}}%) contrast(${{contrastPct}}%)`;
                    ctx.drawImage(imgObj, 0, 0);
                    
                    statusInfo.innerText = `Slice: ${{currentIdx + 1}} / ${{total}} | W: ${{Math.round(windowWidth)}} L: ${{Math.round(windowLevel)}}`;
                }};
            }}
            
            canvas.addEventListener('wheel', function(e) {{
                e.preventDefault();
                if (e.deltaY > 0) {{
                    currentIdx = (currentIdx + 1) % total;
                }} else {{
                    currentIdx = (currentIdx - 1 + total) % total;
                }}
                renderSlice();
            }});

            canvas.addEventListener('mousedown', function(e) {{
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
            }});

            window.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;
                let deltaX = e.clientX - startX;
                let deltaY = e.clientY - startY;
                
                startX = e.clientX;
                startY = e.clientY;

                windowWidth += deltaX * 2;
                windowLevel -= deltaY * 2;
                
                renderSlice();
            }});

            window.addEventListener('mouseup', function() {{
                isDragging = false;
            }});
            
            renderSlice();
        </script>
        """
        
        st.components.v1.html(html_code, height=620)
        
    else:
        st.error("No valid medical imaging scans found in this ZIP archive.")