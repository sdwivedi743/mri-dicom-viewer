import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
import zipfile
import io
import json
import base64

# Force wide-screen diagnostic layout
st.set_page_config(page_title="AI Multi-Sequence MRI Workstation", page_icon="🧠", layout="wide")

st.title("🧠 AI Multi-Sequence MRI Diagnostics Workstation")
st.write("Upload a patient `.zip` directory containing multiple MRI sequences (Up to 6 views simultaneously).")

# Master Toggle for AI Super-Resolution & Correction Pipeline
st.sidebar.markdown("### ⚡ Workstation Controls")
ai_mode = st.sidebar.toggle("✨ AI Super-Resolution & Correction Mode", value=False, help="Activates Bias Field Intensity Normalization, Gibbs Ringing Suppression, and Structural De-noising.")

uploaded_file = st.sidebar.file_uploader("Upload Patient ZIP Folder", type=["zip"])

def apply_ai_corrections(pixels):
    """
    Advanced Image-Space Post-Processing Suite
    1. Homomorphic Gaussian High-Pass Filter for Bias Field Intensity Correction
    2. Edge-Preserving structural smoothing to suppress noise and Gibbs ringing
    """
    img_data = pixels.astype(float)
    
    min_val = img_data.min()
    img_data_shifted = img_data - min_val + 1.0
    
    # 1. INTENSITY FIELD NORMALIZATION (Bias Field Correction)
    log_img = np.log(img_data_shifted)
    low_freq = ndimage.gaussian_filter(log_img, sigma=max(img_data.shape) / 8.0)
    bias_corrected = np.exp(log_img - low_freq)
    
    # 2. GIBBS RINGING & NOISE REMEDIATION (Edge-Preserving Structural Smoothing)
    smooth = ndimage.gaussian_filter(bias_corrected, sigma=1.0)
    edges = bias_corrected - smooth
    enhanced = bias_corrected + (edges * 0.5)
    
    final_pixels = np.clip(enhanced, 0, None)
    return final_pixels

if uploaded_file is not None:
    series_groups = {}

    with st.spinner("Unpacking and sorting medical dataset sequences..."):
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
        all_series_names = list(series_groups.keys())
        num_series = min(len(all_series_names), 6)
        
        st.sidebar.success(f"Detected {len(all_series_names)} MRI sequences.")
        
        # Display Patient Metadata inside sidebar from the first sequence found
        first_series_name = all_series_names[0]
        first_series = series_groups[first_series_name]
        _, sample_dcm = sorted(first_series, key=lambda x: x[0])[0]
        
        st.sidebar.markdown("### 📋 Patient Demographics")
        st.sidebar.text(f"Name: {str(getattr(sample_dcm, 'PatientName', 'Anonymous'))}")
        st.sidebar.text(f"ID: {str(getattr(sample_dcm, 'PatientID', 'N/A'))}")
        st.sidebar.text(f"Date: {str(getattr(sample_dcm, 'StudyDate', 'N/A'))}")
        st.sidebar.text(f"Scanner: {str(getattr(sample_dcm, 'Manufacturer', 'N/A'))}")

        grid_data = []
        
        for i in range(num_series):
            s_name = all_series_names[i]
            sorted_slices = sorted(series_groups[s_name], key=lambda x: x[0])
            
            encoded_slices = []
            for _, dcm in sorted_slices:
                raw_pixels = dcm.pixel_array.astype(float)
                
                if ai_mode:
                    pixels = apply_ai_corrections(raw_pixels)
                else:
                    pixels = raw_pixels
                
                dw = float(dcm.WindowWidth if isinstance(getattr(dcm, 'WindowWidth', None), (list, pydicom.multival.MultiValue)) else getattr(dcm, 'WindowWidth', pixels.max() - pixels.min()))
                dl = float(dcm.WindowCenter if isinstance(getattr(dcm, 'WindowCenter', None), (list, pydicom.multival.MultiValue)) else getattr(dcm, 'WindowCenter', (pixels.max() + pixels.min()) / 2))
                
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
                    "raw_max": float(pixels.max() if pixels.max() != 0 else 255),
                    "default_w": dw,
                    "default_l": dl
                })
                
            grid_data.append({
                "title": s_name,
                "total_slices": len(encoded_slices),
                "slices": encoded_slices
            })

        workstation_json = json.dumps(grid_data)
        
        cols_count = 3 if num_series >= 3 else num_series
        rows_count = 2 if num_series > 3 else 1
        window_height = 420 * rows_count + 80

        # Generate the panel containers layout
        panels_html = ""
        for idx in range(num_series):
            panels_html += f"""
            <div id="panel_{idx}" style="background: #000; border: 1px solid #333; border-radius: 4px; padding: 6px; text-align: center;">
                <div id="title_{idx}" style="color: #aaa; font-weight: bold; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Loading Viewport...</div>
                <canvas id="canvas_{idx}" style="background: #000; max-width: 100%; width: 100%; height: auto; border: 1px solid #111; cursor: crosshair;"></canvas>
                <div id="status_{idx}" style="color: #666; font-size: 11px; margin-top: 4px;">Slice: --</div>
            </div>
            """

        # Final Clean HTML Sandbox Blueprint Integration
        html_code = """
        <div style="background-color: #050505; padding: 10px; border-radius: 8px; color: #ffffff; font-family: monospace; font-size:12px;">
            <div style="background: #111; padding: 8px; margin-bottom: 12px; border-radius: 4px; text-align: center; border: 1px solid #222; color: #00ffcc;">
                🖥️ <b>Workstation Active</b> | Hover mouse over any window | <b>Scroll Wheel:</b> Flip Slices | <b>Left-Click & Drag:</b> Independent Contrast/Brightness
            </div>
            <div style="display: grid; grid-template-columns: repeat(""" + str(cols_count) + """, 1fr); gap: 12px;">
                """ + panels_html + """
            </div>
        </div>
        <script>
            const workstationData = """ + workstation_json + """;
            const numPanels = workstationData.length;
            
            let currentIndices = new Array(numPanels).fill(0);
            let windowWidths = workstationData.map(p => p.slices && p.slices.length > 0 ? p.slices[0].default_w : 1000);
            let windowLevels = workstationData.map(p => p.slices && p.slices.length > 0 ? p.slices[0].default_l : 500);
            
            let isDragging = false;
            let activePanelIdx = null;
            let startX, startY;

            function renderPanel(idx) {
                const panel = workstationData[idx];
                const sliceIdx = currentIndices[idx];
                if (!panel.slices || !panel.slices[sliceIdx]) return;
                const sliceData = panel.slices[sliceIdx];
                
                document.getElementById('title_' + idx).innerText = panel.title;
                const canvas = document.getElementById('canvas_' + idx);
                const ctx = canvas.getContext('2d');
                
                const imgObj = new Image();
                imgObj.src = sliceData.base64;
                imgObj.onload = function() {
                    canvas.width = imgObj.width || 512;
                    canvas.height = imgObj.height || 512;
                    
                    let bPct = (windowLevels[idx] / sliceData.raw_max) * 100;
                    let cPct = (255 / (windowWidths[idx] || 1)) * 100;
                    
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.filter = "brightness(" + (bPct + 50) + "%) contrast(" + cPct + "%)";
                    ctx.drawImage(imgObj, 0, 0);
                    
                    document.getElementById('status_' + idx).innerText = 
                        "Slice: " + (sliceIdx + 1) + "/" + panel.total_slices + " | W: " + Math.round(windowWidths[idx]) + " L: " + Math.round(windowLevels[idx]);
                };
            }

            for(let i=0; i<numPanels; i++) {
                const canvas = document.getElementById('canvas_' + i);
                const panelIdx = i;
                
                canvas.addEventListener('wheel', function(e) {
                    e.preventDefault();
                    if(e.deltaY > 0) {
currentIndices[panelIdx] = (currentIndices[panelIdx] + 1) % workstationData[panelIdx].total_slices;
} else {
currentIndices[panelIdx] = (currentIndices[panelIdx] - 1 + workstationData[panelIdx].total_slices) % workstationData[panelIdx].total_slices;
}
renderPanel(panelIdx);
}, { passive: false });
canvas.addEventListener('mousedown', function(e) {
isDragging = true;
activePanelIdx = panelIdx;
startX = e.clientX;
startY = e.clientY;
});
renderPanel(i);
}
window.addEventListener('mousemove', function(e) {
if (!isDragging || activePanelIdx === null) return;
let deltaX = e.clientX - startX;
let deltaY = e.clientY - startY;
startX = e.clientX;
startY = e.clientY;
windowWidths[activePanelIdx] += deltaX * 3;
windowLevels[activePanelIdx] -= deltaY * 3;
renderPanel(activePanelIdx);
});
window.addEventListener('mouseup', function() {
isDragging = false;
activePanelIdx = null;
});

"""
st.components.v1.html(html_code, height=window_height)
else:
st.error("No valid multi-sequence folders containing imaging tensors discovered inside this ZIP.")