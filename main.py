import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
import zipfile
import io
import json
import base64

# Force wide-screen professional diagnostic layout
st.set_page_config(page_title="AI Glassmorphic MRI Workstation", page_icon="🧠", layout="wide")

# Inject premium dark theme with frosted glassmorphism elements
st.html("""
    <style>
    body, .stApp { 
        background: linear-gradient(135deg, #060913 0%, #0b1121 100%) !important; 
        color: #f1f5f9 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    [data-testid="stSidebar"] { 
        background: rgba(15, 23, 42, 0.45) !important; 
        backdrop-filter: blur(16px) !important; 
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important; 
    }
    .stMetric, .stToggle, div[data-testid="stFileUploader"] { 
        background: rgba(255, 255, 255, 0.03) !important; 
        backdrop-filter: blur(12px) !important; 
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 14px !important; 
        border-radius: 12px !important; 
        border: 1px solid rgba(255, 255, 255, 0.07) !important; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
    }
    h1, h2, h3 { 
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600 !important; 
    }
    </style>
""")

st.title("🧠 AI Glassmorphic MRI Diagnostics Workstation")
st.write("Upload a patient `.zip` directory containing multiple MRI sequences.")

# Master Toggles inside the Left Dashboard Sidebar Panel
st.sidebar.markdown("### ⚡ Workstation Controls")
ai_mode = st.sidebar.toggle("✨ AI Super-Resolution & Correction Mode", value=False, help="Activates Bias Field Uniformity Correction, Anisotropic Gibbs Ringing Suppression, and Laplacian Structural De-noising.")

uploaded_file = st.sidebar.file_uploader("Upload Patient ZIP Folder", type=["zip"])

def apply_ai_corrections(pixels):
    img_data = pixels.astype(float)
    min_val = img_data.min()
    img_data_shifted = img_data - min_val + 1.0
    
    # 1. BIAS FIELD SHADING CORRECTION
    log_img = np.log(img_data_shifted)
    low_freq = ndimage.gaussian_filter(log_img, sigma=max(img_data.shape) / 6.0)
    bias_corrected = np.exp(log_img - low_freq)
    
    # 2. GIBBS TRUNCATION RINGING MITIGATION
    smooth_gradient = ndimage.gaussian_filter(bias_corrected, sigma=1.2)
    diff = bias_corrected - smooth_gradient
    g_suppressed = bias_corrected - (diff * 0.45)
    
    # 3. LAPLACIAN STRUCTURAL DETAIL ENHANCEMENT
    laplacian = ndimage.laplace(g_suppressed)
    sharpened = g_suppressed - (0.35 * laplacian)
    
    final_pixels = np.clip(sharpened, 0, None)
    return final_pixels

if uploaded_file is not None:
    series_groups = {}

    with st.spinner("Executing deep directory extraction and unpacking sequences..."):
        with zipfile.ZipFile(uploaded_file) as z:
            for file_info in z.infolist():
                if not file_info.is_dir() and not file_info.filename.startswith('__MACOSX'):
                    try:
                        with z.open(file_info) as f:
                            bytes_data = f.read()
                            if len(bytes_data) < 132: 
                                continue
                                
                            dcm = pydicom.dcmread(io.BytesIO(bytes_data), force=True)
                            
                            if hasattr(dcm, 'pixel_array'):
                                series_name = getattr(dcm, 'SeriesDescription', None)
                                if not series_name or str(series_name).strip() == "":
                                    series_name = getattr(dcm, 'ProtocolName', None)
                                if not series_name or str(series_name).strip() == "":
                                    s_num = getattr(dcm, 'SeriesNumber', 'Unknown')
                                    series_name = f"Sequence {s_num}"
                                
                                series_name = str(series_name).strip()
                                instance_num = int(getattr(dcm, 'InstanceNumber', 0))
                                
                                if series_name not in series_groups:
                                    series_groups[series_name] = []
                                series_groups[series_name].append((instance_num, dcm))
                    except Exception:
                        continue

    if len(series_groups) > 0:
        all_series_names = sorted(list(series_groups.keys()))
        
        st.sidebar.markdown("### 📁 Sequence Directory")
        selected_sequences = st.sidebar.multiselect(
            "Select sequences to display (Up to 6 views):",
            options=all_series_names,
            default=all_series_names[:min(len(all_series_names), 3)]
        )
        
        active_sequences = selected_sequences[:6]
        num_series = len(active_sequences)
        
        if num_series == 0:
            st.warning("Please check at least one sequence from the left directory to display the diagnostics console.")
        else:
            p_name, p_id, s_date, s_man = "Anonymous", "N/A", "N/A", "N/A"
            px_spacing_x, px_spacing_y = 1.0, 1.0
            try:
                first_key = active_sequences[0]
                first_slices = series_groups[first_key]
                if len(first_slices) > 0:
                    _, sample_dcm = first_slices[0]
                    p_name = str(getattr(sample_dcm, 'PatientName', 'Anonymous'))
                    p_id = str(getattr(sample_dcm, 'PatientID', 'N/A'))
                    s_date = str(getattr(sample_dcm, 'StudyDate', 'N/A'))
                    s_man = str(getattr(sample_dcm, 'Manufacturer', 'N/A'))
                    
                    spacing = getattr(sample_dcm, 'PixelSpacing', [1.0, 1.0])
                    px_spacing_x = float(spacing[0])
                    px_spacing_y = float(spacing[1])
            except Exception:
                pass
                
            st.sidebar.markdown("### 📋 Patient Demographics")
            st.sidebar.text(f"Name: {p_name}")
            st.sidebar.text(f"ID: {p_id}")
            st.sidebar.text(f"Date: {s_date}")
            st.sidebar.text(f"Scanner: {s_man}")

            grid_data = []
            
            for i in range(num_series):
                s_name = active_sequences[i]
                sorted_slices = sorted(series_groups[s_name], key=lambda x: x[0])
                
                encoded_slices = []
                for _, dcm in sorted_slices:
                    raw_pixels = dcm.pixel_array.astype(float)
                    
                    if ai_mode:
                        pixels = apply_ai_corrections(raw_pixels)
                    else:
                        pixels = raw_pixels
                    
                    try:
                        raw_w = getattr(dcm, 'WindowWidth', pixels.max() - pixels.min())
                        dw = float(raw_w[0] if isinstance(raw_w, (list, pydicom.multival.MultiValue)) else raw_w)
                        raw_l = getattr(dcm, 'WindowCenter', (pixels.max() + pixels.min()) / 2)
                        dl = float(raw_l[0] if isinstance(raw_l, (list, pydicom.multival.MultiValue)) else raw_l)
                    except Exception:
                        dw = float(pixels.max() - pixels.min())
                        dl = float((pixels.max() + pixels.min()) / 2)
                    
                    if pixels.max() != pixels.min():
                        norm = ((pixels - pixels.min()) / (pixels.max() - pixels.min())) * 255.0
                    else:
                        norm = np.zeros(pixels.shape)
                        
                    img = Image.fromarray(norm.astype(np.uint8))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=90)
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
            window_height = 460 * rows_count + 120

            panels_html = ""
            for idx in range(num_series):
                panels_html += f"""
                <div id="panel_{idx}" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px; text-align: center; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.21);">
                    <div id="title_{idx}" style="color: #00f2fe; font-weight: bold; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Loading Viewport...</div>
                    <canvas id="canvas_{idx}" style="background: #000; max-width: 100%; width: 100%; height: auto; border: 1px solid rgba(255,255,255,0.05); cursor: crosshair; border-radius: 8px;"></canvas>
                    <div id="status_{idx}" style="color: #94a3b8; font-size: 11px; margin-top: 8px; font-family: monospace;">Slice: --</div>
                </div>
                """

            html_code = """
            <div style="background: transparent; padding: 5px; font-family: -apple-system, sans-serif; color: #fff;">
                <div style="background: rgba(255, 255, 255, 0.04); padding: 12px; margin-bottom: 20px; border-radius: 12px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.08); color: #cbd5e1; font-size:13px; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);">
                    ⚙️ <b>PACS Spatial Console Active</b> | <b>Scroll:</b> Flip Slices | <b>Left-Click & Drag:</b> Independent Contrast | 📏 <b>Shift + Click & Drag:</b> Measurement Caliper
                </div>
                <div style="display: grid; grid-template-columns: repeat(""" + str(cols_count) + """, 1fr); gap: 20px;">
                    """ + panels_html + """
                </div>
            </div>
            <script>
                const workstationData = """ + workstation_json + """;
                const numPanels = workstationData.length;
                const pSpacingX = """ + str(px_spacing_x) + """;
                const pSpacingY = """ + str(px_spacing_y) + """;
                
                let currentIndices = new Array(numPanels).fill(0);
                let windowWidths = workstationData.map(p => p.slices && p.slices.length > 0 ? p.slices[0].default_w : 1000);
                let windowLevels = workstationData.map(p => p.slices && p.slices.length > 0 ? p.slices[0].default_l : 500);
                
                let isDragging = false;
                let isCaliperActive = false;
                let activePanelIdx = null;
                let startX, startY;
                let caliperStartX, caliperStartY;
                let caliperEndX, caliperEndY;

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
                        ctx.save();
                        ctx.filter = "brightness(" + (bPct + 50) + "%) contrast(" + cPct + "%)";
                        ctx.drawImage(imgObj, 0, 0);
                        ctx.restore();
                        
                        // DRAW MEASUREMENT CALIPER IF ACTIVE
                        if (isCaliperActive && activePanelIdx === idx) {
                            ctx.strokeStyle = '#00ffcc';
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(caliperStartX, caliperStartY);
                            ctx.lineTo(caliperEndX, caliperEndY);
                            ctx.stroke();
                            
                            // Draw end ticks
                            ctx.fillStyle = '#00ffcc';
                            ctx.fillRect(caliperStartX - 3, caliperStartY - 3, 6, 6);
                            ctx.fillRect(caliperEndX - 3, caliperEndY - 3, 6, 6);
                            
                            // Calculate actual spatial distance
                            let dxMm = (caliperEndX - caliperStartX) * pSpacingX;
                            let dyMm = (caliperEndY - caliperStartY) * pSpacingY;
                            let distMm = Math.sqrt(dxMm * dxMm + dyMm * dyMm);
                            
                            ctx.font = 'bold 14px monospace';
                            ctx.fillStyle = '#00ffcc';
                            ctx.shadowColor = 'black';
                            ctx.shadowBlur = 4;
                            ctx.fillText(distMm.toFixed(2) + " mm", caliperEndX + 10, caliperEndY + 10);
                        }
                        
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
                        const rect = canvas.getBoundingClientRect();
                        const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
                        const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
                        
                        activePanelIdx = panelIdx;
                        startX = e.clientX;
                        startY = e.clientY;
                        
                        if (e.shiftKey) {
                            isCaliperActive = true;
                            caliperStartX = canvasX;
                            caliperStartY = canvasY;
                            caliperEndX = canvasX;
                            caliperEndY = canvasY;
                        } else {
                            isDragging = true;
                            isCaliperActive = false;
                        }
                    });
                }

                window.addEventListener('mousemove', function(e) {
                    if (activePanelIdx === null) return;
                    
                    const canvas = document.getElementById('canvas_' + activePanelIdx);
                    const rect = canvas.getBoundingClientRect();
                    const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
                    const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

                    if (isCaliperActive) {
                        caliperEndX = canvasX;
                        caliperEndY = canvasY;
                        renderPanel(activePanelIdx);
                    } else if (isDragging) {
                        let deltaX = e.clientX - startX;
                        let deltaY = e.clientY - startY;
                        
                        startX = e.clientX;
                        startY = e.clientY;

                        windowWidths[activePanelIdx] += deltaX * 3;
                        windowLevels[activePanelIdx] -= deltaY * 3;
                        
                        renderPanel(activePanelIdx);
                    }
                });

                window.addEventListener('mouseup', function() {
                    isDragging = false;
                    // Keep caliper trace visible until next click
                });
            </script>
            """
            st.components.v1.html(html_code, height=window_height)
    else:
        st.error("No valid multi-sequence folders containing imaging tensors discovered inside this ZIP.")
