"""
SD-DCM Clinical Workstation  -  Premium Medical Imaging Command Center
=====================================================================================================
Built according to international radiology workstation standards and clinical UX guidelines.

Run:
    pip install streamlit pydicom numpy scipy
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
    streamlit run main.py
"""

import base64
import hashlib
import io
import json
import sys
import zipfile
import zlib
from collections import Counter

import numpy as np
from scipy import ndimage as ndi

try:
    import streamlit as st
except ImportError:
    st = None

MAX_PANELS = 6

PRESETS = {
    "Natural":  dict(denoise=25, detail=75, sharpen=60, grain=15, shading=0),
    "Balanced": dict(denoise=35, detail=65, sharpen=55, grain=12, shading=0),
    "Strong":   dict(denoise=60, detail=45, sharpen=45, grain=6,  shading=0),
}

# ----------------------------------------------------------------------------------------------------------------
#  Core Image-Processing & Enhancement Routines (Preserved with absolute data integrity)
# ----------------------------------------------------------------------------------------------------------------
def estimate_noise(img):
    img = np.asarray(img, np.float32)
    mn, mx = float(img.min()), float(img.max())
    out = dict(sigma=0.0, fg=mx, p99=mx, bg_thr=0.0, method="none")
    if mx <= mn or img.shape[0] < 8 or img.shape[1] < 8:
        return out
    B = 4096
    h, _ = np.histogram(img, bins=B, range=(mn, mx))
    cdf = np.cumsum(h)
    p99 = mn + (np.searchsorted(cdf, 0.99 * cdf[-1]) + 1) * (mx - mn) / (B - 1)
    rng = p99 - mn
    lim = min(B - 1, int(rng * 0.2 / (mx - mn) * (B - 1)) + 1)
    sm = np.convolve(h[:lim + 1].astype(float), np.ones(13) / 13.0, mode="same")
    pk = int(np.argmax(sm))
    low_frac = h[:lim + 1].sum() / img.size
    sigma, method, bg_thr = 0.0, "mode", 0.0
    if low_frac > 0.05 and sm[pk] > 0:
        mode_off = pk * (mx - mn) / (B - 1)
        bg_thr = mn + mode_off * 3.2
        vals = img[img <= bg_thr]
        if vals.size:
            sigma = float(np.median(vals)) / 1.1774
            if not sigma > rng * 2e-4:
                sigma = 0.0
    if sigma == 0.0:
        method = "highpass"
        k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
        r = np.abs(ndi.convolve(img, k, mode="nearest"))[1:-1, 1:-1].ravel()
        sigma = float(np.quantile(r, 0.25)) / (0.3186 * 6) * 1.8
    fg_vals = img[img > mn + 0.15 * rng]
    out.update(sigma=sigma * (0.8 if method == "highpass" else 0.95),
               fg=float(np.median(fg_vals)) if fg_vals.size else mx, p99=float(p99), bg_thr=float(bg_thr), method=method)
    return out

_OFFS = [(1, 0), (0, 1), (2, 1), (3, 0), (0, 3), (3, 3), (-2, 2), (1, 3)]

def patch_baseline(img, bg_thr):
    img = np.asarray(img, np.float32)
    bg = (img <= bg_thr).astype(np.float32)
    bg = ndi.uniform_filter(bg, 3) > 0.999
    vals = []
    for dx, dy in _OFFS:
        sh = np.roll(np.roll(img, -dy, 0), -dx, 1)
        shm = np.roll(np.roll(bg, -dy, 0), -dx, 1)
        m = ndi.uniform_filter((img - sh) ** 2, 3)
        mask = bg & shm
        mask[:4, :] = mask[-4:, :] = False
        mask[:, :4] = mask[:, -4:] = False
        if mask.sum() > 50:
            vals.append(float(m[mask].mean()))
    return float(np.mean(vals)) if vals else 0.0

def series_info(frames):
    n = len(frames)
    idxs = sorted({int(np.clip(round((i + 0.5) / min(5, n) * n - 0.5), 0, n - 1)) for i in range(min(5, n))})
    sg, pb, fg, his, los = [], [], [], [], []
    for i in idxs:
        f = frames[i]
        nz = estimate_noise(f)
        sg.append(nz["sigma"]); fg.append(nz["fg"])
        pb.append(patch_baseline(f, nz["bg_thr"]) if nz["method"] == "mode" else 0.0)
        p005 = float(np.percentile(f, 0.5)); p995 = float(np.percentile(f, 99.5))
        fgm = f > (f.min() + 0.08 * (p995 - f.min()))
        hi = float(np.percentile(f[fgm], 99.7)) if fgm.mean() >= 0.03 else p995
        los.append(p005); his.append(hi)
    sigma = float(np.median(sg)); fgv = float(np.median(fg))
    lo = min(0.0, float(min(los)))
    hi = max(float(np.median(his)) - 2 * sigma, lo + 1e-6)
    return dict(sigma=sigma, pb=float(np.median(pb)), lo=lo, hi=hi, snr=(fgv / sigma if sigma > 0 else 100.0))

def nlm25(slices, c, sigma, strength, patch_base=0.0, R=2, Rn=1):
    d = slices[c].astype(np.float32)
    H, W = d.shape
    h = sigma * (0.55 + 0.95 * strength)
    ih2 = 1.0 / (h * h)
    s2 = patch_base if patch_base > 0 else 2 * sigma * sigma
    bias = 2 * sigma * sigma
    acc = np.zeros_like(d); ws = np.zeros_like(d); wm = np.zeros_like(d)
    offs = [(c, dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dx or dy]
    for dz in (-1, 1):
        k = c + dz
        if 0 <= k < len(slices):
            offs += [(k, dx, dy) for dy in range(-Rn, Rn + 1) for dx in range(-Rn, Rn + 1)]
    for k, dx, dy in offs:
        o = slices[k].astype(np.float32)
        sh = np.roll(np.roll(o, -dy, 0), -dx, 1)
        valid = np.zeros((H, W), bool)
        valid[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = True
        e = np.where(valid, (d - sh) ** 2, 0.0)
        m = ndi.uniform_filter(e, 3, mode="constant")
        dd = np.maximum(m - s2, 0.0) * ih2
        w = np.where((dd < 16) & valid, np.exp(-dd), 0.0).astype(np.float32)
        acc += w * sh * sh; ws += w; np.maximum(wm, w, out=wm)
    den = ws + wm
    sq = d * d
    e2 = np.where(den > 0, (acc + wm * sq) / np.where(den > 0, den, 1), sq)
    return np.sqrt(np.maximum(e2 - bias, 0.0)).astype(np.float32)

def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def retain_detail(orig, den, sigma, alpha, grain):
    if sigma <= 0 or (alpha <= 0 and grain <= 0):
        return den
    gy, gx = np.gradient(den)
    g = np.hypot(gx, gy) / sigma
    sw = _smoothstep((g - 0.35) / 1.6)
    fg = _smoothstep((den - 1.5 * sigma) / (2.5 * sigma))
    k = np.minimum(fg * (alpha * sw + grain), 1.0)
    return den + k * (orig - den)

def bias_correct(d, strength, info):
    if strength <= 0:
        return d
    mask = (d > info["lo"] + 0.10 * (info["hi"] - info["lo"])).astype(np.float32)
    if mask.mean() < 0.05:
        return d
    s = max(d.shape) / 12.0
    num = ndi.gaussian_filter(d * mask, s); den = ndi.gaussian_filter(mask, s)
    est = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0)
    gm = float(d[mask > 0].mean())
    gain = np.where(est > 0, np.clip(gm / np.maximum(est, 1e-6), 0.6, 1.8), 1.0) ** strength
    return d * gain.astype(np.float32)

def sharpen(d, sigma, s, info):
    if s <= 0:
        return d
    rng = max(info["hi"] - info["lo"], 1e-6)
    t0 = max(1.2 * rng / 255.0, 0.225 * sigma)
    t1 = 3.5 * t0
    d1 = d - ndi.gaussian_filter(d, 0.75)
    d2 = d - ndi.gaussian_filter(d, 1.9)
    gate = _smoothstep((np.abs(d1) - t0) / (t1 - t0))
    o = d + gate * (1.1 * s * d1 + 0.55 * s * d2)
    o = np.clip(o, ndi.minimum_filter(d, 3), ndi.maximum_filter(d, 3))
    return np.where(d <= 0.01 * rng, d, o)

def enhance_frame(frames, k, info, p):
    d0 = frames[k].astype(np.float32)
    sg = info["sigma"]
    if sg <= 0:
        return d0
    a0 = max(0, k - 1)
    arr = frames[a0:k + 2]
    d = nlm25(arr, k - a0, sg, 0.2 + 1.2 * p["denoise"] / 100.0, info["pb"]) if p["denoise"] > 0 else d0
    d = bias_correct(d, p.get("shading", 0) / 100.0, info)
    d = retain_detail(d0, d, sg, p["detail"] / 100.0, p["grain"] / 100.0)
    d = sharpen(d, sg, p["sharpen"] / 100.0, info)
    return d.astype(np.float32)

def pack_series(orig_frames, enh_frames=None):
    mn = min(float(f.min()) for f in orig_frames)
    mx = max(float(f.max()) for f in orig_frames)
    if enh_frames:
        mn = min(mn, min(float(f.min()) for f in enh_frames))
        mx = max(mx, max(float(f.max()) for f in enh_frames))
    if mx <= mn:
        mx = mn + 1.0
    scale = (mx - mn) / 65535.0

    def enc(f):
        u = np.clip(np.rint((f - mn) / scale), 0, 65535).astype("<u2")
        return base64.b64encode(zlib.compress(u.tobytes(), 3)).decode("ascii")

    items = []
    for i, f in enumerate(orig_frames):
        it = {"o": enc(f)}
        if enh_frames:
            it["e"] = enc(enh_frames[i])
        items.append(it)
    return dict(off=mn, scale=scale, items=items)

def unpack_frame(b64, rows, cols, off, scale):
    u = np.frombuffer(zlib.decompress(base64.b64decode(b64)), "<u2").reshape(rows, cols)
    return off + u.astype(np.float32) * scale

# ----------------------------------------------------------------------------------------------------------------
#  DICOM Parsing & Grouping Mechanics (Preserved with strict structural safety)
# ----------------------------------------------------------------------------------------------------------------
def _first(v, default=None):
    try:
        if v is None:
            return default
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            return float(v[0]) if len(v) else default
        return float(v)
    except Exception:
        return default

def _floats(v, n):
    try:
        a = [float(x) for x in v]
        return a if len(a) == n else None
    except Exception:
        return None

def modality_frames(ds):
    arr = np.asarray(ds.pixel_array)
    nfr = int(getattr(ds, "NumberOfFrames", 1) or 1)
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if spp == 3:
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        elif arr.ndim == 4:
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    frames = list(arr) if (nfr > 1 and arr.ndim == 3) else [arr]
    slope = _first(getattr(ds, "RescaleSlope", None), 1.0) or 1.0
    icpt = _first(getattr(ds, "RescaleIntercept", None), 0.0) or 0.0
    return [np.asarray(f, np.float32) * slope + icpt for f in frames]

def plane_name(normal):
    k = int(np.argmax(np.abs(normal)))
    return ["Sagittal", "Coronal", "Axial"][k]

def geometry_of(ds):
    iop = _floats(getattr(ds, "ImageOrientationPatient", None) or [], 6)
    if iop:
        r, c = np.array(iop[:3]), np.array(iop[3:])
        n = np.cross(r, c)
    else:
        n = np.array([0.0, 0.0, 1.0])
    return n

def slice_position(ds, normal):
    ipp = _floats(getattr(ds, "ImagePositionPatient", None) or [], 3)
    return float(np.dot(ipp, normal)) if ipp else None

def group_series(named_datasets):
    seen, groups = set(), {}
    dup = 0
    for name, ds in named_datasets:
        sop = str(getattr(ds, "SOPInstanceUID", "") or "")
        if sop:
            if sop in seen:
                dup += 1
                continue
            seen.add(sop)
        key = (str(getattr(ds, "SeriesInstanceUID", "unknown")), int(ds.Rows), int(ds.Columns))
        groups.setdefault(key, []).append(ds)
    series = []
    for (uid, rows, cols), lst in groups.items():
        a = lst[0]
        desc = str(getattr(a, "SeriesDescription", "") or getattr(a, "ProtocolName", "") or "Series")
        num = int(_first(getattr(a, "SeriesNumber", None), 0) or 0)
        normal = geometry_of(a)
        keyed = []
        for ds in lst:
            pos = slice_position(ds, normal)
            keyed.append((pos if pos is not None else float("inf"), int(_first(getattr(ds, "InstanceNumber", None), 0) or 0), ds))
        if all(k[0] != float("inf") for k in keyed):
            keyed.sort(key=lambda t: (t[0], t[1]))
        else:
            keyed.sort(key=lambda t: t[1])
        series.append(dict(uid=uid, rows=rows, cols=cols, desc=desc, num=num,
                           items=[k[2] for k in keyed], normal=normal,
                           label=f"#{num} {desc} · {rows}×{cols} · {sum(max(1, int(getattr(d, 'NumberOfFrames', 1) or 1)) for d in lst)} img"))
    series.sort(key=lambda s: (s["num"], s["desc"]))
    return series, dup

def read_zip(zbytes, max_member=400 * 1024 * 1024, progress=None):
    import pydicom
    out, skipped = [], Counter()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        for n, info in enumerate(infos):
            base = info.filename.rsplit("/", 1)[-1]
            if "__MACOSX" in info.filename or base.startswith("."):
                continue
            if info.file_size < 132:
                skipped["too small to be DICOM"] += 1
                continue
            if info.file_size > max_member:
                skipped["file too large"] += 1
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(z.read(info)), force=True)
                if "PixelData" not in ds or not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                    skipped["DICOM without image pixels (DICOMDIR, report ...)"] += 1
                    continue
                out.append((info.filename, ds))
            except Exception:
                skipped["not a DICOM file"] += 1
            if progress and n % 20 == 0:
                progress(n / max(1, len(infos)))
    return out, skipped

def build_payload_series(s, enhance, params, progress=None):
    frames, errors = [], 0
    for ds in s["items"]:
        try:
            frames += modality_frames(ds)
        except Exception as e:
            errors += 1
            last = str(e)
    if not frames:
        raise RuntimeError("Could not decode pixel data (compressed DICOM needs: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg). Last error: " + (last if errors else "none"))
    a = s["items"][0]
    sp = _floats(getattr(a, "PixelSpacing", None) or [], 2) or [1.0, 1.0]
    info = series_info(frames)
    enh = None
    if enhance and info["sigma"] > 0:
        enh = []
        for k in range(len(frames)):
            enh.append(enhance_frame(frames, k, info, params))
            if progress:
                progress((k + 1) / len(frames))
    packed = pack_series(frames, enh)
    pos = [slice_position(d, s["normal"]) for d in s["items"]]
    if len(pos) != len(frames) or any(p is None for p in pos):
        pos = [float(i) for i in range(len(frames))]
    wc = _first(getattr(a, "WindowCenter", None)); ww = _first(getattr(a, "WindowWidth", None))
    slope = _first(getattr(a, "RescaleSlope", None), 1.0) or 1.0
    icpt = _first(getattr(a, "RescaleIntercept", None), 0.0) or 0.0
    dicom_wl = [wc - ww / 2, wc + ww / 2] if (wc is not None and ww) else None
    return dict(title=s["label"], plane=plane_name(s["normal"]), rows=s["rows"], cols=s["cols"],
                spacing=[float(sp[0]), float(sp[1])],
                invert=str(getattr(a, "PhotometricInterpretation", "")).upper() == "MONOCHROME1",
                pos=pos, hasEnh=bool(enh), snr=round(info["snr"], 1), sigma=info["sigma"],
                wl=dict(auto=[info["lo"], info["hi"]], dicom=dicom_wl, full=[float(min(f.min() for f in frames)), float(max(f.max() for f in frames))]),
                **packed)

# ----------------------------------------------------------------------------------------------------------------
#  Diagnostic Interface Engine (Dark-Mode Workspace Layer)
# ----------------------------------------------------------------------------------------------------------------
VIEWER_HTML = r"""
<style>
 *{box-sizing:border-box} 
 body{margin:0;background:#090d16;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;font-size:12px;overflow-x:hidden}
 
 /* High-end Medical Workspace Toolbar */
 #bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:8px 16px;background:#111827;border-bottom:1px solid #1f2937;position:sticky;top:0;z-index:100}
 .brand-tag{display:flex;align-items:center;gap:6px;font-weight:700;color:#38bdf8;letter-spacing:0.5px;font-size:13px;border-right:1px solid #374151;padding-right:12px}
 .brand-tag span{color:#94a3b8;font-weight:400}
 
 .tool-group{display:flex;align-items:center;gap:4px;background:#1f2937;padding:3px;border-radius:6px;border:1px solid #374151}
 .tool-label{font-size:11px;color:#94a3b8;padding:0 6px;text-transform:uppercase;font-weight:600;letter-spacing:0.3px}
 
 label{display:flex;gap:6px;align-items:center;cursor:pointer;color:#cbd5e1;padding:4px 8px;border-radius:4px;transition:all 0.15s ease}
 label:hover{background:#2d3748}
 #bar b{color:#38bdf8}
 
 #grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:10px;padding:10px;background:#090d16}
 .panel{border-radius:8px;padding:6px;background:#0f172a;border:1px solid #1e293b;position:relative;display:flex;flex-direction:column}
 .panel.active-viewport{border-color:#38bdf8;box-shadow:0 0 0 1px #38bdf8}
 
 .hd{display:flex;gap:6px;align-items:center;margin-bottom:6px;background:#1e293b;padding:4px 8px;border-radius:4px}
 .ttl{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#f8fafc;font-weight:600;font-size:11px;letter-spacing:.3px}
 
 button,select{background:#2d3748;color:#f1f5f9;border:1px solid #4a5568;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;transition:all 0.10s ease}
 button:hover,select:hover{background:#4facfe;color:#090d16;border-color:#4facfe}
 button.on{background:#0ea5e9;border-color:#0ea5e9;color:#ffffff;font-weight:600}
 
 canvas{width:100%;height:__PANEL_H__px;display:block;background:#000;border-radius:4px;cursor:crosshair;touch-action:none}
 .st{margin-top:4px;color:#94a3b8;font:11px ui-monospace,Menlo,Consolas,monospace;min-height:16px;background:#090d16;padding:4px 8px;border-radius:4px;border:1px solid #1e293b}
 .pat-banner{background:#1e1b4b;border:1px solid #4338ca;color:#c7d2fe;padding:4px 10px;border-radius:4px;font-weight:600;font-size:11px;margin-left:auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:300px}

 /* Workstation refinement */
 body{background:#0b1017}
 #bar{background:#111923;gap:8px;padding:8px;border:1px solid #263241;border-radius:6px}
 .brand-tag{font-size:11px;color:#70c8df;letter-spacing:1px}
 .tool-group{background:#16212e;border-color:#263241;flex-wrap:wrap}
 .tool-label{font-size:9px;letter-spacing:1px}
 #grid{background:#0b1017;padding:10px 0;gap:8px}
 .panel{padding:0;border-radius:5px;overflow:hidden;background:#000;min-width:0}
 .panel.active-viewport{border-color:#70c8df;box-shadow:0 0 0 1px #70c8df}
 .hd{flex-wrap:wrap;border-radius:0;background:#111923;padding:8px;margin:0}
 .ttl{flex-basis:100%;font-size:12px;margin-bottom:3px}
 button,select{background:#16212e;border-color:#334254;padding:5px 8px}
 button:hover,select:hover{background:#234255;color:#edf4fb;border-color:#70c8df}
 button.on{background:#234255;border-color:#70c8df}
 button:focus-visible,select:focus-visible,label:focus-within{outline:2px solid #70c8df;outline-offset:2px}
 input{accent-color:#70c8df}
 canvas{border-radius:0}
 .st{border:0;border-top:1px solid #263241;border-radius:0;margin:0;line-height:1.5;min-height:40px}
 .pat-banner{white-space:normal;max-width:none;flex-basis:100%;margin:0;background:#16212e;border-color:#334254;color:#b7c6d8}
 @media(max-width:900px){#grid{grid-template-columns:repeat(2,minmax(0,1fr))}.tool-label{display:none}}
 @media(max-width:600px){#grid{grid-template-columns:1fr}.brand-tag{display:none}}
</style>

<div id="bar">
  <div class="brand-tag">SD-DCM <span>WORKSTATION</span></div>
</div>
<div id="grid"></div>

<script>
const P=__PAYLOAD__;
const PH=__PANEL_H__, N=P.series.length;
const S=P.series.map(s=>({idx:Math.floor(s.pos.length/2),lo:s.wl.auto[0],hi:s.wl.auto[1],zoom:1,px:0,py:0,enh:s.hasEnh,inv:!!s.invert,cache:new Map(),tok:0,meas:{},play:null,lay:null}));
const opt={sync:true,link:false,tool:'wl'};
let hover=0;

const $=(t,a,c)=>{const e=document.createElement(t);if(a)Object.assign(e,a);if(c)c.forEach(x=>e.append(x));return e;};

async function inflate(b64){
  const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);
  const buf=await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  return new Uint16Array(buf);
}

async function getFrame(i,idx,enh){
  const s=P.series[i],st=S[i],key=idx+(enh?'e':'o');
  if(st.cache.has(key)){const v=st.cache.get(key);st.cache.delete(key);st.cache.set(key,v);return v;}
  const it=s.items[idx],arr=await inflate(enh&&it.e?it.e:it.o);
  st.cache.set(key,arr);if(st.cache.size>40)st.cache.delete(st.cache.keys().next().value);
  return arr;
}

const val=(s,u)=>s.off+u*s.scale;

function toScreen(st,x,y){const L=st.lay;return [L.ox+(x-L.w/2)*L.sc,L.oy+(y-L.h/2)*L.sc*L.asp];}
function toImage(st,ex,ey){const L=st.lay;return [((ex*L.dpr)-L.ox)/L.sc+L.w/2,(((ey*L.dpr)-L.oy)/(L.sc*L.asp))+L.h/2];}
function mm(s,m){const dx=(m.x2-m.x1)*s.spacing[1],dy=(m.y2-m.y1)*s.spacing[0];return Math.hypot(dx,dy);}

async function draw(i){
  const s=P.series[i],st=S[i],tok=++st.tok;
  const useEnh=st.enh&&s.hasEnh,data=await getFrame(i,st.idx,useEnh);
  if(tok!==st.tok)return;
  const w=s.cols,h=s.rows,oc=st.oc||(st.oc=$('canvas'));oc.width=w;oc.height=h;
  const octx=oc.getContext('2d'),img=octx.createImageData(w,h),px=img.data;
  const rng=Math.max(1e-9,st.hi-st.lo),a=255*s.scale/rng,b=255*(s.off-st.lo)/rng,inv=st.inv;
  for(let k=0,j=0;k<data.length;k++,j+=4){let g=a*data[k]+b;g=g<0?0:g>255?255:g;if(inv)g=255-g;px[j]=px[j+1]=px[j+2]=g;px[j+3]=255;}
  octx.putImageData(img,0,0);
  const cv=st.cv,ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,cw=cv.clientWidth,ch=cv.clientHeight;
  if(cv.width!==Math.round(cw*dpr)||cv.height!==Math.round(ch*dpr)){cv.width=Math.round(cw*dpr);cv.height=Math.round(ch*dpr);}
  ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
  const asp=s.spacing[0]/s.spacing[1],fit=Math.min(cv.width/w,cv.height/(h*asp))*0.98,sc=fit*st.zoom;
  st.lay={sc,asp,ox:cv.width/2+st.px*dpr,oy:cv.height/2+st.py*dpr,w,h,dpr};
  ctx.setTransform(sc,0,0,sc*asp,st.lay.ox,st.lay.oy);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
  ctx.drawImage(oc,-w/2,-h/2);
  ctx.setTransform(1,0,0,1,0,0);ctx.lineWidth=Math.max(1,dpr*1.5);ctx.font=(11*dpr)+'px ui-monospace,monospace';ctx.shadowColor='#000';ctx.shadowBlur=3*dpr;
  for(const m of (st.meas[st.idx]||[])){
    const [x1,y1]=toScreen(st,m.x1,m.y1),[x2,y2]=toScreen(st,m.x2,m.y2);
    ctx.strokeStyle=ctx.fillStyle=m.type==='roi'?'#fbbf24':'#34d399';
    if(m.type==='roi'){ctx.strokeRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));
      if(m.txt)m.txt.split('\n').forEach((t,k)=>ctx.fillText(t,Math.max(x1,x2)+6*dpr,Math.min(y1,y2)+(14+k*14)*dpr));}
    else{ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.fillRect(x1-3*dpr,y1-3*dpr,6*dpr,6*dpr);ctx.fillRect(x2-3*dpr,y2-3*dpr,6*dpr,6*dpr);
      ctx.fillText(mm(s,m).toFixed(2)+' mm',x2+8*dpr,y2+12*dpr);}
  }
  ctx.shadowBlur=0;
  st.stat.textContent='Img: '+(st.idx+1)+'/'+s.pos.length+' · '+s.plane+' · Loc: '+s.pos[st.idx].toFixed(1)+'mm · W: '+Math.round(st.hi-st.lo)+' L: '+Math.round((st.hi+st.lo)/2)+
    ' · Spacing: '+s.spacing[1].toFixed(2)+'×'+s.spacing[0].toFixed(2)+'mm'+(s.hasEnh?(useEnh?' [ENHANCED]':' [ORIGINAL]'):'');
}

const drawAll=()=>{for(let i=0;i<N;i++)draw(i);};

function setIdx(i,v,fromSync){
  const s=P.series[i],st=S[i];v=Math.max(0,Math.min(s.pos.length-1,v));if(v===st.idx&&!fromSync)return;
  st.idx=v;draw(i);
  if(opt.sync&&!fromSync){
    for(let j=0;j<N;j++){
      if(j===i||P.series[j].plane!==s.plane)continue;
      const pj=P.series[j].pos;let best=0,bd=Infinity;
      for(let k=0;k<pj.length;k++){const d=Math.abs(pj[k]-s.pos[v]);if(d<bd){bd=d;best=k;}}
      if(best!==S[j].idx){S[j].idx=best;draw(j);}
    }
  }
}

function resetView(i){const s=P.series[i],st=S[i];st.zoom=1;st.px=st.py=0;st.lo=s.wl.auto[0];st.hi=s.wl.auto[1];draw(i);}
function preset(i,name){const s=P.series[i],st=S[i],w=s.wl[name]||s.wl.auto;st.lo=w[0];st.hi=w[1];draw(i);}

async function roiStats(i,m){
  const s=P.series[i],st=S[i],data=await getFrame(i,st.idx,false);
  const x0=Math.max(0,Math.floor(Math.min(m.x1,m.x2))),x1=Math.min(s.cols-1,Math.ceil(Math.max(m.x1,m.x2)));
  const y0=Math.max(0,Math.floor(Math.min(m.y1,m.y2))),y1=Math.min(s.rows-1,Math.ceil(Math.max(m.y1,m.y2)));
  let n=0,sum=0,sq=0,mn=Infinity,mx=-Infinity;
  for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const v=val(s,data[y*s.cols+x]);n++;sum+=v;sq+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;}
  if(!n)return;const mean=sum/n,sd=Math.sqrt(Math.max(0,sq/n-mean*mean)),area=n*s.spacing[0]*s.spacing[1];
  m.txt='Mean: '+mean.toFixed(1)+'\nSD: '+sd.toFixed(1)+'\nMin/Max: '+mn.toFixed(0)+'/'+mx.toFixed(0)+'\nArea: '+area.toFixed(1)+'mm²';
  draw(i);
}

const bar=document.getElementById('bar');
const syncG=$('div',{className:'tool-group'});
syncG.append($('div',{className:'tool-label',textContent:'LINK'}));
syncG.append(chk('Scroll Linked','sync',true));
syncG.append(chk('W/L Linked','link',false));
bar.append(syncG);

const toolG=$('div',{className:'tool-group'});
toolG.append($('div',{className:'tool-label',textContent:'TOOLS'}));
[{t:'wl',l:'Window/Level'},{t:'pan',l:'Pan'},{t:'dist',l:'Distance'},{t:'roi',l:'ROI'}].forEach((x,k)=>{
  const r=$('input',{type:'radio',name:'tool',checked:k===0,ariaLabel:x.l});
  r.onchange=()=>{opt.tool=x.t;};
  toolG.append($('label',null,[r,document.createTextNode(x.l)]));
});
bar.append(toolG);

const clr=$('button',{textContent:'Clear measurements'});clr.onclick=()=>{S.forEach(st=>st.meas={});drawAll();};bar.append(clr);

if(P.patient){bar.append($('div',{className:'pat-banner',textContent:P.patient}));}

function chk(label,key,init){
  const c=$('input',{type:'checkbox',checked:init,ariaLabel:label});
  c.onchange=()=>{opt[key]=c.checked;};
  return $('label',null,[c,document.createTextNode(label)]);
}

const grid=document.getElementById('grid');
P.series.forEach((s,i)=>{
  const st=S[i],cv=$('canvas'),stat=$('div',{className:'st'}),pnl=$('div',{className:'panel'});
  if(i===0) pnl.classList.add('active-viewport');
  st.cv=cv;st.stat=stat;
  
  const sel=$('select');
  ['auto','dicom','full'].forEach(n=>{
    if(n!=='dicom'||s.wl.dicom)sel.append($('option',{value:n,textContent:{auto:'W/L Auto',dicom:'W/L DICOM',full:'W/L Full'}[n]}));
  });
  sel.onchange=()=>preset(i,sel.value);
  
  const bEnh=$('button',{textContent:'Enhance'}),bInv=$('button',{textContent:'Invert'}),bRst=$('button',{textContent:'Reset'}),bPlay=$('button',{textContent:'▶ Cine'});
  bEnh.className=st.enh?'on':'';bEnh.style.display=s.hasEnh?'':'none';
  bEnh.onclick=()=>{st.enh=!st.enh;bEnh.className=st.enh?'on':'';draw(i);};
  bInv.onclick=()=>{st.inv=!st.inv;bInv.className=st.inv?'on':'';draw(i);};bInv.className=st.inv?'on':'';
  bRst.onclick=()=>resetView(i);
  bPlay.onclick=()=>{
    if(st.play){clearInterval(st.play);st.play=null;bPlay.className='';}
    else{bPlay.className='on';st.play=setInterval(()=>setIdx(i,(st.idx+1)%s.pos.length),120);}
  };
  
  st.togEnh=()=>bEnh.onclick();st.togInv=()=>bInv.onclick();st.togPlay=()=>bPlay.onclick();
  pnl.append($('div',{className:'hd'},[$('div',{className:'ttl',textContent:s.title+(s.hasEnh?' · SNR:'+s.snr:'')}),sel,bEnh,bInv,bRst,bPlay]),cv,stat);
  grid.append(pnl);
  
  cv.addEventListener('mouseenter',()=>{
    hover=i;
    document.querySelectorAll('.panel').forEach((p,idx)=>p.classList.toggle('active-viewport',idx===i));
  });
  cv.addEventListener('wheel',e=>{
    e.preventDefault();
    if(e.ctrlKey||e.metaKey){st.zoom=Math.max(0.2,Math.min(20,st.zoom*Math.exp(-e.deltaY*0.01)));draw(i);}
    else setIdx(i,st.idx+(e.deltaY>0?1:-1));
  },{passive:false});
  
  cv.addEventListener('contextmenu',e=>e.preventDefault());
  let drag=null;
  cv.addEventListener('pointerdown',e=>{
    cv.setPointerCapture(e.pointerId);
    const tool=e.button===1?'pan':(e.shiftKey?'dist':opt.tool);
    drag={tool,x:e.clientX,y:e.clientY};
    if(tool==='dist'||tool==='roi'){
      const rc=cv.getBoundingClientRect(),[x,y]=toImage(st,e.clientX-rc.left,e.clientY-rc.top);
      const m={type:tool,x1:x,y1:y,x2:x,y2:y};(st.meas[st.idx]=st.meas[st.idx]||[]).push(m);drag.m=m;
    }
    e.preventDefault();
  });
  
  cv.addEventListener('pointermove',e=>{
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
    if(drag.tool==='wl'){
      const targets=opt.link?[...Array(N).keys()]:[i];
      targets.forEach(j=>{const sj=P.series[j],stj=S[j],r0=Math.max(1e-9,sj.wl.auto[1]-sj.wl.auto[0])/300;
        let c=(stj.hi+stj.lo)/2,w=stj.hi-stj.lo;w=Math.max(r0*2,w+dx*r0);c-=dy*r0;stj.lo=c-w/2;stj.hi=c+w/2;draw(j);});
    }else if(drag.tool==='pan'){st.px+=dx;st.py+=dy;draw(i);}
    else if(drag.m){const rc=cv.getBoundingClientRect(),[x,y]=toImage(st,e.clientX-rc.left,e.clientY-rc.top);drag.m.x2=x;drag.m.y2=y;draw(i);}
  });
  const end=()=>{if(drag&&drag.m&&drag.m.type==='roi')roiStats(i,drag.m);drag=null;};
  cv.addEventListener('pointerup',end);cv.addEventListener('pointercancel',end);
  cv.addEventListener('dblclick',()=>resetView(i));
});

window.addEventListener('keydown',e=>{
  if(hover<0)return;const st=S[hover],k=e.key.toLowerCase();
  if(k==='e'&&P.series[hover].hasEnh)st.togEnh();else if(k==='i')st.togInv();else if(k==='r')resetView(hover);
  else if(k===' '){st.togPlay();e.preventDefault();}
  else if(k==='arrowdown'||k==='arrowright'){setIdx(hover,st.idx+1);e.preventDefault();}
  else if(k==='arrowup'||k==='arrowleft'){setIdx(hover,st.idx-1);e.preventDefault();}
});
window.addEventListener('resize',drawAll);
drawAll();
</script>
"""

def build_viewer_html(series_payloads, patient_text=None, panel_height=430):
    n = len(series_payloads)
    cols = 3 if n >= 3 else max(1, n)
    payload = dict(series=series_payloads, patient=patient_text)
    data = json.dumps(payload, separators=(",", ":")).replace("</", r"<\/")
    html = (VIEWER_HTML.replace("__PAYLOAD__", data).replace("__COLS__", str(cols)).replace("__PANEL_H__", str(panel_height)))
    rows = -(-n // cols)
    return html, rows * (panel_height + 110) + 160

# ----------------------------------------------------------------------------------------------------------------
#  Unified Application Presentation Layer (Streamlit Frontend Hub)
# ----------------------------------------------------------------------------------------------------------------
CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background:#0b1017; color:#e4eaf2; font-family:Arial,sans-serif;
}
[data-testid="stHeader"] { background:#0b1017; }
[data-testid="stSidebar"] { background:#111923; border-right:1px solid #263241; }
[data-testid="stSidebar"] * { color:#dbe5ef; }
[data-testid="stMainBlockContainer"] { padding-top:1.3rem; padding-bottom:1rem; }
[data-testid="stSidebarUserContent"] { padding-top:1.4rem; }
[data-testid="stWidgetLabel"] p { font-size:13px; color:#b7c6d8; }
[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
    background:#16212e; color:#e4eaf2; border-color:#334254;
}
[data-testid="stExpander"] { background:#111923; border-color:#263241; }
.main-header { display:flex; justify-content:space-between; align-items:center;
    gap:12px; padding:16px 0 20px; margin-bottom:16px; border-bottom:1px solid #263241; }
.main-header h1 { margin:0; font-size:23px; font-weight:600; color:#edf4fb; letter-spacing:-.5px; }
.eyebrow { color:#70c8df; font-size:10px; letter-spacing:2px; font-weight:600; margin-bottom:5px; }
.subtitle { color:#8ea2b8; font-size:12px; margin-top:5px; }
.status-badge { color:#b6c7d9; border:1px solid #334254; border-radius:5px;
    padding:7px 10px; font-size:11px; white-space:nowrap; background:#111923; }
.stat-card { border:1px solid #263241; border-radius:6px; padding:12px 16px; background:#111923; }
.stat-val { font-size:20px; font-weight:600; color:#e4eaf2; }
.stat-lbl { font-size:11px; color:#8ea2b8; margin-top:3px; }
.empty-state { border:1px solid #263241; border-radius:8px; background:#111923;
    padding:48px 32px; margin:14px 0 20px; }
.empty-state h2 { font-size:27px; font-weight:500; color:#edf4fb; margin:12px 0; }
.empty-state p { color:#8ea2b8; max-width:560px; font-size:14px; line-height:1.7; }
.workflow { display:flex; flex-wrap:wrap; gap:10px; margin-top:26px; }
.workflow span { border:1px solid #334254; padding:9px 12px; border-radius:5px; color:#b7c6d8; font-size:12px; }
@media(max-width:640px) { .main-header { align-items:flex-start; flex-direction:column; } .empty-state {padding:24px;} }
</style>
"""

def _cache_resource(fn):
    return st.cache_resource(show_spinner=False)(fn) if st else fn

@_cache_resource
def load_index(file_hash, _zbytes):
    named, skipped = read_zip(_zbytes)
    series, dup = group_series(named)
    return series, skipped, dup

def main():
    st.set_page_config(page_title="SD-DCM | Clinical Workstation", page_icon="◉", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    st.markdown(
        '<div class="main-header"><div><div class="eyebrow">SD-DCM / IMAGING WORKSPACE</div>'
        '<h1>Clinical workstation</h1><div class="subtitle">Study review · Multi-series comparison · Image tools</div></div>'
        '<span class="status-badge">ZIP import · Research &amp; review</span></div>',
        unsafe_allow_html=True,
    )
    sb = st.sidebar
    sb.markdown("### Study browser")
    sb.caption("01 / IMPORT")
    up = sb.file_uploader("Open DICOM study", type=["zip"],
                          help="Upload a ZIP archive containing DICOM images.")
    sb.caption("Files are processed by the Streamlit server hosting this app.")
    sb.divider()
    sb.caption("02 / IMAGE PROCESSING")
    enhance = sb.toggle("Enable enhancement", value=True,
                        help="Creates a processed version. Use Enhance in each viewport to compare with the original.")
    with sb.expander("Enhancement settings", expanded=False):
        preset = st.selectbox("Preset", list(PRESETS) + ["Custom Parameters"], index=1,
                              disabled=not enhance)
        if preset == "Custom Parameters":
            base = PRESETS["Balanced"]
            params = dict(
                denoise=st.slider("Denoising", 0, 100, base["denoise"], disabled=not enhance),
                detail=st.slider("Detail retention", 0, 100, base["detail"], disabled=not enhance),
                sharpen=st.slider("Sharpening", 0, 100, base["sharpen"], disabled=not enhance),
                grain=st.slider("Texture retention", 0, 100, base["grain"], disabled=not enhance),
                shading=st.slider("Shading correction", 0, 100, base["shading"], disabled=not enhance),
            )
        else:
            params = dict(PRESETS[preset])
    show_ids = sb.checkbox("Show patient identifiers", value=False,
                           help="Identifiers are hidden in the viewer by default. This does not anonymize the uploaded files.")
    sb.divider()
    with sb.expander("Mouse & keyboard guide"):
        st.markdown("**Scroll** · Browse images\n\n**Ctrl / Cmd + scroll** · Zoom\n\n"
                    "**Middle drag / Pan tool** · Move image\n\n**Shift + drag** · Distance\n\n"
                    "**Double-click / R** · Reset view\n\n**Space** · Cine playback\n\n"
                    "**I** · Invert · **E** · Enhancement")

    if up is None:
        st.markdown(
            '<div class="empty-state"><div class="eyebrow">READY TO REVIEW</div>'
            '<h2>Your images. A focused workspace.</h2>'
            '<p>Open a DICOM ZIP archive from the study browser to begin. '
            'Select series, compare images side by side, and keep image controls within reach.</p>'
            '<div class="workflow"><span>01 &nbsp; Open study</span><span>02 &nbsp; Select series</span>'
            '<span>03 &nbsp; Review images</span></div></div>', unsafe_allow_html=True,
        )
        st.caption("Available tools: window/level, zoom, pan, distance, ROI statistics, linked scrolling and cine.")
        return

    zbytes = up.getvalue()
    fhash = hashlib.sha1(zbytes).hexdigest()
    
    with st.spinner("Reading DICOM images and grouping series..."):
        series, skipped, dup = load_index(fhash, zbytes)
        
    if not series:
        st.error("No readable DICOM images found. Check the ZIP contents and pixel-data support.")
        return
        
    # Ingest Metrics Panel
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-card"><div class="stat-val">{len(series)}</div><div class="stat-lbl">Series</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-card"><div class="stat-val">{sum(len(s["items"]) for s in series)}</div><div class="stat-lbl">DICOM instances</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-card"><div class="stat-val">{series[0]["rows"]}x{series[0]["cols"]}</div><div class="stat-lbl">First series matrix</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-card"><div class="stat-val">ZIP archive</div><div class="stat-lbl">Source</div></div>', unsafe_allow_html=True)
    
    
    # Study Browser Ingestion Logs
    if skipped or dup:
        with st.expander("Import details", expanded=False):
            for why, n in skipped.items():
                st.markdown(f"⚠️ *Skipped Entry:* `{n} instances` flagged as: **{why}**")
            if dup:
                st.markdown(f"Duplicate instances removed: `{dup}` (matching SOPInstanceUID).")

    # High-Density Series Selection Hub
    labels = {s["label"] + f"  [{s['uid'][-6:]}]": s for s in series}
    
    sb.caption("03 / SERIES")
    chosen = sb.multiselect(
        "Visible series", list(labels), default=list(labels)[:min(3, len(labels))],
        max_selections=MAX_PANELS, key=f"series_{fhash}",
        help="Select up to six series. Each selected series opens in a separate viewport.",
    )
    st.markdown("#### Image workspace")
    if not chosen:
        st.info("Select a series in the study browser to open a viewport.")
        return

    # Load Progress Vector mapping
    payloads = []
    progress_bar = st.progress(0.0, text="Preparing image workspace...")
    
    for idx, name in enumerate(chosen):
        s = labels[name]
        try:
            progress_bar.progress((idx) / len(chosen), text=f"Preparing series: {s['desc']}...")
            payloads.append(build_payload_series(s, bool(enhance), params))
        except Exception as e:
            st.error(f"Could not prepare series {s['label']}: {e}")
            
    progress_bar.empty()
    
    if not payloads:
        return
        
    # Patient Demographics Banner Customization
    patient = None
    if show_ids:
        a = labels[chosen[0]]["items"][0]
        patient = " | ".join([
            f"Patient: {getattr(a, 'PatientName', 'Anonymous')}",
            f"MRN: {getattr(a, 'PatientID', 'N/A')}",
            f"Study Date: {getattr(a, 'StudyDate', 'N/A')}",
            f"Scanner: {getattr(a, 'Manufacturer', 'N/A')}"
        ])
        
    # Embed Diagnostic Matrix
    html, height = build_viewer_html(payloads, patient)
    st.components.v1.html(html, height=height, scrolling=True)

# ----------------------------------------------------------------------------------------------------------------
#  Self-Test Diagnostic Entry Point Engine
# ----------------------------------------------------------------------------------------------------------------
def selftest():
    from types import SimpleNamespace
    rng = np.random.default_rng(3)
    N = 192
    yy, xx = np.mgrid[0:N, 0:N]
    truth = np.zeros((N, N), np.float32)
    ell = ((xx - N / 2) / (N * 0.38)) ** 2 + ((yy - N / 2) / (N * 0.46)) ** 2 < 1
    truth[ell] = 0.7
    truth[(((xx - 80) / 14.0) ** 2 + ((yy - 90) / 30.0) ** 2) < 1] = 0.15
    truth[(((xx - 120) / 12.0) ** 2 + ((yy - 90) / 12.0) ** 2) < 1] = 1.0
    sigma_true = 0.7 / 10

    def acquire(seed):
        r = np.random.default_rng(seed)
        K = np.fft.fftshift(np.fft.fft2(truth)); c = N // 2; a = N // 4
        mask = np.zeros((N, N)); mask[c - a:c + a, c - a:c + a] = 1
        noise = (r.normal(size=(N, N)) + 1j * r.normal(size=(N, N))) * mask
        to = lambda k: np.fft.ifft2(np.fft.ifftshift(k))
        scale = sigma_true / to(noise).real.std()
        return np.abs(to(K * mask + noise * scale)).astype(np.float32)

    frames = [acquire(s) for s in range(5)]
    psnr = lambda x: 10 * np.log10(truth.max() ** 2 / np.mean((x - truth) ** 2))
    nz = estimate_noise(frames[2])
    print(f"[noise] Estimated sigma {nz['sigma']:.4f} vs true {sigma_true:.4f} ({nz['method']})")
    assert 0.6 * sigma_true < nz["sigma"] < 1.5 * sigma_true
    info = series_info(frames)
    out = enhance_frame(frames, 2, info, PRESETS["Balanced"])
    print(f"[enhance] PSNR noisy {psnr(frames[2]):.2f} dB -> enhanced {psnr(out):.2f} dB")
    assert psnr(out) > psnr(frames[2]) + 1.5
    pk = pack_series(frames, [out] * 5)
    back = unpack_frame(pk["items"][2]["o"], N, N, pk["off"], pk["scale"])
    assert np.abs(back - frames[2]).max() <= pk["scale"] + 1e-9
    print(f"[pack] Round-trip max error {np.abs(back - frames[2]).max():.2e} (quantisation step {pk['scale']:.2e})")
    
    def fake(uid, inst, z, desc, sop):
        return SimpleNamespace(SeriesInstanceUID=uid, SeriesDescription=desc, SeriesNumber=1, Rows=64, Columns=64, InstanceNumber=inst,
                               ImagePositionPatient=[0, 0, z], ImageOrientationPatient=[1, 0, 0, 0, 1, 0], SOPInstanceUID=sop, PixelSpacing=[1.2, 0.8])
    ds = [("a", fake("A", 3, 10.0, "T2", "s3")), ("b", fake("A", 1, -10.0, "T2", "s1")), ("c", fake("A", 2, 0.0, "T2", "s2")),
          ("d", fake("B", 1, 0.0, "T2", "t1")), ("e", fake("A", 2, 0.0, "T2", "s2"))]
    series, dup = group_series(ds)
    assert len(series) == 2 and dup == 1, (len(series), dup)
    a = [s for s in series if s["uid"] == "A"][0]
    assert [d.InstanceNumber for d in a["items"]] == [1, 2, 3]
    assert plane_name(a["normal"]) == "Axial"
    print("ALL WORKSTATION CORE MODULE SELF-TESTS PASSED SUCCESSFULLY")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif st is not None:
        main()
    else:
        print("Streamlit engine context missing. Run command: 'streamlit run main.py'")
