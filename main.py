"""
SD-DCM Workstation  -  improved Streamlit version of the "AI Glassmorphic MRI Workstation" prototype
=====================================================================================================

Run:
    pip install streamlit pydicom numpy scipy
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg     # only needed for compressed DICOM (JPEG, JPEG 2000)
    streamlit run main.py                                          # (save this file as main.py)
    # large ZIPs / many series:  streamlit run main.py --server.maxUploadSize 2000 --server.maxMessageSize 500

Optional self-test of the image-processing / data-handling code (needs only numpy + scipy):
    python main.py --selftest

What changed compared with the original prototype (and why)
-----------------------------------------------------------
Correctness
  * PixelSpacing is [row spacing, column spacing]. The prototype swapped x/y and reused the first series' spacing
    for every panel; measurements are now computed per series with the right axis order.
  * Series are grouped by SeriesInstanceUID (+ matrix size), not by description text, so two different series that
    share a name are no longer merged. Duplicate SOPInstanceUIDs are dropped.
  * Slices are ordered by their position along the slice normal (ImagePositionPatient / ImageOrientationPatient),
    falling back to InstanceNumber.
  * RescaleSlope / RescaleIntercept are applied; MONOCHROME1 is inverted; multi-frame files are expanded.
Image fidelity
  * The prototype squashed every slice to 8 bits and a JPEG, then faked window/level with CSS filters. Now the full
    16-bit values are sent (zlib-compressed) and window/level is a real intensity window computed in the browser.
  * Window/level is shared across a series (no per-slice min/max re-normalisation, so no brightness flicker).
  * Measurements (ROI mean / SD) always read the ORIGINAL data, even while the enhanced image is displayed.
Enhancement (replaces the old "AI" filters, which blurred and amplified noise)
  * Noise is measured from the scan background (correlation-safe Rayleigh estimate), then Rician-aware non-local-means
    denoising uses the neighbouring slices, followed by structure-aware detail restoration and noise-gated sharpening.
  * Display only: source pixel data are never modified; Original / Enhanced can be toggled per panel (key E).
Workstation features
  * Up to 6 sequences, synchronised scrolling between series of the same plane, optional linked window/level,
    zoom / pan, cine, invert, W/L presets, distance caliper (mm), rectangular ROI statistics, patient info hidden
    by default (privacy), clear load report.

NOT a certified medical device. For review and research use only.
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

try:                                    # Streamlit is imported lazily so the processing code can be tested without it
    import streamlit as st
except ImportError:                     # pragma: no cover
    st = None

MAX_PANELS = 6

# ----------------------------------------------------------------------------------------------------------------
#  Enhancement presets (slider values, 0-100)
# ----------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Natural":  dict(denoise=25, detail=75, sharpen=60, grain=15, shading=0),
    "Balanced": dict(denoise=35, detail=65, sharpen=55, grain=12, shading=0),
    "Strong":   dict(denoise=60, detail=45, sharpen=45, grain=6,  shading=0),
}

# ----------------------------------------------------------------------------------------------------------------
#  Noise analysis
# ----------------------------------------------------------------------------------------------------------------
def estimate_noise(img):
    """Per-channel Gaussian sigma of the complex noise behind a magnitude image.
    Primary method: Rayleigh statistics of the background (works for the correlated noise left by zero-fill
    interpolation). Fallback for scans without a clean background: robust high-pass estimate."""
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
            sigma = float(np.median(vals)) / 1.1774          # Rayleigh median = 1.1774 sigma
            if not sigma > rng * 2e-4:
                sigma = 0.0
    if sigma == 0.0:
        method = "highpass"
        k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
        r = np.abs(ndi.convolve(img, k, mode="nearest"))[1:-1, 1:-1].ravel()
        sigma = float(np.quantile(r, 0.25)) / (0.3186 * 6) * 1.8
    fg_vals = img[img > mn + 0.15 * rng]
    out.update(sigma=sigma * (0.8 if method == "highpass" else 0.95),   # kept conservative on purpose
               fg=float(np.median(fg_vals)) if fg_vals.size else mx, p99=float(p99), bg_thr=float(bg_thr), method=method)
    return out


_OFFS = [(1, 0), (0, 1), (2, 1), (3, 0), (0, 3), (3, 3), (-2, 2), (1, 3)]


def patch_baseline(img, bg_thr):
    """Mean 3x3 patch distance between noise-only pixels (captures noise correlation); 0 if unavailable."""
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
    """Series-level analysis from up to 5 sample slices: noise, display window, SNR."""
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


# ----------------------------------------------------------------------------------------------------------------
#  Enhancement (display copy only)
# ----------------------------------------------------------------------------------------------------------------
def nlm25(slices, c, sigma, strength, patch_base=0.0, R=2, Rn=1):
    """Rician-aware non-local means using neighbouring slices (independent noise, similar anatomy)."""
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
    """Noise is least visible at edges / texture: there part of the original fine detail is restored."""
    if sigma <= 0 or (alpha <= 0 and grain <= 0):
        return den
    gy, gx = np.gradient(den)
    g = np.hypot(gx, gy) / sigma
    sw = _smoothstep((g - 0.35) / 1.6)
    fg = _smoothstep((den - 1.5 * sigma) / (2.5 * sigma))
    k = np.minimum(fg * (alpha * sw + grain), 1.0)
    return den + k * (orig - den)


def bias_correct(d, strength, info):
    """Gentle, foreground-only multiplicative shading correction (optional, off by default)."""
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
    """Noise-gated two-scale edge definition with overshoot control."""
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


# ----------------------------------------------------------------------------------------------------------------
#  Packing for the browser viewer (true 16-bit values, zlib-compressed)
# ----------------------------------------------------------------------------------------------------------------
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
#  DICOM handling (pydicom imported lazily; works with any object exposing the same attributes)
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
    """Decode a dataset to a list of 2-D float32 frames in modality units. Raises on undecodable data."""
    arr = np.asarray(ds.pixel_array)
    nfr = int(getattr(ds, "NumberOfFrames", 1) or 1)
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if spp == 3:                                     # colour: convert to luminance for display
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
    """named_datasets: list of (filename, dataset). Returns a list of series dicts (headers only, no pixel decode)."""
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
    """Read every DICOM image in a ZIP. Returns (list[(name, ds)], skipped Counter)."""
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
    """Decode one series and build the browser payload (+ enhanced copy when requested)."""
    frames, errors = [], 0
    for ds in s["items"]:
        try:
            frames += modality_frames(ds)
        except Exception as e:                       # e.g. compressed DICOM without a pixel-data handler installed
            errors += 1
            last = str(e)
    if not frames:
        raise RuntimeError("could not decode pixel data (compressed DICOM needs: pip install pylibjpeg pylibjpeg-libjpeg "
                           "pylibjpeg-openjpeg).  Last error: " + (last if errors else "none"))
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
                spacing=[float(sp[0]), float(sp[1])],                 # [row spacing (y), column spacing (x)]  <- correct order
                invert=str(getattr(a, "PhotometricInterpretation", "")).upper() == "MONOCHROME1",
                pos=pos, hasEnh=bool(enh), snr=round(info["snr"], 1), sigma=info["sigma"],
                wl=dict(auto=[info["lo"], info["hi"]], dicom=dicom_wl, full=[float(min(f.min() for f in frames)), float(max(f.max() for f in frames))]),
                **packed)


# ----------------------------------------------------------------------------------------------------------------
#  Browser viewer (HTML + JS). True window/level, sync scroll, linked W/L, measurements, ROI statistics.
# ----------------------------------------------------------------------------------------------------------------
VIEWER_HTML = r"""
<style>
 *{box-sizing:border-box} body{margin:0;background:transparent;color:#e6edf6;font:13px -apple-system,Segoe UI,Roboto,sans-serif}
 #bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 14px;margin-bottom:12px;border-radius:12px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09)}
 #bar label{display:flex;gap:6px;align-items:center;cursor:pointer;color:#cbd5e1} #bar b{color:#7dd3fc}
 #grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:14px}
 .panel{border-radius:12px;padding:8px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);backdrop-filter:blur(8px)}
 .hd{display:flex;gap:6px;align-items:center;margin-bottom:6px}
 .ttl{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#7dd3fc;font-weight:600;font-size:12px;letter-spacing:.3px}
 button,select{background:rgba(255,255,255,.08);color:#e6edf6;border:1px solid rgba(255,255,255,.14);border-radius:6px;padding:3px 7px;font-size:11px;cursor:pointer}
 button.on{background:#0ea5a0;border-color:#0ea5a0;color:#04201f}
 canvas{width:100%;height:__PANEL_H__px;display:block;background:#000;border-radius:8px;cursor:crosshair;touch-action:none}
 .st{margin-top:6px;color:#94a3b8;font:11px ui-monospace,Menlo,Consolas,monospace;min-height:15px}
 .pat{color:#fbbf24}
</style>
<div id="bar"></div><div id="grid"></div>
<script>
const P=__PAYLOAD__;
const PH=__PANEL_H__, N=P.series.length;
const S=P.series.map(s=>({idx:Math.floor(s.pos.length/2),lo:s.wl.auto[0],hi:s.wl.auto[1],zoom:1,px:0,py:0,enh:s.hasEnh,inv:!!s.invert,cache:new Map(),tok:0,meas:{},play:null,lay:null}));
const opt={sync:true,link:false,tool:'wl'};
let hover=-1;
const $=(t,a,c)=>{const e=document.createElement(t);if(a)Object.assign(e,a);if(c)c.forEach(x=>e.append(x));return e;};

/* ---------- decoding (true 16-bit values) ---------- */
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

/* ---------- drawing ---------- */
function toScreen(st,x,y){const L=st.lay;return [L.ox+(x-L.w/2)*L.sc,L.oy+(y-L.h/2)*L.sc*L.asp];}
function toImage(st,ex,ey){const L=st.lay;return [((ex*L.dpr)-L.ox)/L.sc+L.w/2,(((ey*L.dpr)-L.oy)/(L.sc*L.asp))+L.h/2];}
function mm(s,m){const dx=(m.x2-m.x1)*s.spacing[1],dy=(m.y2-m.y1)*s.spacing[0];return Math.hypot(dx,dy);}   // spacing = [row(y), col(x)]
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
  ctx.setTransform(1,0,0,1,0,0);ctx.lineWidth=Math.max(1,dpr*1.5);ctx.font=(12*dpr)+'px ui-monospace,monospace';ctx.shadowColor='#000';ctx.shadowBlur=3*dpr;
  for(const m of (st.meas[st.idx]||[])){
    const [x1,y1]=toScreen(st,m.x1,m.y1),[x2,y2]=toScreen(st,m.x2,m.y2);
    ctx.strokeStyle=ctx.fillStyle=m.type==='roi'?'#fbbf24':'#34d399';
    if(m.type==='roi'){ctx.strokeRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));
      if(m.txt)m.txt.split('\n').forEach((t,k)=>ctx.fillText(t,Math.max(x1,x2)+6*dpr,Math.min(y1,y2)+(14+k*14)*dpr));}
    else{ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.fillRect(x1-3*dpr,y1-3*dpr,6*dpr,6*dpr);ctx.fillRect(x2-3*dpr,y2-3*dpr,6*dpr,6*dpr);
      ctx.fillText(mm(s,m).toFixed(2)+' mm',x2+8*dpr,y2+12*dpr);}
  }
  ctx.shadowBlur=0;
  st.stat.textContent='Slice '+(st.idx+1)+'/'+s.pos.length+' · '+s.plane+' · pos '+s.pos[st.idx].toFixed(1)+' · W '+Math.round(st.hi-st.lo)+' L '+Math.round((st.hi+st.lo)/2)+
    ' · '+s.spacing[1].toFixed(2)+'×'+s.spacing[0].toFixed(2)+' mm'+(s.hasEnh?(useEnh?' · ENHANCED (display only)':' · ORIGINAL'):'');
}
const drawAll=()=>{for(let i=0;i<N;i++)draw(i);};

/* ---------- slice navigation (+ synchronised scrolling by slice position, same plane only) ---------- */
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

/* ---------- ROI statistics always come from the ORIGINAL data ---------- */
async function roiStats(i,m){
  const s=P.series[i],st=S[i],data=await getFrame(i,st.idx,false);
  const x0=Math.max(0,Math.floor(Math.min(m.x1,m.x2))),x1=Math.min(s.cols-1,Math.ceil(Math.max(m.x1,m.x2)));
  const y0=Math.max(0,Math.floor(Math.min(m.y1,m.y2))),y1=Math.min(s.rows-1,Math.ceil(Math.max(m.y1,m.y2)));
  let n=0,sum=0,sq=0,mn=Infinity,mx=-Infinity;
  for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const v=val(s,data[y*s.cols+x]);n++;sum+=v;sq+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;}
  if(!n)return;const mean=sum/n,sd=Math.sqrt(Math.max(0,sq/n-mean*mean)),area=n*s.spacing[0]*s.spacing[1];
  m.txt='mean '+mean.toFixed(1)+'\nSD '+sd.toFixed(1)+'\nmin/max '+mn.toFixed(0)+'/'+mx.toFixed(0)+'\narea '+area.toFixed(1)+' mm² (original data)';
  draw(i);
}

/* ---------- build UI ---------- */
const bar=document.getElementById('bar');
const chk=(label,key,init)=>{const c=$('input',{type:'checkbox',checked:init});c.onchange=()=>{opt[key]=c.checked;};return $('label',null,[c,document.createTextNode(label)]);};
bar.append($('span',null,[Object.assign($('b'),{textContent:'SD-DCM'}),document.createTextNode(' workstation')]),chk('Sync scroll','sync',true),chk('Link W/L','link',false));
['wl','dist','roi'].forEach((t,k)=>{const r=$('input',{type:'radio',name:'tool',checked:k===0});r.onchange=()=>{opt.tool=t;};bar.append($('label',null,[r,document.createTextNode({wl:'Window/Level',dist:'Distance',roi:'ROI stats'}[t])]));});
const clr=$('button',{textContent:'Clear measurements'});clr.onclick=()=>{S.forEach(st=>st.meas={});drawAll();};bar.append(clr);
bar.append($('span',{style:'color:#94a3b8;font-size:11px',textContent:'Wheel: slices · Ctrl+wheel: zoom · Shift+drag: distance · Middle-drag: pan · E: original/enhanced · I: invert · R: reset · Space: play'}));
if(P.patient){bar.append($('span',{className:'pat',textContent:P.patient}));}

const grid=document.getElementById('grid');
P.series.forEach((s,i)=>{
  const st=S[i],cv=$('canvas'),stat=$('div',{className:'st'});st.cv=cv;st.stat=stat;
  const sel=$('select');['auto','dicom','full'].forEach(n=>{if(n!=='dicom'||s.wl.dicom)sel.append($('option',{value:n,textContent:{auto:'W/L: auto',dicom:'W/L: DICOM',full:'W/L: full range'}[n]}));});
  sel.onchange=()=>preset(i,sel.value);
  const bEnh=$('button',{textContent:'Enhanced'}),bInv=$('button',{textContent:'Invert'}),bRst=$('button',{textContent:'Reset'}),bPlay=$('button',{textContent:'▶'});
  bEnh.className=st.enh?'on':'';bEnh.style.display=s.hasEnh?'':'none';
  bEnh.onclick=()=>{st.enh=!st.enh;bEnh.className=st.enh?'on':'';draw(i);};
  bInv.onclick=()=>{st.inv=!st.inv;bInv.className=st.inv?'on':'';draw(i);};bInv.className=st.inv?'on':'';
  bRst.onclick=()=>resetView(i);
  bPlay.onclick=()=>{if(st.play){clearInterval(st.play);st.play=null;bPlay.className='';}else{bPlay.className='on';st.play=setInterval(()=>setIdx(i,(st.idx+1)%s.pos.length),120);}};
  st.togEnh=()=>bEnh.onclick();st.togInv=()=>bInv.onclick();st.togPlay=()=>bPlay.onclick();
  grid.append($('div',{className:'panel'},[$('div',{className:'hd'},[$('div',{className:'ttl',textContent:s.title+(s.hasEnh?'  · SNR≈'+s.snr:'')}),sel,bEnh,bInv,bRst,bPlay]),cv,stat]));
  cv.addEventListener('mouseenter',()=>{hover=i;});
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
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (VIEWER_HTML.replace("__PAYLOAD__", data).replace("__COLS__", str(cols)).replace("__PANEL_H__", str(panel_height)))
    rows = -(-n // cols)
    return html, rows * (panel_height + 70) + 90


# ----------------------------------------------------------------------------------------------------------------
#  Streamlit app
# ----------------------------------------------------------------------------------------------------------------
CSS = """
<style>
body,.stApp{background:linear-gradient(135deg,#060913 0%,#0b1121 100%)!important;color:#f1f5f9!important;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif!important}
[data-testid="stSidebar"]{background:rgba(15,23,42,.55)!important;backdrop-filter:blur(16px);border-right:1px solid rgba(255,255,255,.08)!important}
h1,h2,h3{background:linear-gradient(90deg,#00f2fe 0%,#4facfe 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:600!important}
</style>
"""


def _cache_resource(fn):
    return st.cache_resource(show_spinner=False)(fn) if st else fn


def _cache_data(fn):
    return st.cache_data(show_spinner=False, max_entries=24)(fn) if st else fn


@_cache_resource
def load_index(file_hash, _zbytes):
    named, skipped = read_zip(_zbytes)
    series, dup = group_series(named)
    return series, skipped, dup


@_cache_data
def get_payload(file_hash, series_uid, rows, cols, enhance, params_key, _series):
    return build_payload_series(_series, enhance, dict(params_key))


def main():
    st.set_page_config(page_title="SD-DCM Workstation", page_icon="🧠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🧠 SD-DCM MRI Workstation")
    st.caption("Review and research use only – not a certified medical device. Images are processed in this session; enhancement is "
               "display-only and the original data are always available (toggle per panel or press E).")

    sb = st.sidebar
    sb.markdown("### ⚡ Workstation")
    up = sb.file_uploader("Upload a patient ZIP (DICOM folder)", type=["zip"])
    enhance = sb.toggle("Enhance images (display only)", value=True,
                        help="Noise-aware denoising, detail restoration and noise-gated sharpening. Never alters the source data.")
    preset = sb.selectbox("Enhancement style", list(PRESETS) + ["Custom"], index=1)
    if preset == "Custom":
        base = PRESETS["Balanced"]
        params = dict(
            denoise=sb.slider("Noise reduction", 0, 100, base["denoise"]),
            detail=sb.slider("Detail preservation", 0, 100, base["detail"]),
            sharpen=sb.slider("Edge definition", 0, 100, base["sharpen"]),
            grain=sb.slider("Texture retention", 0, 100, base["grain"]),
            shading=sb.slider("Shading correction", 0, 100, base["shading"]))
    else:
        params = dict(PRESETS[preset])
    show_ids = sb.checkbox("Show patient identifiers", value=False)

    if up is None:
        st.info("Upload a ZIP containing the patient's DICOM files to begin.")
        return
    zbytes = up.getvalue()
    fhash = hashlib.sha1(zbytes).hexdigest()
    with st.spinner("Reading DICOM files…"):
        series, skipped, dup = load_index(fhash, zbytes)
    if not series:
        st.error("No viewable DICOM images found in this ZIP.")
    if skipped or dup:
        with sb.expander("Load report"):
            for why, n in skipped.items():
                st.write(f"{n} × {why}")
            if dup:
                st.write(f"{dup} × duplicate image")
    if not series:
        return

    labels = {s["label"] + f"  [{s['uid'][-6:]}]": s for s in series}
    chosen = sb.multiselect(f"Sequences to display (up to {MAX_PANELS})", list(labels), default=list(labels)[:min(3, len(labels))])
    chosen = chosen[:MAX_PANELS]
    if not chosen:
        st.warning("Select at least one sequence in the sidebar.")
        return

    payloads, bar = [], st.progress(0.0, text="Preparing images…")
    pkey = tuple(sorted(params.items()))
    for n, name in enumerate(chosen):
        s = labels[name]
        try:
            bar.progress(n / len(chosen), text=f"Preparing {s['desc']}…")
            payloads.append(get_payload(fhash, s["uid"], s["rows"], s["cols"], bool(enhance), pkey, s))
        except Exception as e:
            st.error(f"{s['label']}: {e}")
    bar.empty()
    if not payloads:
        return

    patient = None
    if show_ids:
        a = labels[chosen[0]]["items"][0]
        patient = " | ".join([f"Name: {getattr(a, 'PatientName', 'N/A')}", f"ID: {getattr(a, 'PatientID', 'N/A')}",
                              f"Date: {getattr(a, 'StudyDate', 'N/A')}", f"Scanner: {getattr(a, 'Manufacturer', 'N/A')}"])
    html, height = build_viewer_html(payloads, patient)
    st.components.v1.html(html, height=height, scrolling=True)


# ----------------------------------------------------------------------------------------------------------------
#  Self-test (python main.py --selftest): needs only numpy + scipy
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

    def acquire(seed):                                   # k-space truncation + complex noise + zero-fill + magnitude
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
    print(f"[noise] estimated sigma {nz['sigma']:.4f} vs true {sigma_true:.4f}  ({nz['method']})")
    assert 0.6 * sigma_true < nz["sigma"] < 1.5 * sigma_true
    info = series_info(frames)
    out = enhance_frame(frames, 2, info, PRESETS["Balanced"])
    print(f"[enhance] PSNR noisy {psnr(frames[2]):.2f} dB -> enhanced {psnr(out):.2f} dB")
    assert psnr(out) > psnr(frames[2]) + 1.5
    # pack / unpack round trip
    pk = pack_series(frames, [out] * 5)
    back = unpack_frame(pk["items"][2]["o"], N, N, pk["off"], pk["scale"])
    assert np.abs(back - frames[2]).max() <= pk["scale"] + 1e-9
    print(f"[pack] round-trip max error {np.abs(back - frames[2]).max():.2e} (quantisation step {pk['scale']:.2e})")
    # series grouping / ordering / duplicates with stand-in datasets
    def fake(uid, inst, z, desc, sop):
        return SimpleNamespace(SeriesInstanceUID=uid, SeriesDescription=desc, SeriesNumber=1, Rows=64, Columns=64, InstanceNumber=inst,
                               ImagePositionPatient=[0, 0, z], ImageOrientationPatient=[1, 0, 0, 0, 1, 0], SOPInstanceUID=sop, PixelSpacing=[1.2, 0.8])
    ds = [("a", fake("A", 3, 10.0, "T2", "s3")), ("b", fake("A", 1, -10.0, "T2", "s1")), ("c", fake("A", 2, 0.0, "T2", "s2")),
          ("d", fake("B", 1, 0.0, "T2", "t1")), ("e", fake("A", 2, 0.0, "T2", "s2"))]     # same description, different UID; one duplicate
    series, dup = group_series(ds)
    assert len(series) == 2 and dup == 1, (len(series), dup)
    a = [s for s in series if s["uid"] == "A"][0]
    assert [d.InstanceNumber for d in a["items"]] == [1, 2, 3]
    assert plane_name(a["normal"]) == "Axial"
    print("[series] grouped by SeriesInstanceUID, ordered by position, duplicate dropped, plane =", plane_name(a["normal"]))
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif st is not None:
        main()
    else:
        print("Streamlit is not installed. Run:  pip install streamlit pydicom numpy scipy   then   streamlit run main.py")