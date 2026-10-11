"""
SD-DCM Clinical Workstation  -  Premium Medical Imaging Command Center
=====================================================================================================
Built according to international radiology workstation standards and clinical UX guidelines.

Run:
    pip install -r requirements.txt
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

def decode_pixels(ds):
    """Use pydicom's automatic decoder selection; retain source pixel data."""
    from pydicom.pixels import get_decoder
    uid = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if uid is None:
        raise RuntimeError("Missing DICOM TransferSyntaxUID; the encoding cannot be determined safely.")
    try:
        decoder = get_decoder(uid)
    except NotImplementedError as exc:
        raise RuntimeError(f"Unsupported DICOM transfer syntax: {uid.name} ({uid}).") from exc
    if not decoder.is_available:
        packages = {
            "1.2.840.10008.1.2.4.80": "pyjpegls>=1.2 (JPEG-LS)",
            "1.2.840.10008.1.2.4.81": "pyjpegls>=1.2 (JPEG-LS)",
            "1.2.840.10008.1.2.4.90": "pylibjpeg and pylibjpeg-openjpeg (JPEG 2000)",
            "1.2.840.10008.1.2.4.91": "pylibjpeg and pylibjpeg-openjpeg (JPEG 2000)",
        }
        needed = packages.get(str(uid), "the matching DICOM pixel decoder")
        raise RuntimeError(
            f"Decoder unavailable for {uid.name} ({uid}). Install {needed} on the hosting server "
            "using the updated requirements.txt, then rebuild/reboot the app. "
            "Re-uploading or repacking the ZIP will not install the decoder."
        )
    try:
        return np.asarray(ds.pixel_array)
    except Exception as exc:
        raise RuntimeError(f"Pixel decoding failed for {uid.name} ({uid}): {exc}") from exc


def modality_frames(ds):
    arr = decode_pixels(ds)
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
    frames, errors = [], []
    for index, ds in enumerate(s["items"], 1):
        try:
            frames += modality_frames(ds)
        except Exception as exc:
            errors.append((index, str(exc)))
    if errors:
        index, reason = errors[0]
        raise RuntimeError(
            f"{len(errors)} of {len(s['items'])} DICOM instances could not be decoded "
            f"(first failure: instance {index}). Series not displayed to avoid an incomplete stack. {reason}"
        )
    if not frames:
        raise RuntimeError("Series contains no image frames.")
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
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#080c12;color:#d4deea;font:12px -apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif}
button,select{font:inherit;cursor:pointer;color:#b7c8da;background:#16212e;border:1px solid #2b3b4d;border-radius:4px;padding:6px 9px}
button:hover,select:hover{color:#eef8ff;background:#23384b;border-color:#518caf}
button:focus-visible,select:focus-visible{outline:2px solid #63c5e7;outline-offset:2px}
button.on{background:#173a50;border-color:#4384a6;color:#91dcf5}
button:disabled{opacity:.4;cursor:default}input{accent-color:#63c5e7}
#bar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:8px 10px;background:#131d29;border:1px solid #2b3b4d;border-radius:5px 5px 0 0}
.brand-tag{font-size:10px;letter-spacing:1.2px;color:#7890a8;padding:0 10px 0 0;border-right:1px solid #314255}
.tool-group{display:flex;gap:3px;align-items:center}.tool-label{color:#6b8198;font-size:9px;letter-spacing:1px;margin-right:5px}
label{display:flex;align-items:center;gap:5px;font-size:11px;padding:6px 4px;cursor:pointer}
.tool-button{display:inline-flex;align-items:center;gap:6px;background:transparent;border-color:transparent;padding:7px 9px}
.tool-button svg{width:17px;height:17px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.workspace{display:grid;grid-template-columns:146px 54px minmax(0,1fr);border:1px solid #26313f;border-top:0;min-height:0;flex:1;overflow:hidden}
#series-rail{overflow-y:auto;min-height:0;background:#101720;border-right:1px solid #26313f;padding:12px 9px}
.rail-heading{font-size:9px;letter-spacing:1.5px;color:#70859c;margin:0 0 14px 3px}
.series-card{display:block;width:100%;padding:6px;text-align:left;margin-bottom:10px;background:#121c27;border-color:#29394b}
.series-card.active{border-color:#55bcdf;background:#172d3d}
.series-card canvas{height:76px;width:100%;background:#000;border-radius:2px;cursor:pointer}
.series-card .name{display:block;font-size:11px;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.series-card .meta{font-size:9px;color:#70859c;display:block;margin-top:4px}
#grid{display:grid;grid-template-columns:repeat(__COLS__,minmax(0,1fr));gap:2px;padding:2px;background:#080c12;align-content:stretch;min-height:0;overflow:hidden;grid-template-rows:repeat(2,minmax(0,1fr))}
.panel{position:relative;min-width:0;min-height:0;background:#000;border:1px solid #26313f;display:flex;flex-direction:column;overflow:hidden}
.panel.active-viewport{border-color:#55bcdf}
.hd{display:flex;flex-wrap:wrap;gap:5px;align-items:center;background:#101821;padding:7px 9px;border-bottom:1px solid #26313f}
.ttl{flex:1 1 100%;font-size:11px;color:#cbdcea;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px}
.hd button,.hd select{font-size:10px;padding:4px 6px;background:#15212d}
.panel canvas{width:100%;height:0;min-height:0;flex:1;display:block;background:#000;cursor:crosshair;touch-action:none}
.st{padding:8px 9px;background:#0c131c;color:#7992a9;font:10px ui-monospace,Consolas,monospace;line-height:1.6;min-height:42px;border-top:1px solid #26313f}
.pat-banner{width:100%;padding:7px 10px;background:#172634;color:#c0d5e7;border-top:1px solid #314255;font-size:11px}
.workspace-footer{display:flex;justify-content:space-between;gap:12px;color:#688097;font-size:10px;background:#111923;padding:10px 12px;border:1px solid #26313f;border-top:0}
.empty-canvas{position:relative;min-height:0;flex:1;background:radial-gradient(ellipse at 50% 52%,#101923 0%,#05090e 60%);display:grid;place-items:center}
.empty-canvas .corner{position:absolute;top:14px;left:14px;color:#405268;font:10px ui-monospace,monospace}
.empty-canvas .center{text-align:center;color:#50667d;font-size:11px;line-height:1.8}
.empty-canvas .center svg{width:39px;height:39px;display:block;margin:0 auto 10px;stroke:#354c61;fill:none;stroke-width:1}
.empty-canvas .center b{color:#9eb4c9;font-size:14px;font-weight:400;display:block}
.empty-canvas .bottom{position:absolute;bottom:14px;left:14px;font:9px ui-monospace,monospace;color:#35485c}
.rail-empty{height:85px;border:1px dashed #2c4054;display:grid;place-items:center;color:#4f667d;font-size:25px;margin-bottom:10px}
.rail-hint{color:#70859c;font-size:10px;line-height:1.7;padding:5px}
@media(max-width:1000px){.workspace{grid-template-columns:110px 54px minmax(0,1fr)}.tool-label,.brand-tag{display:none}}
@media(max-width:600px){.workspace{grid-template-columns:44px minmax(0,1fr)}#series-rail{display:none}.tool-button{padding:6px}.tool-button span{display:none}.workspace-footer{flex-wrap:wrap}}
#tool-rail{overflow-y:auto;min-height:0;background:#101720;border-right:1px solid #26313f;padding:7px 5px;display:flex;flex-direction:column;gap:2px;align-items:center}
#tool-rail .tool-button{width:40px;height:30px;justify-content:center;padding:7px;flex-shrink:0}
#tool-rail .tool-button span{display:none}
.rail-separator{width:30px;border-top:1px solid #26313f;margin:5px 0}
.rail-category{font-size:8px;letter-spacing:.4px;color:#70859c;padding:3px 0}
#tool-status{color:#8cb7ce;font-size:10px}
#annotation-text{width:140px;background:#16212e;border:1px solid #314255;border-radius:4px;color:#d4deea;padding:6px;font:11px inherit}
#overlay-list{max-width:155px}
#bar .overlay-controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
html,body{height:100%;overflow:hidden}body{display:flex;flex-direction:column}
#bar,.workspace-footer{flex-shrink:0}#bar{max-height:110px;overflow:auto}
.drop-hint{position:absolute;inset:0;display:grid;place-items:center;color:#688097;font-size:11px;text-align:center;pointer-events:none;background:#080c12}
.panel.drop-target{outline:2px solid #55bcdf;outline-offset:-3px}
.layout-picker{position:relative}.layout-menu{position:fixed;z-index:1000;background:#15212e;border:1px solid #435970;padding:12px;border-radius:5px;box-shadow:0 8px 24px #0009}
.layout-matrix{display:grid;grid-template-columns:repeat(4,25px);gap:4px;margin-top:8px}.layout-cell{width:25px;height:25px;padding:0;background:#101720;border:1px solid #435970}.layout-cell.preview{background:#22546f;border-color:#55bcdf}
#grid.dense .hd{padding:3px}.dense .hd button,.dense .hd select{padding:2px;font-size:9px}.dense .ttl{font-size:9px}.dense .st{padding:3px;font-size:8px;min-height:0}.dense .hd{gap:2px}
</style>

<div id="bar">
  <div class="brand-tag">IMAGE TOOLS</div>
</div>
<div class="workspace"><nav id="series-rail" aria-label="Loaded series"><div class="rail-heading">LOADED SERIES</div></nav><nav id="tool-rail" aria-label="Image tools"></nav><div id="grid"></div></div>
<div class="workspace-footer"><span>Scroll to browse &nbsp; · &nbsp; Ctrl / Cmd + scroll to zoom &nbsp; · &nbsp; Middle drag to pan</span><span>Space: cine &nbsp; · &nbsp; R: reset</span></div>

<script>
function fitHost(){try{const host=window.frameElement;if(host){host.style.height=Math.max(300,window.parent.innerHeight-host.getBoundingClientRect().top-12)+'px';}}catch(e){}}
fitHost();try{window.parent.addEventListener('resize',fitHost);}catch(e){}
const P=__PAYLOAD__;
const catalog=P.series;
P.series=Array.from({length:16},(_,i)=>i<Math.min(4,catalog.length)?catalog[i]:null);
const PH=__PANEL_H__, N=16;
let layoutRows=2,layoutCols=2;
const S=P.series.map(item=>{const s=item||catalog[0];return ({idx:Math.floor(s.pos.length/2),lo:s.wl.auto[0],hi:s.wl.auto[1],zoom:1,px:0,py:0,enh:s.hasEnh,inv:!!s.invert,cache:new Map(),tok:0,meas:{},play:null,lay:null,pendingAngle:null});});
const opt={sync:true,link:false,tool:'wl',overlays:true};
let hover=0;

const $=(t,a,c)=>{const e=document.createElement(t);if(a)Object.assign(e,a);if(c)c.forEach(x=>e.append(x));return e;};

async function inflate(b64){
  const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);
  const buf=await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  return new Uint16Array(buf);
}

async function getFrame(i,idx,enh){
  const s=P.series[i],st=S[i],key=idx+(enh?'e':'o'),cache=st.cache;
  if(cache.has(key)){const v=cache.get(key);cache.delete(key);cache.set(key,v);return v;}
  const it=s.items[idx],arr=await inflate(enh&&it.e?it.e:it.o);
  cache.set(key,arr);if(cache.size>40)cache.delete(cache.keys().next().value);
  return arr;
}

const val=(s,u)=>s.off+u*s.scale;

function toScreen(st,x,y){const L=st.lay;return [L.ox+(x-L.w/2)*L.sc,L.oy+(y-L.h/2)*L.sc*L.asp];}
function toImage(st,ex,ey){const L=st.lay;return [((ex*L.dpr)-L.ox)/L.sc+L.w/2,(((ey*L.dpr)-L.oy)/(L.sc*L.asp))+L.h/2];}
function mm(s,m){const dx=(m.x2-m.x1)*s.spacing[1],dy=(m.y2-m.y1)*s.spacing[0];return Math.hypot(dx,dy);}

async function draw(i){
  const s=P.series[i],st=S[i];if(!s||st.panel.hidden)return;const tok=++st.tok;
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
  if(opt.overlays)for(const m of (st.meas[st.idx]||[])){
    const [x1,y1]=toScreen(st,m.x1,m.y1),[x2,y2]=toScreen(st,m.x2,m.y2);
    ctx.strokeStyle=ctx.fillStyle=(m.type==='roi'||m.type==='ellipse')?'#fbbf24':'#34d399';
    if(m.type==='roi'||m.type==='ellipse'){
      if(m.type==='ellipse'){ctx.beginPath();ctx.ellipse((x1+x2)/2,(y1+y2)/2,Math.abs(x2-x1)/2,Math.abs(y2-y1)/2,0,0,Math.PI*2);ctx.stroke();}
      else ctx.strokeRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));
      if(m.txt)m.txt.split('\n').forEach((t,k)=>ctx.fillText(t,Math.max(x1,x2)+6*dpr,Math.min(y1,y2)+(14+k*14)*dpr));
    }else if(m.type==='angle'){
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
      if(m.x3!==undefined){const [x3,y3]=toScreen(st,m.x3,m.y3);ctx.lineTo(x3,y3);}
      ctx.stroke();const degrees=angleDegrees(s,m);ctx.fillText(degrees===null?'Click next point':degrees.toFixed(1)+'°',x2+8*dpr,y2+12*dpr);
    }else if(m.type==='text'){ctx.fillText(m.label||'Note',x1+5*dpr,y1+12*dpr);}
    else{
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
      if(m.type==='arrow'){const a=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2-10*dpr*Math.cos(a-.45),y2-10*dpr*Math.sin(a-.45));ctx.lineTo(x2,y2);ctx.lineTo(x2-10*dpr*Math.cos(a+.45),y2-10*dpr*Math.sin(a+.45));ctx.stroke();if(m.label)ctx.fillText(m.label,x1+5*dpr,y1-6*dpr);}
      else{ctx.fillRect(x1-3*dpr,y1-3*dpr,6*dpr,6*dpr);ctx.fillRect(x2-3*dpr,y2-3*dpr,6*dpr,6*dpr);ctx.fillText(mm(s,m).toFixed(2)+' mm',x2+8*dpr,y2+12*dpr);}
    }
  }
  ctx.shadowBlur=0;
  if(i===hover)refreshOverlays();
  st.stat.textContent='Img: '+(st.idx+1)+'/'+s.pos.length+' · '+s.plane+' · Loc: '+s.pos[st.idx].toFixed(1)+'mm · W: '+Math.round(st.hi-st.lo)+' L: '+Math.round((st.hi+st.lo)/2)+
    ' · Spacing: '+s.spacing[1].toFixed(2)+'×'+s.spacing[0].toFixed(2)+'mm'+(s.hasEnh?(useEnh?' [ENHANCED]':' [ORIGINAL]'):'');
}

const drawAll=()=>{for(let i=0;i<N;i++)draw(i);};

function setIdx(i,v,fromSync){
  const s=P.series[i],st=S[i];if(!s)return;v=Math.max(0,Math.min(s.pos.length-1,v));if(v===st.idx&&!fromSync)return;
  cancelAngle(i);st.idx=v;draw(i);
  if(opt.sync&&!fromSync){
    for(let j=0;j<N;j++){
      if(j===i||!P.series[j]||S[j].panel.hidden||P.series[j].plane!==s.plane)continue;
      const pj=P.series[j].pos;let best=0,bd=Infinity;
      for(let k=0;k<pj.length;k++){const d=Math.abs(pj[k]-s.pos[v]);if(d<bd){bd=d;best=k;}}
      if(best!==S[j].idx){cancelAngle(j);S[j].idx=best;draw(j);}
    }
  }
}

function resetView(i){const s=P.series[i],st=S[i];if(!s)return;st.zoom=1;st.px=st.py=0;st.lo=s.wl.auto[0];st.hi=s.wl.auto[1];draw(i);}
function preset(i,name){const s=P.series[i],st=S[i],w=s.wl[name]||s.wl.auto;st.lo=w[0];st.hi=w[1];draw(i);}

function angleDegrees(s,m){
 if(m.x3===undefined)return null;
 const ax=(m.x1-m.x2)*s.spacing[1],ay=(m.y1-m.y2)*s.spacing[0],bx=(m.x3-m.x2)*s.spacing[1],by=(m.y3-m.y2)*s.spacing[0];
 const den=Math.hypot(ax,ay)*Math.hypot(bx,by);return den<1e-12?null:Math.acos(Math.max(-1,Math.min(1,(ax*bx+ay*by)/den)))*180/Math.PI;
}
function cancelAngle(i){const st=S[i];if(st.pendingAngle){const a=st.meas[st.pendingAngle.idx]||[];st.meas[st.pendingAngle.idx]=a.filter(m=>m!==st.pendingAngle.m);st.pendingAngle=null;}}
async function roiStats(i,m,idx){
 const s=P.series[i],data=await getFrame(i,idx,false);
 const left=Math.min(m.x1,m.x2),right=Math.max(m.x1,m.x2),top=Math.min(m.y1,m.y2),bottom=Math.max(m.y1,m.y2);
 const cx=(left+right)/2,cy=(top+bottom)/2,rx=(right-left)/2,ry=(bottom-top)/2,values=[];
 for(let y=Math.max(0,Math.ceil(top));y<=Math.min(s.rows-1,Math.floor(bottom));y++)for(let x=Math.max(0,Math.ceil(left));x<=Math.min(s.cols-1,Math.floor(right));x++){
  if(m.type==='ellipse'&&(rx===0||ry===0||((x-cx)/rx)**2+((y-cy)/ry)**2>1))continue;
  values.push(val(s,data[y*s.cols+x]));
 }
 if(!values.length){m.txt='No pixels in ROI';draw(i);return;}
 values.sort((a,b)=>a-b);const n=values.length,mean=values.reduce((a,b)=>a+b,0)/n,sd=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/n),mn=values[0],mx=values[n-1],median=n%2?values[(n-1)/2]:(values[n/2-1]+values[n/2])/2;
 const bins=new Array(64).fill(0),width=(mx-mn)/64;
 for(const v of values)bins[width?Math.min(63,Math.floor((v-mn)/width)):0]++;
 const peak=bins.indexOf(Math.max(...bins)),mode=width?mn+(peak+.5)*width:mn;
 m.txt='Original pixels\nMean: '+mean.toFixed(2)+' · SD: '+sd.toFixed(2)+'\nMedian: '+median.toFixed(2)+'\nMode (64 bins): '+mode.toFixed(2)+'\nMin/Max: '+mn.toFixed(2)+' / '+mx.toFixed(2)+'\nArea: '+(n*s.spacing[0]*s.spacing[1]).toFixed(2)+' mm²';draw(i);
}

const bar=document.getElementById('bar');
const syncG=$('div',{className:'tool-group'});
syncG.append($('div',{className:'tool-label',textContent:'LINK'}));
syncG.append(chk('Scroll Linked','sync',true));
syncG.append(chk('W/L Linked','link',false));
bar.append(syncG);

const icons={
 wl:'<circle cx="12" cy="12" r="8"/><path d="M12 4v16M12 4a8 8 0 0 1 0 16Z"/>',
 pan:'<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
 zoom:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6M10 7v6"/>',
 dist:'<path d="m4 18 14-14 3 3L7 21ZM9 13l2 2M13 9l2 2"/>',
 ellipse:'<ellipse cx="12" cy="12" rx="9" ry="6"/>',
 angle:'<path d="M4 5v15h16M4 14a6 6 0 0 1 6 6"/>',
 arrow:'<path d="m4 20 16-16M10 4h10v10"/>',
 text:'<path d="M5 5h14M12 5v15M8 20h8"/>',
 reset:'<path d="M4 9a8 8 0 1 1 0 7M4 4v5h5"/>',
 mpr:'<path d="M4 4h16v16H4ZM12 4v16M4 12h16"/>',
 mip:'<path d="m3 8 9-5 9 5-9 5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
 vr:'<path d="m4 7 8-4 8 4v10l-8 4-8-4ZM4 7l8 4 8-4M12 11v10"/>',
 roi:'<rect x="4" y="4" width="16" height="16" rx="1" stroke-dasharray="3 3"/><path d="M8 12h8M12 8v8"/>'};
function toolButton(x){
 const button=$('button',{className:'tool-button',title:x.l,ariaLabel:x.l});
 button.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true">'+icons[x.t]+'</svg><span>'+x.l+'</span>';
 return button;
}
const toolG=document.getElementById('tool-rail');
const toolStatus=$('span',{id:'tool-status',textContent:'Window / level · drag to adjust'});bar.append(toolStatus);
const annotationInput=$('input',{id:'annotation-text',type:'text',placeholder:'Annotation text',maxLength:160,ariaLabel:'Annotation text'});annotationInput.hidden=true;bar.append(annotationInput);
const groups=[['VIEW',[{t:'wl',l:'Window / level'},{t:'pan',l:'Pan'},{t:'zoom',l:'Zoom'}]],['MEASURE',[{t:'dist',l:'Distance'},{t:'angle',l:'Angle'}]],['ROI',[{t:'roi',l:'Rectangle ROI'},{t:'ellipse',l:'Ellipse ROI'}]],['NOTES',[{t:'arrow',l:'Arrow annotation'},{t:'text',l:'Text annotation'}]]];
groups.forEach(([name,items])=>{
 toolG.append($('div',{className:'rail-category',textContent:name}));
 items.forEach(x=>{const button=toolButton(x);button.classList.toggle('on',x.t==='wl');button.setAttribute('aria-pressed',String(x.t==='wl'));
 button.onclick=()=>{S.forEach((_,i)=>cancelAngle(i));opt.tool=x.t;toolG.querySelectorAll('button').forEach(b=>{b.classList.remove('on');b.setAttribute('aria-pressed','false');});button.classList.add('on');button.setAttribute('aria-pressed','true');annotationInput.hidden=!['text','arrow'].includes(x.t);toolStatus.textContent=x.l+(x.t==='angle'?' · click endpoint, vertex, endpoint':x.t==='text'?' · enter text, then click image':' · drag on image');drawAll();};toolG.append(button);});
 toolG.append($('div',{className:'rail-separator'}));
});
const fit=toolButton({t:'reset',l:'Fit / reset active viewport'});fit.onclick=()=>resetView(hover);toolG.append(fit);
toolG.append($('div',{className:'rail-category',textContent:'3D'}));
[{t:'mpr',l:'MPR'},{t:'mip',l:'MIP'},{t:'vr',l:'VR'}].forEach(x=>{const b=toolButton(x);b.disabled=true;b.title=x.l+' — volume reconstruction will be added in a later update';toolG.append(b);});
const overlayControls=$('div',{className:'overlay-controls'});
const overlayList=$('select',{id:'overlay-list',ariaLabel:'Measurement or annotation'});let overlaySignature='';
function refreshOverlays(){if(!S[hover])return;const list=S[hover].meas[S[hover].idx]||[],signature=hover+':'+S[hover].idx+':'+list.map(m=>m.id+':'+(m.label||'')).join(',');if(signature===overlaySignature)return;overlaySignature=signature;const prev=overlayList.value;overlayList.replaceChildren($('option',{value:'',textContent:'Select overlay'}));list.forEach((m,k)=>overlayList.append($('option',{value:m.id,textContent:(k+1)+'. '+m.type+(m.label?' · '+m.label:'')})));if(list.some(m=>String(m.id)===prev))overlayList.value=prev;remove.hidden=!overlayList.value;const selected=list.find(m=>String(m.id)===overlayList.value);edit.hidden=!(selected&&['text','arrow'].includes(selected.type));}
const remove=$('button',{textContent:'Delete selected',hidden:true});remove.onclick=()=>{const st=S[hover];cancelAngle(hover);st.meas[st.idx]=(st.meas[st.idx]||[]).filter(m=>String(m.id)!==overlayList.value);draw(hover);};
const edit=$('button',{textContent:'Update note',hidden:true});edit.onclick=()=>{const st=S[hover],m=(st.meas[st.idx]||[]).find(m=>String(m.id)===overlayList.value);if(m&&['text','arrow'].includes(m.type)&&annotationInput.value.trim()){m.label=annotationInput.value.trim();draw(hover);}};
overlayList.onchange=()=>{remove.hidden=!overlayList.value;const m=(S[hover].meas[S[hover].idx]||[]).find(m=>String(m.id)===overlayList.value);edit.hidden=!(m&&['text','arrow'].includes(m.type));annotationInput.hidden=edit.hidden;if(!annotationInput.hidden)annotationInput.value=m.label||'';};
overlayControls.append(overlayList,remove,edit);bar.append(overlayControls);
bar.append(chk('Show overlays','overlays',true));
let nextOverlayId=1;
const layoutPicker=$('div',{className:'layout-picker'}),layout=$('button',{textContent:'Layout: 2 × 2',ariaLabel:'Viewport layout'});
const layoutMenu=$('div',{className:'layout-menu',hidden:true}),layoutCaption=$('div',{textContent:'2 rows × 2 columns'}),matrix=$('div',{className:'layout-matrix'});
const layoutCells=[];
for(let r=1;r<=4;r++)for(let c=1;c<=4;c++){
 const cell=$('button',{className:'layout-cell',title:r+' rows × '+c+' columns',ariaLabel:r+' rows by '+c+' columns'});
 cell.onmouseenter=()=>{layoutCaption.textContent=r+' rows × '+c+' columns';layoutCells.forEach(({button,row,col})=>button.classList.toggle('preview',row<=r&&col<=c));};
 cell.onclick=()=>{setLayout(r,c);layoutMenu.hidden=true;};matrix.append(cell);layoutCells.push({button:cell,row:r,col:c});
}
layoutMenu.append(layoutCaption,matrix);layoutPicker.append(layout);bar.append(layoutPicker);document.body.append(layoutMenu);
layout.onclick=()=>{layoutMenu.hidden=!layoutMenu.hidden;const b=layout.getBoundingClientRect();layoutMenu.style.top=(b.bottom+4)+'px';layoutMenu.style.left=Math.min(b.left,window.innerWidth-150)+'px';};
document.addEventListener('pointerdown',e=>{if(!layoutMenu.contains(e.target)&&e.target!==layout)layoutMenu.hidden=true;});
function setLayout(rows,cols){
 layoutRows=rows;layoutCols=cols;layout.textContent='Layout: '+rows+' × '+cols;
 grid.style.gridTemplateColumns='repeat('+cols+',minmax(0,1fr))';grid.style.gridTemplateRows='repeat('+rows+',minmax(0,1fr))';grid.classList.toggle('dense',rows>=3||cols>=3);
 S.forEach((st,i)=>{st.panel.hidden=i>=rows*cols;if(st.panel.hidden&&st.play){clearInterval(st.play);st.play=null;}});
 if(hover>=rows*cols)activate(0);requestAnimationFrame(drawAll);
}

const clr=$('button',{textContent:'Clear all overlays'});clr.onclick=()=>{S.forEach(st=>{st.meas={};st.pendingAngle=null;});drawAll();};bar.append(clr);

if(P.patient){bar.append($('div',{className:'pat-banner',textContent:P.patient}));}

function chk(label,key,init){
  const c=$('input',{type:'checkbox',checked:init,ariaLabel:label});
  c.onchange=()=>{opt[key]=c.checked;if(key==='overlays')drawAll();};
  return $('label',null,[c,document.createTextNode(label)]);
}

const grid=document.getElementById('grid');
const rail=document.getElementById('series-rail');
function activate(i){hover=i;refreshOverlays();document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('active-viewport',j===i));rail.querySelectorAll('.series-card').forEach((p,j)=>p.classList.toggle('active',catalog[j]===P.series[i]));}
async function thumbnail(i,cv){const s=P.series[i],data=await getFrame(i,Math.floor(s.pos.length/2),false);cv.width=s.cols;cv.height=s.rows;const ctx=cv.getContext('2d'),im=ctx.createImageData(s.cols,s.rows),[lo,hi]=s.wl.auto;for(let k=0;k<data.length;k++){let v=Math.max(0,Math.min(255,255*(val(s,data[k])-lo)/Math.max(1e-9,hi-lo)));if(s.invert)v=255-v;im.data[k*4]=im.data[k*4+1]=im.data[k*4+2]=v;im.data[k*4+3]=255;}ctx.putImageData(im,0,0);}
P.series.forEach((item,i)=>{
  let s=item||catalog[0];
  const st=S[i],cv=$('canvas'),stat=$('div',{className:'st'}),pnl=$('div',{className:'panel'});
  if(i===0) pnl.classList.add('active-viewport');
  st.cv=cv;st.stat=stat;st.panel=pnl;pnl.hidden=i>=4;
  
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
  const hint=$('div',{className:'drop-hint',textContent:'Drop a series here',hidden:!!item});pnl.append(hint);
  st.assign=(index)=>{
    cancelAngle(i);if(st.play){clearInterval(st.play);st.play=null;bPlay.className='';}
    st.tok++;s=catalog[index];P.series[i]=s;
    Object.assign(st,{idx:Math.floor(s.pos.length/2),lo:s.wl.auto[0],hi:s.wl.auto[1],zoom:1,px:0,py:0,enh:s.hasEnh,inv:!!s.invert,cache:new Map(),meas:{},lay:null});
    pnl.querySelector('.ttl').textContent=s.title+(s.hasEnh?' · SNR:'+s.snr:'');
    sel.replaceChildren();['auto','dicom','full'].forEach(n=>{if(n!=='dicom'||s.wl.dicom)sel.append($('option',{value:n,textContent:'W/L '+n}));});
    bEnh.style.display=s.hasEnh?'':'none';bEnh.className=st.enh?'on':'';bInv.className=st.inv?'on':'';hint.hidden=true;activate(i);draw(i);
  };
  pnl.ondragover=e=>{if(e.dataTransfer.types.includes('application/x-dicom-series')){e.preventDefault();e.dataTransfer.dropEffect='copy';pnl.classList.add('drop-target');}};
  pnl.ondragleave=()=>pnl.classList.remove('drop-target');
  pnl.ondrop=e=>{e.preventDefault();pnl.classList.remove('drop-target');const raw=e.dataTransfer.getData('application/x-dicom-series');if(!raw)return;const index=Number(raw);if(Number.isInteger(index)&&index>=0&&index<catalog.length)st.assign(index);};
  pnl.onclick=()=>activate(i);

  cv.addEventListener('mouseenter',()=>{
    activate(i);
  });
  cv.addEventListener('wheel',e=>{
    e.preventDefault();if(!P.series[i])return;
    if(e.ctrlKey||e.metaKey){st.zoom=Math.max(0.2,Math.min(20,st.zoom*Math.exp(-e.deltaY*0.01)));draw(i);}
    else setIdx(i,st.idx+(e.deltaY>0?1:-1));
  },{passive:false});
  
  cv.addEventListener('contextmenu',e=>e.preventDefault());
  let drag=null;
  cv.addEventListener('pointerdown',e=>{
    if(!P.series[i])return;activate(i);cv.setPointerCapture(e.pointerId);
    const tool=e.button===1?'pan':(e.shiftKey?'dist':opt.tool);
    drag={tool,x:e.clientX,y:e.clientY};
    if(!st.lay){drag=null;return;}
    if(['dist','roi','ellipse','arrow','text','angle'].includes(tool)){
      const rc=cv.getBoundingClientRect();let [x,y]=toImage(st,e.clientX-rc.left,e.clientY-rc.top);
      if(x<0||y<0||x>s.cols-1||y>s.rows-1){drag=null;return;}
      if(tool==='angle'){
        if(!st.pendingAngle){const m={id:nextOverlayId++,type:'angle',x1:x,y1:y,x2:x,y2:y};(st.meas[st.idx]=st.meas[st.idx]||[]).push(m);st.pendingAngle={idx:st.idx,m,step:1};}
        else{const p=st.pendingAngle;if(p.step===1){p.m.x2=x;p.m.y2=y;p.step=2;}else{p.m.x3=x;p.m.y3=y;st.pendingAngle=null;}}
        drag=null;draw(i);
      }else if(tool==='text'){
        const label=annotationInput.value.trim();if(label)(st.meas[st.idx]=st.meas[st.idx]||[]).push({id:nextOverlayId++,type:tool,x1:x,y1:y,x2:x,y2:y,label});drag=null;draw(i);
      }else{const m={id:nextOverlayId++,type:tool,x1:x,y1:y,x2:x,y2:y,label:tool==='arrow'?annotationInput.value.trim():''};(st.meas[st.idx]=st.meas[st.idx]||[]).push(m);drag.m=m;drag.idx=st.idx;}
    }
    e.preventDefault();
  });
  
  cv.addEventListener('pointermove',e=>{
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
    if(drag.tool==='wl'){
      const targets=opt.link?[...Array(N).keys()].filter(j=>P.series[j]&&!S[j].panel.hidden):[i];
      targets.forEach(j=>{const sj=P.series[j],stj=S[j],r0=Math.max(1e-9,sj.wl.auto[1]-sj.wl.auto[0])/300;
        let c=(stj.hi+stj.lo)/2,w=stj.hi-stj.lo;w=Math.max(r0*2,w+dx*r0);c-=dy*r0;stj.lo=c-w/2;stj.hi=c+w/2;draw(j);});
    }else if(drag.tool==='zoom'){st.zoom=Math.max(.2,Math.min(20,st.zoom*Math.exp(-dy*.015)));draw(i);}
    else if(drag.tool==='pan'){st.px+=dx;st.py+=dy;draw(i);}
    else if(drag.m){const rc=cv.getBoundingClientRect(),[x,y]=toImage(st,e.clientX-rc.left,e.clientY-rc.top);drag.m.x2=Math.max(0,Math.min(s.cols-1,x));drag.m.y2=Math.max(0,Math.min(s.rows-1,y));draw(i);}
  });
  const end=()=>{if(drag&&drag.m&&['roi','ellipse'].includes(drag.m.type))roiStats(i,drag.m,drag.idx);drag=null;};
  cv.addEventListener('pointerup',end);cv.addEventListener('pointercancel',end);
  cv.addEventListener('dblclick',()=>resetView(i));
});

catalog.forEach((s,index)=>{
 const thumb=$('canvas'),card=$('button',{className:'series-card'+(index===0?' active':''),title:s.title+' · drag into a viewport',ariaLabel:'Load '+s.title,draggable:true});
 card.append(thumb,$('span',{className:'name',textContent:s.title}),$('span',{className:'meta',textContent:s.plane+' · '+s.pos.length+' images'}));
 card.ondragstart=e=>{e.dataTransfer.setData('application/x-dicom-series',String(index));e.dataTransfer.effectAllowed='copy';};
 card.onclick=()=>S[hover].assign(index);rail.append(card);
 // Decode thumbnail independently from viewport assignment.
 (async()=>{const data=await inflate(s.items[Math.floor(s.pos.length/2)].o);thumb.width=s.cols;thumb.height=s.rows;const ctx=thumb.getContext('2d'),im=ctx.createImageData(s.cols,s.rows),[lo,hi]=s.wl.auto;for(let k=0;k<data.length;k++){let v=Math.max(0,Math.min(255,255*(val(s,data[k])-lo)/Math.max(1e-9,hi-lo)));if(s.invert)v=255-v;im.data[k*4]=im.data[k*4+1]=im.data[k*4+2]=v;im.data[k*4+3]=255;}ctx.putImageData(im,0,0);})();
});
setLayout(2,2);
new ResizeObserver(()=>requestAnimationFrame(drawAll)).observe(grid);

window.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  if(e.key==='Escape'){S.forEach((_,i)=>cancelAngle(i));drawAll();return;}
  if(hover<0||!P.series[hover])return;const st=S[hover],k=e.key.toLowerCase();
  if(k==='e'&&P.series[hover].hasEnh)st.togEnh();else if(k==='i')st.togInv();else if(k==='r')resetView(hover);
  else if(k===' '){st.togPlay();e.preventDefault();}
  else if(k==='arrowdown'||k==='arrowright'){setIdx(hover,st.idx+1);e.preventDefault();}
  else if(k==='arrowup'||k==='arrowleft'){setIdx(hover,st.idx-1);e.preventDefault();}
});
window.addEventListener('resize',drawAll);
drawAll();
</script>
"""


def build_empty_workspace():
    theme = VIEWER_HTML.split("</style>")[0] + "</style>"
    theme = theme.replace("__COLS__", "2").replace("__PANEL_H__", "300")
    icon = '<svg viewBox="0 0 48 48"><rect x="5" y="5" width="38" height="38" rx="3"/><path d="M16 24h16M24 16v16"/></svg>'
    tools = '<div id="bar"><div class="brand-tag">IMAGE TOOLS</div>'
    for label in []:
        tools += f'<button disabled>{label}</button>'
    tools += '<span style="margin-left:auto;color:#6b8198;font-size:10px">2 × 2 workspace · Awaiting study</span></div>'
    rail = '<nav id="series-rail"><div class="rail-heading">LOADED SERIES</div><div class="rail-empty">+</div><div class="rail-hint">No series available.<br>Import a study using the panel on the left.</div></nav>'
    cells = ''
    for i in range(4):
        title = "Ready for your next study" if i == 0 else "Viewport available"
        hint = "Open a DICOM ZIP from the study browser" if i == 0 else "Selected series will appear here"
        cells += f'<div class="panel"><div class="empty-canvas"><div class="corner">VIEWPORT {i+1:02d} / NO SERIES</div><div class="center">{icon}<b>{title}</b>{hint}</div><div class="bottom">W — &nbsp; L — &nbsp; / &nbsp; IMAGE —</div></div></div>'
    tool_rail = '<nav id="tool-rail" aria-label="Image tools">' + ''.join(f'<button class="tool-button" disabled title="{name}: load a study first" aria-label="{name}">{symbol}</button>' for name, symbol in [('Window / level','◐'),('Pan','✥'),('Zoom','⊕'),('Distance','╱'),('Angle','∠'),('Rectangle ROI','□'),('Ellipse ROI','◯'),('Arrow annotation','↗'),('Text annotation','T'),('MPR','MPR'),('MIP','MIP'),('VR','VR')]) + '</nav>'
    fit_script = '<script>' + VIEWER_HTML.split('<script>')[1].split('const P=')[0] + '</script>'
    return theme + tools + '<div class="workspace">' + rail + tool_rail + '<div id="grid">' + cells + '</div></div><div class="workspace-footer"><span>No image data loaded</span><span>Window / level · Measurements · Linked scrolling · Cine</span></div>' + fit_script


def build_viewer_html(series_payloads, patient_text=None, panel_height=360):
    n = len(series_payloads)
    cols = 2 if n >= 2 else 1
    payload = dict(series=series_payloads, patient=patient_text)
    data = json.dumps(payload, separators=(",", ":")).replace("</", r"<\/")
    html = (VIEWER_HTML.replace("__PAYLOAD__", data).replace("__COLS__", str(cols)).replace("__PANEL_H__", str(panel_height)))
    rows = -(-n // cols)
    return html, 760

# ----------------------------------------------------------------------------------------------------------------
#  Unified Application Presentation Layer (Streamlit Frontend Hub)
# ----------------------------------------------------------------------------------------------------------------
CSS = """
<style>
html,body,[data-testid="stAppViewContainer"]{background:#080c12!important;color:#d4deea!important;font-family:Arial,sans-serif}
[data-testid="stHeader"]{background:#080c12!important}
[data-testid="stMainBlockContainer"]{height:100vh;overflow:hidden;max-width:none;padding:3rem 1.1rem .3rem!important}
[data-testid="stVerticalBlock"]{gap:.65rem}
[data-testid="stSidebar"]{background:#101720!important;border-right:1px solid #26313f;min-width:260px;max-width:290px}
[data-testid="stSidebarUserContent"]{padding:1.1rem 1rem!important}
[data-testid="stSidebar"] h3{color:#d4deea!important;font-size:15px;letter-spacing:-.2px;padding-top:0}
[data-testid="stSidebar"] p{color:#98aabd;font-size:12px}
[data-testid="stWidgetLabel"] p{color:#c1cedb!important;font-size:12px}
[data-testid="stFileUploaderDropzone"]{background:#151f2b;border:1px dashed #43546a;border-radius:6px}
[data-testid="stFileUploaderDropzone"] button{background:#20364a;color:#bce9ff;border-color:#3a6b88}
[data-baseweb="select"]>div{background:#17212d!important;color:#d4deea!important;border-color:#344357!important}
[data-testid="stExpander"]{background:#121b26;border-color:#2c3a4a}
[data-testid="stSidebar"] hr{margin:14px 0;border-color:#26313f}
[data-testid="stCheckbox"] label,[data-testid="stToggle"] label{color:#c1cedb}
[data-testid="stFileUploaderDropzone"] small{color:#7790a8!important}
[data-testid="stFileUploaderFile"]{background:#16212e!important;color:#bfd0e1!important;border-radius:4px}
[data-testid="stMultiSelect"] [data-rac][role="group"], [data-testid="stSelectbox"] [data-rac][role="group"]{background:#17212d!important;border-color:#344357!important;color:#d4deea!important}
[data-testid="stMultiSelect"] [data-tag]{background:#1b445b!important;color:#b6e8fb!important}
[data-testid="stMultiSelect"] input,[data-testid="stSelectbox"] input{color:#d4deea!important}
[data-testid="stMultiSelect"] button,[data-testid="stSelectbox"] button{color:#9cb4c9!important}
[data-baseweb="tag"]{background:#1b445b!important;color:#b6e8fb!important}
[data-baseweb="select"] input,[data-baseweb="select"] span{color:#d4deea!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong{color:#c7d7e6}

.main-header{height:53px;display:flex;align-items:center;gap:16px;border-bottom:1px solid #26313f;margin-bottom:2px}
.wordmark{font-size:21px;font-weight:700;letter-spacing:-1px;color:#f1f7ff;display:flex;align-items:center;gap:9px}
.logo-mark{width:27px;height:27px;border:1px solid #55bcdf;border-radius:7px;display:grid;place-items:center;color:#55bcdf;font-size:23px}
.brand-sub{font-size:10px;letter-spacing:1.4px;color:#6c8199;line-height:1.5}
.header-tab{align-self:stretch;display:flex;align-items:center;border-bottom:2px solid #55bcdf;color:#d4deea;font-size:12px;padding:0 14px;margin-left:26px}
.header-meta{margin-left:auto;font-size:11px;color:#8499b0;display:flex;gap:15px;align-items:center}
.status-dot{width:6px;height:6px;border-radius:100%;background:#55bcdf;display:inline-block;margin-right:6px}
.study-strip{display:flex;gap:18px;align-items:center;padding:10px 14px;background:#111923;border:1px solid #26313f;border-radius:4px;font-size:12px;color:#a4b7ca;flex-wrap:wrap}
.study-strip b{color:#d9e6f2;font-weight:500}
.study-strip .right{margin-left:auto;color:#6f849a;font-size:11px}
@media(max-width:700px){.header-tab,.brand-sub{display:none}.header-meta{font-size:10px}.main-header{gap:8px}}
</style>
"""


# Curated educational cards. Sources checked 2026-10-11; no network fetch during loading.
LOADING_FACTS = [
    dict(category="RADIOLOGY HISTORY · 1895", title="A new way to see inside the body",
         text="Wilhelm Röntgen discovered X-rays in 1895 and received the first Nobel Prize in Physics in 1901.",
         source="Nobel Prize", url="https://www.nobelprize.org/prizes/physics/1901/rontgen/facts/"),
    dict(category="RADIOLOGY HISTORY · 1979", title="Recognising the development of CT",
         text="Allan Cormack and Godfrey Hounsfield shared the 1979 Nobel Prize in Physiology or Medicine for developing computer-assisted tomography.",
         source="Nobel Prize", url="https://www.nobelprize.org/prizes/medicine/1979/summary/"),
    dict(category="MRI HISTORY · 2003", title="MRI's Nobel milestone",
         text="Paul Lauterbur and Peter Mansfield shared the 2003 Nobel Prize in Physiology or Medicine for discoveries concerning magnetic resonance imaging.",
         source="Nobel Prize", url="https://www.nobelprize.org/prizes/medicine/2003/press-release/"),
    dict(category="MRI PRINCIPLES", title="Signals from fat and water",
         text="MRI uses strong magnetic fields and radiofrequency energy. Its signal comes mainly from protons in fat and water molecules.",
         source="US FDA", url="https://www.fda.gov/radiation-emitting-products/medical-imaging/mri-magnetic-resonance-imaging"),
    dict(category="MRI PRINCIPLES", title="Locating the signal",
         text="Lauterbur showed that magnetic-field gradients could identify where signals originated, enabling two-dimensional MR images.",
         source="Nobel Prize", url="https://www.nobelprize.org/prizes/medicine/2003/press-release/"),
    dict(category="MRI PRINCIPLES", title="A different way to form an image",
         text="MRI forms images without ionizing radiation, using static and changing magnetic fields together with radiofrequency energy.",
         source="US FDA", url="https://www.fda.gov/radiation-emitting-products/mri-magnetic-resonance-imaging/benefits-and-risks"),
]


def build_loading_html():
    facts = json.dumps(LOADING_FACTS, ensure_ascii=True).replace("</", r"<\/")
    return r"""
<style>
*{box-sizing:border-box}body{margin:0;background:#080c12;color:#d4deea;font:13px -apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif}
.loading{height:410px;border:1px solid #26313f;border-radius:5px;background:radial-gradient(ellipse at 24% 50%,#142633,#0d141d 55%);display:grid;grid-template-columns:35% 65%;overflow:hidden}
.visual{display:grid;place-items:center;position:relative;border-right:1px solid #26313f}
.scanner{width:154px;height:154px;position:relative;border:1px solid #284355;border-radius:50%;display:grid;place-items:center}
.scanner:before,.scanner:after{content:"";position:absolute;inset:12px;border:1px solid #315970;border-radius:50%}.scanner:after{inset:35px;border-color:#294455}
.orbit{position:absolute;inset:-8px;border:2px solid transparent;border-top-color:#65c4e1;border-radius:50%;animation:orbit 6s linear infinite}
.scanline{position:absolute;width:112px;height:1px;background:#65c4e1;box-shadow:0 0 12px #55bcdf80;animation:scan 4s ease-in-out infinite}
.cross{font-size:28px;color:#567c92;font-weight:200}.visual-caption{position:absolute;bottom:39px;letter-spacing:2px;font-size:9px;color:#7296ac}
.card{display:flex;flex-direction:column;justify-content:center;padding:34px;min-width:0}
.kicker{font-size:9px;letter-spacing:1.8px;color:#6ab4d0;margin-bottom:18px}h2{font-size:23px;line-height:1.3;font-weight:500;margin:0 0 15px;color:#e0edf7}
p{color:#a5b8ca;line-height:1.8;font-size:14px;margin:0;max-width:590px;min-height:78px}.meta{font-size:10px;color:#70879c;margin-top:17px}.meta a{color:#77bbd4;text-decoration:none}
.controls{display:flex;align-items:center;gap:9px;margin-top:27px}button{border:1px solid #34495c;background:#142432;color:#aac4d8;padding:6px 11px;border-radius:4px;font-size:11px;cursor:pointer}button:hover{border-color:#65c4e1}button:focus-visible,a:focus-visible{outline:2px solid #65c4e1;outline-offset:3px}.counter{font-size:10px;color:#71889d;margin-left:auto}
.paused .orbit,.paused .scanline{animation-play-state:paused}
@keyframes orbit{to{transform:rotate(360deg)}}@keyframes scan{0%,100%{transform:translateY(-46px);opacity:.45}50%{transform:translateY(46px);opacity:1}}
@media(prefers-reduced-motion:reduce){.orbit,.scanline{animation:none}}
@media(max-width:650px){.loading{grid-template-columns:1fr;height:410px}.visual{display:none}.card{padding:24px}h2{font-size:20px}p{font-size:13px}}
</style>
<div class="loading"><div class="visual" aria-hidden="true"><div class="scanner"><div class="orbit"></div><div class="scanline"></div><span class="cross">+</span></div><span class="visual-caption">PREPARING YOUR STUDY</span></div>
<section class="card" aria-label="Radiology facts during study processing"><div class="kicker" id="category"></div><h2 id="title"></h2><p id="fact"></p><div class="meta">Source: <a id="source" target="_blank" rel="noopener noreferrer"></a> · Reviewed 11 Oct 2026</div><div class="controls"><button id="pause" aria-pressed="false">Pause</button><button id="next">Next fact</button><span class="counter" id="counter"></span></div></section></div>
<script>
const facts=__FACTS__;let index=0,paused=matchMedia('(prefers-reduced-motion: reduce)').matches;
const pause=document.getElementById('pause');
function show(){const f=facts[index];document.getElementById('category').textContent=f.category;document.getElementById('title').textContent=f.title;document.getElementById('fact').textContent=f.text;const link=document.getElementById('source');link.textContent=f.source;link.href=f.url;document.getElementById('counter').textContent=(index+1)+' / '+facts.length;}
function state(){document.body.classList.toggle('paused',paused);pause.textContent=paused?'Resume':'Pause';pause.setAttribute('aria-pressed',String(paused));}
pause.onclick=()=>{paused=!paused;state();};document.getElementById('next').onclick=()=>{index=(index+1)%facts.length;show();};
const timer=setInterval(()=>{if(!paused&&!document.hidden){index=(index+1)%facts.length;show();}},10000);
window.addEventListener('pagehide',()=>clearInterval(timer));show();state();
</script>
""".replace("__FACTS__", facts)


def _cache_resource(fn):
    return st.cache_resource(show_spinner=False)(fn) if st else fn

@_cache_resource
def load_index(file_hash, _zbytes, _progress=None):
    named, skipped = read_zip(_zbytes, progress=_progress)
    series, dup = group_series(named)
    return series, skipped, dup

def main():
    st.set_page_config(page_title="SD-DCM | Clinical Workstation", page_icon="◉", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    st.markdown(
        '<div class="main-header"><div class="wordmark"><span class="logo-mark">+</span> SD-DCM</div>'
        '<div class="brand-sub">CLINICAL<br>WORKSTATION</div><div class="header-tab">Study workspace</div>'
        '<div class="header-meta"><span><i class="status-dot"></i>Image review</span><span>Research use</span></div></div>',
        unsafe_allow_html=True,
    )
    sb = st.sidebar
    sb.markdown("### Study browser")
    sb.caption("DICOM ARCHIVE")
    up = sb.file_uploader("Open DICOM study", type=["zip"],
                          help="Upload a ZIP archive containing DICOM images.")
    sb.caption("Files are processed by the Streamlit server hosting this app.")
    sb.divider()
    sb.caption("PROCESSING")
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
        st.markdown('<div class="study-strip"><b>No study loaded</b><span>Open a DICOM ZIP in the study browser</span>'
                    '<span class="right">Identifiers hidden by default</span></div>', unsafe_allow_html=True)
        st.components.v1.html(build_empty_workspace(), height=760, scrolling=False)
        return

    zbytes = up.getvalue()
    fhash = hashlib.sha1(zbytes).hexdigest()
    
    loading = st.empty()
    with loading.container():
        progress_bar = st.progress(0.0, text="Reading DICOM images and grouping series...")
        st.components.v1.html(build_loading_html(), height=420, scrolling=False)
    try:
        series, skipped, dup = load_index(fhash, zbytes,
            _progress=lambda fraction: progress_bar.progress(0.15 * fraction, text="Reading DICOM archive..."))
    except Exception as exc:
        loading.empty()
        st.error(f"Could not read this archive: {exc}")
        return

    if not series:
        loading.empty()
        st.error("No readable DICOM images found. Check the ZIP contents and pixel-data support.")
        return
        
    total_frames = sum(sum(max(1, int(getattr(d, "NumberOfFrames", 1) or 1)) for d in item["items"]) for item in series)
    st.markdown(
        f'<div class="study-strip"><b>Study loaded</b><span>{len(series)} series</span><span>{total_frames} images</span>'
        '<span class="right">ZIP archive · Select series in the study browser</span></div>', unsafe_allow_html=True)

    # Study Browser Ingestion Logs
    if skipped or dup:
        with st.expander("Import details", expanded=False):
            for why, n in skipped.items():
                st.markdown(f"⚠️ *Skipped Entry:* `{n} instances` flagged as: **{why}**")
            if dup:
                st.markdown(f"Duplicate instances removed: `{dup}` (matching SOPInstanceUID).")

    # Show every series from the selected study in the in-viewer thumbnail browser.
    study_groups = {}
    for item in series:
        ds = item["items"][0]
        study_key = (str(getattr(ds, "StudyInstanceUID", "")), str(getattr(ds, "PatientID", "")))
        study_groups.setdefault(study_key, []).append(item)
    if len(study_groups) > 1:
        keys = list(study_groups)
        selected_study = sb.selectbox("Study", range(len(keys)),
            format_func=lambda i: f"Study {i + 1} · {len(study_groups[keys[i]])} series")
        series = study_groups[keys[selected_study]]
    labels = {item["label"] + f" [{item['uid'][-6:]}]": item for item in series}
    chosen = list(labels)
    sb.caption(f"{len(chosen)} series · All listed beside the image workspace")

    # Load Progress Vector mapping
    payloads = []
    progress_bar.progress(0.15, text="Preparing image workspace...")
    
    for idx, name in enumerate(chosen):
        s = labels[name]
        try:
            progress_bar.progress(0.15 + 0.85 * idx / len(chosen), text=f"Preparing series {idx + 1}/{len(chosen)}: {s['desc']}...")
            payloads.append(build_payload_series(s, bool(enhance), params,
                progress=lambda fraction, current=idx: progress_bar.progress(
                    0.15 + 0.85 * (current + fraction) / len(chosen),
                    text=f"Enhancing series {current + 1}/{len(chosen)}...")))
        except Exception as e:
            st.error(f"Could not prepare series {s['label']}: {e}")
            
    loading.empty()
    
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
    st.components.v1.html(html, height=height, scrolling=False)

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
