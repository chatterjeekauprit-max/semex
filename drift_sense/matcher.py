import numpy as np
import torch
from scipy.ndimage import gaussian_filter

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def degrade_reference(ref, factor=10, psf_sigma=5.0):
    """Apply the SAME degradation the 10x optic applied: blur, then decimate.
    1000x1000 @ 1nm/px -> 100x100 @ 10nm/px."""
    a = gaussian_filter(ref.astype(np.float32), psf_sigma)
    h, w = a.shape
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))

def remove_periodic(t, period=6.0):
    """Estimate the lattice by folding the template onto one period,
    then subtract it. What remains is the aperiodic residual."""
    th, tw = t.shape
    p = int(round(period))
    ny, nx = th // p, tw // p
    core = t[:ny * p, :nx * p]
    cell = core.reshape(ny, p, nx, p).mean(axis=(0, 2))
    lattice = np.tile(cell, (ny + 1, nx + 1))[:th, :tw]
    return t - lattice


def zncc_map(search, template):
    """FFT cross-correlation of zero-mean, unit-variance images, on GPU."""
    s = torch.from_numpy(search.astype(np.float32)).to(DEV)
    t = torch.from_numpy(template.astype(np.float32)).to(DEV)

    s = (s - s.mean()) / (s.std() + 1e-6)
    t = (t - t.mean()) / (t.std() + 1e-6)

    H, W = 1152, 1152          # cuFFT-friendly: 1152 = 2^7 * 3^2

    s_hat = torch.fft.rfft2(s, s=(H, W))
    t_hat = torch.fft.rfft2(t, s=(H, W))
    corr = torch.fft.irfft2(s_hat * torch.conj(t_hat), s=(H, W))

    return corr

def zncc_map_normalized(search, template):
    """True ZNCC: local mean/variance normalisation at every position."""
    s = torch.from_numpy(search.astype(np.float32)).to(DEV)
    t = torch.from_numpy(template.astype(np.float32)).to(DEV)

    th, tw = t.shape
    n = float(th * tw)
    H, W = 1152, 1152

    # zero-mean, unit-norm template
    t = t - t.mean()
    t = t / (t.norm() + 1e-8)

    ones = torch.ones((th, tw), device=DEV, dtype=torch.float32)

    S = torch.fft.rfft2(s, s=(H, W))
    S2 = torch.fft.rfft2(s * s, s=(H, W))
    T = torch.fft.rfft2(t, s=(H, W))
    O = torch.fft.rfft2(ones, s=(H, W))

    # 1. numerator: correlation with the zero-mean, unit-norm template
    num = torch.fft.irfft2(S * torch.conj(T), s=(H, W))

    # 2. local sums of the search image and its square
    local_sum = torch.fft.irfft2(S * torch.conj(O), s=(H, W))
    local_sum2 = torch.fft.irfft2(S2 * torch.conj(O), s=(H, W))

    # 3. local variance -> denominator
    local_var = local_sum2 - (local_sum * local_sum) / n

    sh, sw = s.shape
    valid_h = sh - th + 1
    valid_w = sw - tw + 1

    # variance floor scaled to the data, not an absolute epsilon
    floor = 1e-3 * float(local_var[:valid_h, :valid_w].max())
    denom = torch.sqrt(torch.clamp(local_var, min=floor))
    out = num / denom

    # positions where the 100x100 window overhangs the image are meaningless
    mask = torch.full_like(out, float("-inf"))
    mask[:valid_h, :valid_w] = 0.0
    return out + mask

def subpixel_peak(corr, py, px):
    """Parabolic fit through the peak and its 4-neighbours.
    Returns (dx, dy) offsets in [-0.5, 0.5]."""


    h, w = corr.shape
    if py < 1 or px < 1 or py >= h - 1 or px >= w - 1:
        return 0.0, 0.0          

    c = float(corr[py, px])
    l = float(corr[py, px - 1]); r = float(corr[py, px + 1])
    u = float(corr[py - 1, px]); d = float(corr[py + 1, px])

    scale = max(abs(c), 1e-9)
    denom_x = 2.0 * (2.0 * c - l - r)
    denom_y = 2.0 * (2.0 * c - u - d)
    dx = (r - l) / denom_x if abs(denom_x) > 1e-6 * scale else 0.0
    dy = (d - u) / denom_y if abs(denom_y) > 1e-6 * scale else 0.0
    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))

def top_k_peaks(corr, valid_h, valid_w, k=5, suppress=12):
    """K distinct correlation peaks with local non-maximum suppression."""
    work = corr[:valid_h, :valid_w].clone()
    peaks = []
    for _ in range(k):
        idx = int(torch.argmax(work))
        py, px = idx // valid_w, idx % valid_w
        peaks.append((float(work[py, px]), int(py), int(px)))
        y0, y1 = max(0, py - suppress), min(valid_h, py + suppress + 1)
        x0, x1 = max(0, px - suppress), min(valid_w, px + suppress + 1)
        work[y0:y1, x0:x1] = float("-inf")
    return peaks

def top_k_peaks_fast(corr, vh, vw, k=5, min_sep=12):
    """One topk() call plus CPU-side separation filtering. No GPU loop."""
    region = corr[:vh, :vw].reshape(-1)
    vals, idxs = torch.topk(region, 400)
    vals = vals.tolist()
    idxs = idxs.tolist()

    peaks = []
    for v, i in zip(vals, idxs):
        py, px = i // vw, i % vw
        if all(abs(py - p[1]) >= min_sep or abs(px - p[2]) >= min_sep
               for p in peaks):
            peaks.append((v, py, px))
            if len(peaks) == k:
                break
    return peaks


def peak_to_sidelobe(corr, valid_h, valid_w, py, px, exclude=12):
    """PSR with a single device->host transfer."""
    region = corr[:valid_h, :valid_w]
    y0, y1 = max(0, py - exclude), min(valid_h, py + exclude + 1)
    x0, x1 = max(0, px - exclude), min(valid_w, px + exclude + 1)

    mask = torch.ones_like(region, dtype=torch.bool)
    mask[y0:y1, x0:x1] = False
    side = region[mask]

    stats = torch.stack([region[py, px], side.mean(), side.std()])
    peak, mu, sd = stats.tolist()          # ONE sync
    return (peak - mu) / (sd + 1e-9)

def locate(search, ref, factor=10, psf_sigma=5.0):
    tmpl = degrade_reference(ref, factor, psf_sigma)
    tmpl = remove_periodic(tmpl)
    search = remove_periodic(search.astype(np.float32))
    corr = zncc_map_normalized(search, tmpl)

    th, tw = tmpl.shape
    valid = corr[:search.shape[0] - th + 1, :search.shape[1] - tw + 1]
    idx = torch.argmax(valid)
    py = int(idx // valid.shape[1])
    px = int(idx % valid.shape[1])

    dx, dy = subpixel_peak(corr, py, px)
    return px + dx + (tw - 1) / 2.0, py + dy + (th - 1) / 2.0

def locate_full(search, ref, factor=10, psf_sigma=5.0, k=5, margin=0.01):
    """Locate with candidate generation, PSR confidence, and centre tie-break.
    Returns dict: x, y, psr, ambiguous, n_candidates."""
    # 1. same front-end as locate()
    tmpl = degrade_reference(ref, factor, psf_sigma)
    tmpl = remove_periodic(tmpl)
    search = remove_periodic(search.astype(np.float32))
    corr = zncc_map_normalized(search, tmpl)

    # 2. candidate generation
    th, tw = tmpl.shape
    valid_h = search.shape[0] - th + 1
    valid_w = search.shape[1] - tw + 1
    peaks = top_k_peaks(corr, valid_h, valid_w, k)

    # 3. confidence of the leading candidate
    best_score, best_py, best_px = peaks[0]
    psr = peak_to_sidelobe(corr, valid_h, valid_w, best_py, best_px)

    # 4. is the runner-up statistically indistinguishable?
    ambiguous = peaks[1][0] >= best_score * (1.0 - margin)
    low_confidence = psr < 10.0

    # 5. spec rule: among near-tied candidates, take the one closest to centre
    if ambiguous:
        cy, cx = valid_h / 2.0, valid_w / 2.0
        tied = [p for p in peaks if p[0] >= best_score * (1.0 - margin)]
        if tied:
            _, best_py, best_px = min(
            tied, key=lambda p: (p[1] - cy) ** 2 + (p[2] - cx) ** 2
        )

    # 6. subpixel refinement on the chosen candidate
    dx, dy = subpixel_peak(corr, best_py, best_px)
    return {
        "x": best_px + dx + (tw - 1) / 2.0,
        "y": best_py + dy + (th - 1) / 2.0,
        "psr": psr,
        "ambiguous": bool(ambiguous),
        "n_candidates": len(peaks),
        "low_confidence": bool(low_confidence)
    }

# ---------------------------------------------------------------------
# Hot-path API: template prepared once, reused across many stage moves
# ---------------------------------------------------------------------

def prepare_template(ref, factor=10, psf_sigma=5.0,period=6):
    """One-time preprocessing. Returns everything the hot path needs,
    including the two FFTs that are constant across calls."""
    tmpl = degrade_reference(ref, factor, psf_sigma)
    tmpl = remove_periodic(tmpl,period)
    t = torch.from_numpy(np.ascontiguousarray(tmpl)).to(DEV)

    th, tw = t.shape
    H, W = 1152, 1152

    t = t - t.mean()
    t = t / (t.norm() + 1e-8)
    ones = torch.ones((th, tw), device=DEV, dtype=torch.float32)

    return {
        "shape": (th, tw),
        "n": float(th * tw),
        "T": torch.fft.rfft2(t, s=(H, W)),
        "O": torch.fft.rfft2(ones, s=(H, W)),
    }


def remove_periodic_gpu(t, period=6):
    """Lattice suppression, GPU version. `t` is a torch tensor on DEV."""
    th, tw = t.shape
    p = int(period)
    ny, nx = th // p, tw // p
    core = t[:ny * p, :nx * p]
    cell = core.reshape(ny, p, nx, p).mean(dim=(0, 2))
    lattice = cell.repeat(ny + 1, nx + 1)[:th, :tw]
    return t - lattice


def zncc_gpu(s, prep):
    """Correlation using precomputed template/box transforms."""
    th, tw = prep["shape"]
    n = prep["n"]
    H, W = 1152, 1152

    S = torch.fft.rfft2(s, s=(H, W))
    S2 = torch.fft.rfft2(s * s, s=(H, W))

    num = torch.fft.irfft2(S * torch.conj(prep["T"]), s=(H, W))
    local_sum = torch.fft.irfft2(S * torch.conj(prep["O"]), s=(H, W))
    local_sum2 = torch.fft.irfft2(S2 * torch.conj(prep["O"]), s=(H, W))

    local_var = local_sum2 - (local_sum * local_sum) / n
    sh, sw = s.shape
    vh, vw = sh - th + 1, sw - tw + 1

    floor = 1e-3 * local_var[:vh, :vw].max()
    out = num / torch.sqrt(torch.clamp(local_var, min=floor))

    mask = torch.full_like(out, float("-inf"))
    mask[:vh, :vw] = 0.0
    return out + mask, vh, vw


def locate_prepared(search_u8, prep, k=5, margin=0.01,period=6):
    """Hot path: search image in, coordinates out."""
    s = torch.from_numpy(np.ascontiguousarray(search_u8)).to(DEV).float()
    s = remove_periodic_gpu(s, period)

    corr, vh, vw = zncc_gpu(s, prep)
    peaks = top_k_peaks_fast(corr, vh, vw, k)

    best_score, best_py, best_px = peaks[0]
    psr = peak_to_sidelobe(corr, vh, vw, best_py, best_px)
    ambiguous = peaks[1][0] >= best_score * (1.0 - margin) 
    low_confidence = psr < 10.0

    if ambiguous:
        cy, cx = vh / 2.0, vw / 2.0
        tied = [p for p in peaks if p[0] >= best_score * (1.0 - margin)]
        if tied :
             _, best_py, best_px = min(tied, key=lambda p: (p[1]-cy)**2 + (p[2]-cx)**2)

    th, tw = prep["shape"]
    dx, dy = subpixel_peak(corr, best_py, best_px)
    return {"x": best_px + dx + (tw - 1) / 2.0,
            "y": best_py + dy + (th - 1) / 2.0,
            "psr": psr, "ambiguous": bool(ambiguous),
            "low_confidence": bool(low_confidence)}


