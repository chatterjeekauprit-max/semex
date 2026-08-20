import numpy as np
from scipy.ndimage import gaussian_filter
from .sem_physics import apply_sem_imaging

CANVAS = 10000   # 1 nm/px  -> 10 µm field of view
REF_PX = 1000    # reference is 1000x1000 @ 1 nm/px
ZOOM = 10        # search is 10 nm/px


def build_canvas(size=CANVAS, pattern="DRAM", pitch_nm=60, seed=0):
    """Continuous die layout at 1 nm/px. float32 in [0,1]. No noise."""
    y, x = np.ogrid[:size, :size]
    canvas = np.zeros((size, size), np.float32)

    if pattern.upper() == "DRAM":
        canvas = np.where((y % pitch_nm) < 20, 0.75, canvas)
        canvas = np.maximum(canvas, np.where((x % pitch_nm) < 16, 0.55, 0.0))
        dx = (x % pitch_nm) - pitch_nm / 2
        dy = (y % pitch_nm) - pitch_nm / 2
        via = np.exp(-(dx ** 2 + dy ** 2) / (2 * 9.0 ** 2))
        canvas = np.maximum(canvas, via)

    elif pattern.upper() == "FINFET":
        fin_pitch = pitch_nm // 2          # 30 nm - fins are dense
        gate_pitch = pitch_nm * 4          # 240 nm - gates are sparse

        # A. Vertical fins: narrow silicon ridges, ~12 nm wide
        canvas = np.where((x % fin_pitch) < 12, 0.70, canvas)

        # B. Horizontal gate bars: ~45 nm wide, drawn OVER the fins.
        #    Gates sit above fins in the stack, so they occlude rather
        #    than brighten - np.where, not np.maximum.
        canvas = np.where((y % gate_pitch) < 45, 0.85, canvas)
        
    return np.clip(canvas, 0, 1).astype(np.float32)


def place_unique_feature(canvas, cx, cy, rng):
    """Asymmetric L-shaped alignment mark, similar brightness to the metal
    lines so it can't be found by thresholding alone. In-place."""
    v = 0.82
    canvas[cy - 120:cy + 120, cx - 120:cx - 60] = v     # vertical arm
    canvas[cy + 60:cy + 120, cx - 120:cx + 120] = v     # horizontal arm


def decimate(arr, factor=ZOOM, psf_sigma_px=5.0):
    """PSF blur, THEN exact area-average. Never cv2.resize here."""
    if psf_sigma_px:
        arr = gaussian_filter(arr, psf_sigma_px)
    h, w = arr.shape
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def generate_dataset_pair(pattern="DRAM",zoom=ZOOM, seed=42, noise=True, unique_marker=True):
    """One canvas, sampled twice: native crop -> reference, blur+decimate -> search."""
    rng = np.random.default_rng(seed)
    canvas = build_canvas(pattern=pattern, seed=seed)

    margin = REF_PX // 2 + 100
    cx = int(rng.integers(margin, CANVAS - margin))
    cy = int(rng.integers(margin, CANVAS - margin))
    if unique_marker:
        place_unique_feature(canvas, cx, cy, rng)

    left = cx - REF_PX // 2
    top = cy - REF_PX // 2
    reference = canvas[top:top + REF_PX, left:left + REF_PX].copy()
    search = decimate(canvas)

    truth = ((left + 495) / 10.0, (top + 495) / 10.0)

    if noise:
        rng_ref = np.random.default_rng(seed * 7919 + 1)
        rng_srch = np.random.default_rng(seed * 7919 + 2)
        reference = apply_sem_imaging(reference, rng_ref, dose=800.0, psf_sigma=1.2)
        search = apply_sem_imaging(search, rng_srch, dose=60.0, psf_sigma=0.8,do_jitter=True)
    else:
        reference = (np.clip(reference, 0, 1) * 255).astype(np.uint8)
        search = (np.clip(search, 0, 1) * 255).astype(np.uint8)

    return search, reference, truth