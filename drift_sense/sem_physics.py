import numpy as np
from scipy.ndimage import gaussian_filter,zoom


def edge_brightening(img, strength=0.6):
    """SE yield rises at sidewalls: escape depth is fixed, tilted surface
    exposes more of it. Reimer (1998), Goldstein et al. (2018)."""
    gy, gx = np.gradient(img)
    return img + strength * np.hypot(gx, gy)


def charging(img, rng, amplitude=0.08):
    """Insulator charging -> slow low-frequency brightness gradient.
    Reimer (1998), ch. on specimen charging."""
    h, w = img.shape
    coarse = rng.normal(0.0, 1.0, (8, 8))          # tiny random field
    coarse = gaussian_filter(coarse, 1.0)           # smooth it further
    field = zoom(coarse, (h / 8.0, w / 8.0), order=3)  # bicubic stretch
    field = field[:h, :w]                           # zoom can overshoot by 1px
    field = field / (np.abs(field).max() + 1e-9)    # normalise to +/-1
    return img + amplitude * field

def scan_jitter(img, rng, max_shift_px=0.7):
    """Raster scan: rows captured sequentially, so stage drift appears as
    correlated per-row horizontal offset."""
    h, w = img.shape
    walk = np.cumsum(rng.normal(0.0, 0.08, h))      # random walk = drift
    walk = walk - walk.mean()                        # no net translation
    peak = np.abs(walk).max() + 1e-9
    walk = walk / peak * max_shift_px                # scale to budget
    out = np.empty_like(img)
    for i in range(h):
        out[i] = np.roll(img[i], int(round(walk[i])))
    return out


def apply_sem_imaging(clean, rng, dose=300.0, psf_sigma=1.2,
                      read_noise=0.02, do_jitter=True):
    """Full forward model. `clean` is float32 in [0,1]. Returns uint8.

    dose = electrons per pixel. HIGH for the reference (slow scan),
    LOW for the wide search image (fast scan over 100x the area).
    """
    img = gaussian_filter(clean.astype(np.float32), psf_sigma)
    img = edge_brightening(img)
    img = charging(img, rng)
    img = np.clip(img, 0.0, None)

    # Poisson shot noise on electron counts - Reimer (1998)
    img = rng.poisson(img * dose) / dose

    # Detector/amplifier readout noise
    img = img + rng.normal(0.0, read_noise, img.shape)

    if do_jitter:
        img = scan_jitter(img, rng)

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)