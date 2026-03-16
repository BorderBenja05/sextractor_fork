"""
jax_back.py
-----------
JAX-accelerated background estimation, replacing the computational core of
back.c (backstat, backhisto, backguess, backline generation).

The original C code processes astronomical images in streaming mesh blocks,
computing per-mesh mean and sigma statistics then refining them through a
histogram.  Here we expose the same algorithm operating on entire 2-D JAX
arrays, making it GPU/JIT-friendly.

Public API
----------
backstat(image, weight, mesh_w, mesh_h, wthresh)
    Compute mean and sigma for each background mesh (replaces backstat()).

backhisto(image, backmeshes, mesh_w, mesh_h, wthresh)
    Fill per-mesh histograms (replaces backhisto()).

backguess(histo, qzero, qscale, nlevels)
    Estimate robust background mean+sigma from a histogram (backguess()).

make_backmap(image, weight, mesh_w, mesh_h, wthresh, pearson)
    Full pipeline: compute background map and sigma map for the whole image.

interp_backmap(backmap, sigma_map, image_shape)
    Bicubic-spline interpolate background map to full image resolution.
"""

import jax
import jax.numpy as jnp
from functools import partial
import math

BIG = 1e29
BACK_MINGOODFRAC = 0.5      # minimum fraction of valid pixels per mesh
QUANTIF_NSIGMA = 5          # half-range for histogram in sigma units
QUANTIF_NMAXLEVELS = 4096   # maximum number of histogram bins
QUANTIF_AMIN = 4.0          # minimum number of pixels per bin

# ---------------------------------------------------------------------------
# Mesh decomposition helpers
# ---------------------------------------------------------------------------

def _mesh_slices(image_h, image_w, mesh_h, mesh_w):
    """
    Return list of (row_start, row_end, col_start, col_end) for each mesh.
    """
    slices = []
    for row in range(0, image_h, mesh_h):
        for col in range(0, image_w, mesh_w):
            slices.append((row, min(row + mesh_h, image_h),
                           col, min(col + mesh_w, image_w)))
    return slices


# ---------------------------------------------------------------------------
# backstat  –  first-pass statistics for each mesh
# ---------------------------------------------------------------------------

def backstat(image: jnp.ndarray,
             weight: jnp.ndarray | None,
             mesh_w: int,
             mesh_h: int,
             wthresh: float = 0.0):
    """
    Compute robust mean and sigma for each background mesh.

    Replicates backstat() from back.c:
      1. Compute mean/sigma with sigma-clipping (±2σ cuts).
      2. Returns per-mesh dicts with 'mean', 'sigma', 'qzero', 'qscale',
         'nlevels', 'ny', 'nx', 'grid' shape info.

    Parameters
    ----------
    image   : jnp.ndarray, shape (H, W), float32
    weight  : jnp.ndarray or None, shape (H, W)
    mesh_w  : int – mesh width in pixels
    mesh_h  : int – mesh height in pixels
    wthresh : float – weight threshold for bad pixels

    Returns
    -------
    meshes : list of dicts, length nx*ny
        Each dict has keys: 'mean', 'sigma', 'lcut', 'hcut', 'nlevels',
        'qscale', 'qzero', 'npix', 'row0', 'col0', 'row1', 'col1'
    grid_shape : (ny, nx) – number of meshes in each direction
    """
    H, W = image.shape
    step = math.sqrt(2.0 / math.pi) * QUANTIF_NSIGMA / QUANTIF_AMIN

    meshes = []
    rows = list(range(0, H, mesh_h))
    cols = list(range(0, W, mesh_w))

    for r0 in rows:
        for c0 in cols:
            r1 = min(r0 + mesh_h, H)
            c1 = min(c0 + mesh_w, W)
            patch = image[r0:r1, c0:c1].astype(jnp.float32)

            if weight is not None:
                wpatch = weight[r0:r1, c0:c1].astype(jnp.float32)
                good = (wpatch < wthresh) & (patch > -BIG)
            else:
                wpatch = None
                good = patch > -BIG

            npix = int(jnp.sum(good.astype(jnp.int32)))
            min_pix = int((r1 - r0) * (c1 - c0) * BACK_MINGOODFRAC)

            if npix < min_pix:
                meshes.append({'mean': -BIG, 'sigma': -BIG,
                               'lcut': 0.0, 'hcut': 0.0, 'nlevels': 0,
                               'qscale': 1.0, 'qzero': 0.0, 'npix': 0,
                               'row0': r0, 'col0': c0, 'row1': r1, 'col1': c1})
                continue

            vals = jnp.where(good, patch, 0.0)
            mean = float(jnp.sum(vals)) / npix
            sigma2 = float(jnp.sum(jnp.where(good, vals * vals, 0.0))) / npix - mean * mean
            sigma = math.sqrt(max(sigma2, 0.0))

            lcut = mean - 2.0 * sigma
            hcut = mean + 2.0 * sigma

            # Second pass: clip at ±2σ
            good2 = good & (patch >= lcut) & (patch <= hcut)
            npix2 = int(jnp.sum(good2.astype(jnp.int32)))

            if npix2 < 1:
                npix2 = 1
            vals2 = jnp.where(good2, patch, 0.0)
            mean2 = float(jnp.sum(vals2)) / npix2
            sigma2b = float(jnp.sum(jnp.where(good2, vals2 * vals2, 0.0))) / npix2 - mean2 * mean2
            sigma2_val = math.sqrt(max(sigma2b, 0.0))

            nlevels = min(int(step * npix2 + 1), QUANTIF_NMAXLEVELS)
            qscale = (2.0 * QUANTIF_NSIGMA * sigma2_val / nlevels) if sigma2_val > 0.0 else 1.0
            qzero = mean2 - QUANTIF_NSIGMA * sigma2_val

            meshes.append({
                'mean': mean2, 'sigma': sigma2_val,
                'lcut': lcut, 'hcut': hcut,
                'nlevels': nlevels, 'qscale': qscale, 'qzero': qzero,
                'npix': npix2,
                'row0': r0, 'col0': c0, 'row1': r1, 'col1': c1,
            })

    ny = len(rows)
    nx = len(cols)
    return meshes, (ny, nx)


# ---------------------------------------------------------------------------
# backhisto  –  fill per-mesh histograms
# ---------------------------------------------------------------------------

def backhisto(image: jnp.ndarray,
              meshes: list,
              weight: jnp.ndarray | None = None,
              wthresh: float = 0.0):
    """
    Fill integer histograms for each mesh (replaces backhisto() in back.c).

    Parameters
    ----------
    image   : jnp.ndarray, shape (H, W)
    meshes  : list of dicts from :func:`backstat`
    weight  : optional weight map
    wthresh : bad-pixel threshold

    Returns
    -------
    meshes : same list, each dict now has a 'histo' key (jnp.ndarray int32)
    """
    for m in meshes:
        if m['mean'] <= -BIG or m['nlevels'] == 0:
            m['histo'] = None
            continue

        r0, r1, c0, c1 = m['row0'], m['row1'], m['col0'], m['col1']
        patch = image[r0:r1, c0:c1].astype(jnp.float32)

        if weight is not None:
            wpatch = weight[r0:r1, c0:c1].astype(jnp.float32)
            good = (wpatch < wthresh) & (patch > -BIG)
        else:
            good = patch > -BIG

        qscale = m['qscale']
        cste = 0.499999 - m['qzero'] / qscale
        bins = (patch / qscale + cste).astype(jnp.int32)
        valid = good & (bins >= 0) & (bins < m['nlevels'])

        histo = jnp.zeros(m['nlevels'], dtype=jnp.int32)
        # Scatter-add (histogram)
        histo = histo.at[jnp.where(valid, bins, 0)].add(
            valid.astype(jnp.int32).ravel()
        )
        m['histo'] = histo

    return meshes


# ---------------------------------------------------------------------------
# backguess  –  robust background estimate from histogram
# ---------------------------------------------------------------------------

def backguess(bkg: dict) -> tuple:
    """
    Estimate robust background mean and sigma from a filled mesh dict.

    Replicates backguess() from back.c.  Uses iterative Gaussian fitting
    on the histogram until convergence.

    Parameters
    ----------
    bkg : dict from backhisto (must have 'histo', 'nlevels', 'qscale', 'qzero')

    Returns
    -------
    (mean, sigma) : floats.  Returns (-BIG, -BIG) for bad meshes.
    """
    if bkg['mean'] <= -BIG or bkg['histo'] is None:
        return -BIG, -BIG

    histo = jnp.array(bkg['histo'], dtype=jnp.float32)
    nlevels = bkg['nlevels']
    qscale = bkg['qscale']
    qzero = bkg['qzero']
    EPS = 1e-4

    lcut = 0
    hcut = nlevels - 1
    sig = 10.0 * (nlevels - 1)
    sig1 = 1.0

    # Iterative sigma-clipping on the histogram
    for _ in range(50):
        if abs(sig - sig1) <= EPS * sig1:
            break
        sig1 = sig

        # Moments from histogram
        idx = jnp.arange(nlevels, dtype=jnp.float32)
        mask = (idx >= lcut) & (idx <= hcut)
        h = jnp.where(mask, histo, 0.0)

        total = float(jnp.sum(h))
        if total <= 0:
            break

        mean_bin = float(jnp.sum(h * idx)) / total
        var_bin = float(jnp.sum(h * idx * idx)) / total - mean_bin * mean_bin
        sig = math.sqrt(max(var_bin, 0.0))

        lcut = max(0, int(mean_bin - 2.0 * sig))
        hcut = min(nlevels - 1, int(mean_bin + 2.0 * sig))

    # Convert bin coordinates back to pixel values
    mean = qzero + (mean_bin + 0.5) * qscale
    sigma = sig * qscale

    return mean, sigma


# ---------------------------------------------------------------------------
# make_backmap  –  full background map pipeline
# ---------------------------------------------------------------------------

def make_backmap(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mesh_w: int,
    mesh_h: int,
    wthresh: float = 0.0,
    pearson: float = 0.3,
) -> tuple:
    """
    Compute the background map and sigma map for the whole image.

    Equivalent to makeback() + backline rendering in back.c.

    Parameters
    ----------
    image    : jnp.ndarray, shape (H, W), float32
    weight   : optional weight map
    mesh_w   : mesh width (pixels)
    mesh_h   : mesh height (pixels)
    wthresh  : bad-pixel weight threshold
    pearson  : Pearson mode fraction (prefs.back_pearson)

    Returns
    -------
    back_map  : jnp.ndarray, shape (ny, nx), float32  – background per mesh
    sigma_map : jnp.ndarray, shape (ny, nx), float32  – sigma per mesh
    grid_shape : (ny, nx)
    """
    meshes, (ny, nx) = backstat(image, weight, mesh_w, mesh_h, wthresh)
    meshes = backhisto(image, meshes, weight, wthresh)

    back_arr = []
    sigma_arr = []
    for m in meshes:
        mean, sigma = backguess(m)
        back_arr.append(mean)
        sigma_arr.append(sigma)

    back_map = jnp.array(back_arr, dtype=jnp.float32).reshape(ny, nx)
    sigma_map = jnp.array(sigma_arr, dtype=jnp.float32).reshape(ny, nx)

    return back_map, sigma_map, (ny, nx)


# ---------------------------------------------------------------------------
# interp_backmap  –  interpolate background to full image resolution
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("image_h", "image_w"))
def interp_backmap(back_map: jnp.ndarray,
                   image_h: int,
                   image_w: int) -> jnp.ndarray:
    """
    Bilinear interpolation of the coarse background map to full image size.

    Replaces the ``backline()`` calls that compute per-row background
    estimates in the original C code.

    Parameters
    ----------
    back_map : jnp.ndarray, shape (ny, nx), float32
    image_h  : int – full image height
    image_w  : int – full image width

    Returns
    -------
    back_full : jnp.ndarray, shape (image_h, image_w), float32
    """
    # jax.image.resize uses bilinear by default for float arrays
    return jax.image.resize(back_map, shape=(image_h, image_w), method='linear')


# ---------------------------------------------------------------------------
# Vectorised background subtraction
# ---------------------------------------------------------------------------

@jax.jit
def subtract_background(image: jnp.ndarray,
                        back_full: jnp.ndarray) -> jnp.ndarray:
    """
    Subtract interpolated background from image.

    Parameters
    ----------
    image     : jnp.ndarray, shape (H, W), float32
    back_full : jnp.ndarray, shape (H, W), float32  – from interp_backmap

    Returns
    -------
    corrected : jnp.ndarray, shape (H, W), float32
    """
    return image.astype(jnp.float32) - back_full


# ---------------------------------------------------------------------------
# Batch processing: backmap for multiple images with vmap
# ---------------------------------------------------------------------------

def make_backmap_batch(images: jnp.ndarray,
                       mesh_w: int,
                       mesh_h: int,
                       wthresh: float = 0.0) -> tuple:
    """
    Compute background maps for a batch of images.

    Parameters
    ----------
    images  : jnp.ndarray, shape (N, H, W), float32
    mesh_w  : int
    mesh_h  : int
    wthresh : float

    Returns
    -------
    back_maps  : list of N back_map arrays
    sigma_maps : list of N sigma_map arrays
    """
    back_maps = []
    sigma_maps = []
    for img in images:
        bm, sm, _ = make_backmap(img, None, mesh_w, mesh_h, wthresh)
        back_maps.append(bm)
        sigma_maps.append(sm)
    return back_maps, sigma_maps
