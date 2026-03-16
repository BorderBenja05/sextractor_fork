"""
jax_photom.py
-------------
JAX-accelerated photometry routines, replacing photom.c.

Implements the three main photometric aperture measurements:

  computeaperflux   – circular aperture flux (with oversampling at edge)
  computeautoflux   – Kron auto-aperture flux
  computepetroflux  – Petrosian aperture flux

All functions operate on 2-D NumPy/JAX arrays representing image strips,
fully vectorised with ``jnp`` operations.  No Python-level pixel loops.

Key differences from the C implementation
------------------------------------------
* We operate on complete 2-D patches rather than streaming strips.
* The oversampled annulus in ``computeaperflux`` is implemented via a
  subpixel grid broadcast rather than nested loops.
* Weight map handling uses JAX conditional masking.
* ``jax.jit`` is applied where the computation graph is static.
"""

import jax
import jax.numpy as jnp
from functools import partial

# Default oversampling factor (must match APER_OVERSAMP in photom.h = 5)
APER_OVERSAMP = 5

BIG = 1e29   # sentinel for bad pixels (matches define.h)


# ---------------------------------------------------------------------------
# Helper: sub-pixel area within a circular aperture (vectorised)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("oversamp",))
def _circular_area(dx: jnp.ndarray, dy: jnp.ndarray,
                   raper: float, oversamp: int = APER_OVERSAMP) -> jnp.ndarray:
    """
    Compute the fractional area of each pixel lying within a circular aperture
    of radius *raper*, using sub-pixel oversampling.

    Replicates the double-loop:
        for sy in range(oversamp):
            for sx in range(oversamp):
                if dx1**2 + dy2 < raper**2: locarea += scale2

    Parameters
    ----------
    dx, dy : jnp.ndarray, any broadcastable shape
        Distance of pixel centre from aperture centre.
    raper  : float – aperture radius in pixels.
    oversamp : int – sub-pixel oversampling factor.

    Returns
    -------
    locarea : jnp.ndarray, same shape as dx / dy, float32
        Fractional area in [0, 1].
    """
    scalex = scaley = 1.0 / oversamp
    scale2 = scalex * scaley
    offsetx = 0.5 * (scalex - 1.0)
    offsety = 0.5 * (scaley - 1.0)
    raper2 = raper * raper

    # Sub-pixel offsets: shape (oversamp,)
    sub = jnp.arange(oversamp, dtype=jnp.float32)
    sub_ox = offsetx + sub * scalex   # (oversamp,)
    sub_oy = offsety + sub * scaley   # (oversamp,)

    # Broadcast: dx has shape (...), sub has shape (oversamp,)
    # Result shape: (..., oversamp, oversamp)
    dx_sub = dx[..., None, None] + sub_ox[None, :]         # (..., 1, oversamp)
    dy_sub = dy[..., None, None] + sub_oy[:, None]         # (..., oversamp, 1)

    inside = (dx_sub * dx_sub + dy_sub * dy_sub) < raper2  # (..., oversamp, oversamp)
    locarea = jnp.sum(inside.astype(jnp.float32), axis=(-2, -1)) * scale2
    return locarea


# ---------------------------------------------------------------------------
# compute_aperflux  –  circular aperture photometry
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("oversamp",))
def compute_aperflux(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float, my: float,
    raper: float,
    backsig: float,
    gain: float,
    bkg: float,
    ngamma: float = 0.0,
    wthresh: float = 0.0,
    weightgain_flag: bool = False,
    photo_flag: bool = False,
    oversamp: int = APER_OVERSAMP,
) -> tuple:
    """
    Compute circular aperture flux (computeaperflux in photom.c).

    Parameters
    ----------
    image   : jnp.ndarray, shape (H, W), float32  – image strip
    weight  : jnp.ndarray or None, shape (H, W)   – weight map (inverse variance)
    mx, my  : float  – object centroid (pixels, 0-based)
    raper   : float  – aperture radius in pixels
    backsig : float  – background noise sigma
    gain    : float  – detector gain (e-/ADU), 0 = unknown
    bkg     : float  – local background level per pixel
    ngamma  : float  – log-scaling gamma (>0 for PHOTO detection type)
    wthresh : float  – weight map bad-pixel threshold
    weightgain_flag : bool
    photo_flag      : bool  – use log-scaling
    oversamp : int  – sub-pixel oversampling factor

    Returns
    -------
    (flux, fluxerr, area) : tuple of float
    """
    H, W = image.shape
    backnoise2 = backsig * backsig

    # Pixel coordinate grids
    iy = jnp.arange(H, dtype=jnp.float32)
    ix = jnp.arange(W, dtype=jnp.float32)
    ix_grid, iy_grid = jnp.meshgrid(ix, iy)   # (H, W) each

    dx = ix_grid - mx
    dy = iy_grid - my
    r2 = dx * dx + dy * dy

    rintlim = raper - 0.75
    rintlim2 = rintlim * rintlim if rintlim > 0.0 else 0.0
    rextlim2 = (raper + 0.75) * (raper + 0.75)

    # Fully inside the aperture
    locarea_full = jnp.ones((H, W), dtype=jnp.float32)
    # Oversampled annulus
    locarea_over = _circular_area(dx, dy, raper, oversamp)

    locarea = jnp.where(r2 < rintlim2, locarea_full,
                        jnp.where(r2 < rextlim2, locarea_over, 0.0))

    # Pixel validity
    pix = image.astype(jnp.float32)
    good_pix = (pix > -BIG)
    if weight is not None:
        w = weight.astype(jnp.float32)
        good_pix = good_pix & (w < wthresh)
        var = jnp.where(good_pix, w, backnoise2)
    else:
        w = jnp.full_like(pix, backnoise2)
        var = w

    # Bad pixel correction: replace with 0
    pix = jnp.where(good_pix, pix, 0.0)

    # Photo (log) mode
    if photo_flag and ngamma > 0.0:
        pix_eff = jnp.exp(pix / ngamma)
        sigtv_pix = var * locarea * pix_eff * pix_eff
    else:
        pix_eff = pix
        sigtv_pix = var * locarea

    in_aper = locarea > 0.0

    tv = jnp.sum(jnp.where(in_aper, locarea * pix_eff, 0.0))
    sigtv = jnp.sum(jnp.where(in_aper, sigtv_pix, 0.0))
    area = jnp.sum(jnp.where(in_aper, locarea, 0.0))

    # Gain correction
    if gain > 0.0 and not weightgain_flag:
        if photo_flag:
            pass  # handled above
        else:
            tv_pos = jnp.maximum(tv, 0.0)
            sigtv = sigtv + tv_pos / gain

    # Background subtraction
    if photo_flag and ngamma > 0.0:
        tv = ngamma * (tv - area * jnp.exp(bkg / ngamma))
        sigtv = sigtv / (ngamma * ngamma)
    else:
        tv = tv - area * bkg

    fluxerr = jnp.sqrt(jnp.maximum(sigtv, 0.0))
    return float(tv), float(fluxerr), float(area)


# ---------------------------------------------------------------------------
# Kron radius helper
# ---------------------------------------------------------------------------

@jax.jit
def compute_kron_radius(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float, my: float,
    cx2: float, cy2: float, cxy: float,
    klim2: float,
    wthresh: float = 0.0,
) -> float:
    """
    Compute the first moment radius r1/v1 used for the Kron factor.

    Replaces the first loop in ``computeautoflux()``.
    """
    H, W = image.shape
    iy = jnp.arange(H, dtype=jnp.float32)
    ix = jnp.arange(W, dtype=jnp.float32)
    ix_grid, iy_grid = jnp.meshgrid(ix, iy)

    dx = ix_grid - mx
    dy = iy_grid - my
    r2 = cx2 * dx * dx + cy2 * dy * dy + cxy * dx * dy

    pix = image.astype(jnp.float32)
    good = (pix > -BIG) & (r2 <= klim2)
    if weight is not None:
        w = weight.astype(jnp.float32)
        good = good & (w < wthresh)

    pix = jnp.where(good, pix, 0.0)
    r_val = jnp.sqrt(jnp.maximum(r2, 0.0))
    r1 = jnp.sum(jnp.where(good, r_val * pix, 0.0))
    v1 = jnp.sum(jnp.where(good, pix, 0.0))
    return float(r1), float(v1)


# ---------------------------------------------------------------------------
# Elliptical aperture integration (shared by auto and petro)
# ---------------------------------------------------------------------------

@jax.jit
def _elliptical_flux(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float, my: float,
    cx2: float, cy2: float, cxy: float,
    klim2: float,
    bkg: float,
    backsig: float,
    gain: float,
    ngamma: float,
    wthresh: float,
    weightgain_flag: bool,
    photo_flag: bool,
) -> tuple:
    """
    Integrate flux within an elliptical aperture defined by the quadratic form
    ``cx2*dx² + cy2*dy² + cxy*dx*dy ≤ klim2``.

    Returns (flux, fluxerr, area, areab)
    """
    H, W = image.shape
    backnoise2 = backsig * backsig

    iy = jnp.arange(H, dtype=jnp.float32)
    ix = jnp.arange(W, dtype=jnp.float32)
    ix_grid, iy_grid = jnp.meshgrid(ix, iy)
    dx = ix_grid - mx
    dy = iy_grid - my
    r2 = cx2 * dx * dx + cy2 * dy * dy + cxy * dx * dy

    in_aper = r2 <= klim2

    pix = image.astype(jnp.float32)
    good_pix = (pix > -BIG)
    if weight is not None:
        w = weight.astype(jnp.float32)
        good_pix_w = good_pix & (w < wthresh)
        var = jnp.where(good_pix_w, w, backnoise2)
        bad_pix = in_aper & ~good_pix_w
    else:
        var = jnp.full_like(pix, backnoise2)
        bad_pix = in_aper & ~good_pix

    pix = jnp.where(good_pix, pix, 0.0)

    area = jnp.sum(in_aper.astype(jnp.float32))
    areab = jnp.sum(bad_pix.astype(jnp.float32))

    if photo_flag and ngamma > 0.0:
        pix_eff = jnp.exp(pix / ngamma)
        sigtv_pix = var * pix_eff * pix_eff
    else:
        pix_eff = pix
        sigtv_pix = var

    tv = jnp.sum(jnp.where(in_aper, pix_eff, 0.0))
    sigtv = jnp.sum(jnp.where(in_aper, sigtv_pix, 0.0))

    if gain > 0.0 and not weightgain_flag:
        if not photo_flag:
            tv_pos = jnp.maximum(tv, 0.0)
            sigtv = sigtv + tv_pos / gain

    if photo_flag and ngamma > 0.0:
        tv = ngamma * (tv - area * jnp.exp(bkg / ngamma))
        sigtv = sigtv / (ngamma * ngamma)
    else:
        tv = tv - area * bkg

    return float(tv), float(jnp.sqrt(jnp.maximum(sigtv, 0.0))), float(area), float(areab)


# ---------------------------------------------------------------------------
# compute_autoflux  –  Kron auto-aperture photometry
# ---------------------------------------------------------------------------

def compute_autoflux(
    image: jnp.ndarray,
    det_image: jnp.ndarray,
    weight: jnp.ndarray | None,
    det_weight: jnp.ndarray | None,
    mx: float, my: float,
    cxx: float, cyy: float, cxy: float,
    a: float, b: float,
    autoparam: tuple,
    autoaper: tuple,
    backsig: float, gain: float, bkg: float,
    ngamma: float = 0.0, wthresh: float = 0.0,
    weightgain_flag: bool = False,
    photo_flag: bool = False,
    kron_nsig: float = 6.0,
):
    """
    Compute Kron auto-aperture flux (computeautoflux in photom.c).

    Parameters
    ----------
    image, det_image    : (H,W) arrays for measurement and detection
    weight, det_weight  : optional weight maps
    mx, my              : centroid coordinates
    cxx, cyy, cxy       : second-moment ellipse parameters
    a, b                : semi-major/minor axes
    autoparam           : (kron_factor_scale, min_radius)
    autoaper            : (search_radius_min, integration_radius_min)
    ...

    Returns
    -------
    flux, fluxerr, kronfactor
    """
    KRON_NSIG = kron_nsig
    import math

    # Determine search ellipse
    if KRON_NSIG * math.sqrt(a * b) > autoaper[0] / 2.0:
        dxlim_sq = cxx - cxy * cxy / (4.0 * cyy) if cyy != 0 else 0.0
        dxlim = KRON_NSIG / math.sqrt(dxlim_sq) if dxlim_sq > 0 else 0.0
        dylim_sq = cyy - cxy * cxy / (4.0 * cxx) if cxx != 0 else 0.0
        dylim = KRON_NSIG / math.sqrt(dylim_sq) if dylim_sq > 0 else 0.0
        klim2 = KRON_NSIG * KRON_NSIG
        cx2, cy2, cxy_e = cxx, cyy, cxy
    else:
        cx2 = cy2 = 1.0
        cxy_e = 0.0
        dxlim = dylim = autoaper[0] / 2.0
        klim2 = dxlim * dxlim

    # Step 1: Kron radius from detection image
    r1, v1 = compute_kron_radius(
        det_image, det_weight, mx, my, cx2, cy2, cxy_e, klim2, wthresh)

    if r1 > 0.0 and v1 > 0.0:
        kronfactor = autoparam[0] * r1 / v1
        kronfactor = max(kronfactor, autoparam[1])
    else:
        kronfactor = autoparam[1]

    # Step 2: Determine integration ellipse
    if kronfactor * math.sqrt(a * b) > autoaper[1] / 2.0:
        cx2i, cy2i, cxyi = cxx, cyy, cxy
        dxlim_sq = cx2i - cxyi * cxyi / (4.0 * cy2i) if cy2i != 0 else 0.0
        dxlim2 = kronfactor / math.sqrt(dxlim_sq) if dxlim_sq > 0 else 0.0
        dylim_sq = cy2i - cxyi * cxyi / (4.0 * cx2i) if cx2i != 0 else 0.0
        dylim2 = kronfactor / math.sqrt(dylim_sq) if dylim_sq > 0 else 0.0
        klim2_int = kronfactor * kronfactor
    else:
        cx2i = cy2i = 1.0
        cxyi = 0.0
        dxlim2 = dylim2 = autoaper[1] / 2.0
        klim2_int = dxlim2 * dxlim2
        kronfactor = 0.0

    # Step 3: Integrate
    flux, fluxerr, area, areab = _elliptical_flux(
        image, weight, mx, my,
        cx2i, cy2i, cxyi, klim2_int,
        bkg, backsig, gain, ngamma, wthresh, weightgain_flag, photo_flag)

    return flux, fluxerr, kronfactor


# ---------------------------------------------------------------------------
# compute_petroflux  –  Petrosian aperture photometry
# ---------------------------------------------------------------------------

def compute_petroflux(
    image: jnp.ndarray,
    det_image: jnp.ndarray,
    weight: jnp.ndarray | None,
    det_weight: jnp.ndarray | None,
    mx: float, my: float,
    cxx: float, cyy: float, cxy: float,
    a: float, b: float,
    petroparam: tuple,
    autoaper: tuple,
    backsig: float, gain: float, bkg: float,
    ngamma: float = 0.0, wthresh: float = 0.0,
    weightgain_flag: bool = False,
    photo_flag: bool = False,
    petro_nsig: float = 6.0,
):
    """
    Compute Petrosian aperture flux (computepetroflux in photom.c).

    The Petrosian radius is found by scanning annuli and checking when the
    mean surface brightness in the annulus drops below 0.2× the mean within.

    Returns
    -------
    flux, fluxerr, petrofactor
    """
    import math

    PETRO_NSIG = petro_nsig

    if PETRO_NSIG * math.sqrt(a * b) > autoaper[0] / 2.0:
        dxlim_sq = cxx - cxy * cxy / (4.0 * cyy) if cyy != 0 else 0.0
        dxlim = PETRO_NSIG / math.sqrt(dxlim_sq) if dxlim_sq > 0 else 0.0
        dylim_sq = cyy - cxy * cxy / (4.0 * cxx) if cxx != 0 else 0.0
        dylim = PETRO_NSIG / math.sqrt(dylim_sq) if dylim_sq > 0 else 0.0
        klim2 = PETRO_NSIG * PETRO_NSIG
        cx2, cy2, cxy_e = cxx, cyy, cxy
    else:
        cx2 = cy2 = 1.0
        cxy_e = 0.0
        dxlim = dylim = autoaper[0] / 2.0
        klim2 = dxlim * dxlim

    klim = math.sqrt(klim2)
    kstep = klim / 20.0

    # Scan annuli to find Petrosian factor
    petrofactor = petroparam[1]
    kmea = 0.0
    H, W = det_image.shape
    iy = jnp.arange(H, dtype=jnp.float32)
    ix = jnp.arange(W, dtype=jnp.float32)
    ix_grid, iy_grid = jnp.meshgrid(ix, iy)
    dx = ix_grid - mx
    dy = iy_grid - my
    r2_full = cx2 * dx * dx + cy2 * dy * dy + cxy_e * dx * dy

    pix_det = det_image.astype(jnp.float32)
    good_det = pix_det > -BIG
    if det_weight is not None:
        dw = det_weight.astype(jnp.float32)
        good_det = good_det & (dw < wthresh)

    kmin = kstep
    while True:
        kmax = kmin * 1.2
        if kmax >= klim:
            break
        kmea = (kmin + kmax) / 2.0
        kmea2 = kmea * kmea
        kmin2 = kmin * kmin
        kmax2 = kmax * kmax

        in_outer = r2_full <= kmax2
        in_annulus = in_outer & (r2_full >= kmin2)
        in_inner = r2_full < kmea2

        munum_pixels = jnp.where(in_annulus & good_det, pix_det, 0.0)
        muden_pixels = jnp.where(in_inner & good_det, pix_det, 0.0)
        areanum = jnp.sum((in_annulus & good_det).astype(jnp.float32))
        areaden = jnp.sum((in_inner & good_det).astype(jnp.float32))

        if areanum > 0 and areaden > 0:
            munum = float(jnp.sum(munum_pixels)) / float(areanum)
            muden = float(jnp.sum(muden_pixels)) / float(areaden)
            if munum < muden * 0.2:
                petrofactor = petroparam[0] * kmea
                petrofactor = max(petrofactor, petroparam[1])
                break

        kmin += kstep

    # Integrate within Petrosian ellipse
    if petrofactor * math.sqrt(a * b) > autoaper[1] / 2.0:
        cx2i, cy2i, cxyi = cxx, cyy, cxy
        dxlim_sq = cx2i - cxyi * cxyi / (4.0 * cy2i) if cy2i != 0 else 0.0
        dxlim2 = petrofactor / math.sqrt(dxlim_sq) if dxlim_sq > 0 else 0.0
        dylim_sq = cy2i - cxyi * cxyi / (4.0 * cx2i) if cx2i != 0 else 0.0
        dylim2 = petrofactor / math.sqrt(dylim_sq) if dylim_sq > 0 else 0.0
        klim2_int = petrofactor * petrofactor
    else:
        cx2i = cy2i = 1.0
        cxyi = 0.0
        dxlim2 = dylim2 = autoaper[1] / 2.0
        klim2_int = dxlim2 * dxlim2
        petrofactor = 0.0

    flux, fluxerr, area, areab = _elliptical_flux(
        image, weight, mx, my,
        cx2i, cy2i, cxyi, klim2_int,
        bkg, backsig, gain, ngamma, wthresh, weightgain_flag, photo_flag)

    return flux, fluxerr, petrofactor


# ---------------------------------------------------------------------------
# Magnitude conversion helpers
# ---------------------------------------------------------------------------

@jax.jit
def flux_to_mag(flux: jnp.ndarray, zeropoint: float = 0.0) -> jnp.ndarray:
    """Convert flux to AB magnitude: -2.5 * log10(flux) + zeropoint."""
    return jnp.where(flux > 0.0, -2.5 * jnp.log10(jnp.maximum(flux, 1e-30)) + zeropoint, 99.0)


@jax.jit
def flux_to_magerr(flux: jnp.ndarray, fluxerr: jnp.ndarray) -> jnp.ndarray:
    """Convert flux error to magnitude error: 1.086 * fluxerr / flux."""
    return jnp.where(flux > 0.0, 1.086 * fluxerr / jnp.maximum(flux, 1e-30), 99.0)
