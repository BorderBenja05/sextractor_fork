"""
jax_models.py
-------------
JAX-accelerated galaxy profile generation, replacing the computational core
of profit.c (specifically ``prof_add()`` and the profile rendering loops).

Supported model types (matching profit.h / MODEL_* constants):
  - MODEL_BACK          : flat background offset
  - MODEL_DIRAC         : point source (delta function at centre)
  - MODEL_SERSIC        : Sérsic spheroid (general n)
  - MODEL_DEVAUCOULEURS : de Vaucouleurs spheroid (n=4 fixed)
  - MODEL_EXPONENTIAL   : exponential disk (n=1 fixed)
  - MODEL_ARMS          : logarithmic spiral arms
  - MODEL_BAR           : galactic bar
  - MODEL_INRING        : inner ring
  - MODEL_OUTRING       : outer ring
  - MODEL_TABULATED     : tabulated Sérsic look-up (same as SERSIC here)

All renderers return a 2-D float32 array of shape ``(height, width)``
representing the *un-normalised* surface brightness map in model pixel space.
Flux scaling and PSF convolution are handled separately (see jax_fft.py).

Design notes
------------
* All loops from the C ``prof_add()`` function are replaced by vectorised
  ``jnp`` operations on coordinate grids computed once with
  ``jnp.mgrid`` / ``jnp.meshgrid``.
* The smooth/sharp transition trick used in the C code (polynomial near
  centre + exponential outside) is preserved for numerical fidelity.
* Functions are ``@jax.jit``-compiled; parameters are static-safe scalars or
  traced arrays where appropriate.
* ``jax.vmap`` wrappers are provided to render a batch of objects at once.
"""

import jax
import jax.numpy as jnp
from functools import partial
import math

# ---------------------------------------------------------------------------
# Constants (matching profit.h)
# ---------------------------------------------------------------------------

DEG = math.pi / 180.0          # degrees → radians
PROFIT_SMOOTHR = 0.5           # smooth/sharp transition radius (in profile units)
PROFIT_MAXR2MAX = 1e10         # hard ceiling on r^2_max
PROFIT_BARXFADE = 0.1          # bar cross-fade width fraction

# ---------------------------------------------------------------------------
# Helper: build the 2-D pixel coordinate grids
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def _make_coords(width: int, height: int):
    """
    Return (x1_grid, x2_grid) centred at image centre, matching the C
    convention where x10 = -x1cout, x2 = -x2cout.

    Shape of each output: (height, width), float32.
    """
    x1cout = width / 2.0
    x2cout = height / 2.0
    # ix1 runs 0..width-1 → x1 runs -x1cout .. (width-1-x1cout)
    ix1 = jnp.arange(width, dtype=jnp.float32)
    ix2 = jnp.arange(height, dtype=jnp.float32)
    x1 = ix1 - x1cout          # shape (W,)
    x2 = ix2 - x2cout          # shape (H,)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)  # both (H, W)
    return x1_grid, x2_grid


# ---------------------------------------------------------------------------
# CD-matrix (coordinate transform) helpers
# ---------------------------------------------------------------------------

def _cd_matrix(scale: float, aspect: float, posangle_deg: float,
               pixstep: float, typscale: float = 1.0):
    """
    Compute the 2×2 coordinate-transform matrix used to map model pixel
    coordinates (x1, x2) to the profile's internal (u, v) coordinates.

    Matches the C code:
        ctheta = cos(posangle*DEG); stheta = sin(posangle*DEG)
        xscale = scaling / scale          (scaling = pixstep/typscale)
        yscale = scaling / (scale*aspect)
        cd11 = xscale*ctheta; cd12 = xscale*stheta
        cd21 = -yscale*stheta; cd22 = yscale*ctheta
    """
    scaling = pixstep / typscale
    xscale = abs(scaling / scale) if scale != 0.0 else 0.0
    yscale = abs(scaling / (scale * abs(aspect))) if scale * aspect != 0.0 else 0.0
    ctheta = math.cos(posangle_deg * DEG)
    stheta = math.sin(posangle_deg * DEG)
    cd11 = xscale * ctheta
    cd12 = xscale * stheta
    cd21 = -yscale * stheta
    cd22 = yscale * ctheta
    return cd11, cd12, cd21, cd22


# ---------------------------------------------------------------------------
# Background model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_back(flux: float, width: int, height: int) -> jnp.ndarray:
    """Flat background offset of value ``|flux|``."""
    return jnp.full((height, width), jnp.abs(jnp.float32(flux)), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Dirac (point-source) model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_dirac(width: int, height: int) -> jnp.ndarray:
    """
    Point-source model: 1 at the centre pixel, 0 elsewhere.
    Flux scaling is applied externally.
    """
    pix = jnp.zeros((height, width), dtype=jnp.float32)
    cx, cy = width // 2, height // 2
    return pix.at[cy, cx].set(1.0)


# ---------------------------------------------------------------------------
# Sérsic / de Vaucouleurs / Exponential  (the heavy-weight renderer)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def _render_sersic_exponential(
    n: float,
    bn: float,
    cd11: float, cd12: float,
    cd21: float, cd22: float,
    width: int, height: int,
    rs: float,
) -> jnp.ndarray:
    """
    Core renderer shared by MODEL_SERSIC / MODEL_DEVAUCOULEURS /
    MODEL_EXPONENTIAL.

    Implements the smooth + sharp profile using only vectorised JAX ops.
    The C code computes half the image and mirrors it (using symmetry of the
    profile); here we compute all pixels directly – cleaner for JAX.

    Parameters
    ----------
    n, bn  : Sérsic index and corresponding b_n value.
    cd**   : 2×2 coordinate matrix entries.
    width, height : output image size.
    rs     : smooth/sharp transition radius in internal (u,v) space.

    Returns
    -------
    pix : jnp.ndarray, shape (height, width), float32  (un-normalised)
    """
    x1_grid, x2_grid = _make_coords(width, height)

    # Transform to profile internal coordinates
    x1in = cd11 * x1_grid + cd12 * x2_grid   # shape (H, W)
    x2in = cd21 * x1_grid + cd22 * x2_grid

    ra = x1in * x1in + x2in * x2in  # r² in profile space

    rs2 = rs * rs
    invn = 1.0 / n
    hinvn = 0.5 / n
    k = -bn

    # Polynomial coefficients for the smooth central region  (matches C)
    krspinvn = (k * rs) if n == 1.0 else k * jnp.exp(jnp.log(rs) * invn)
    ekrspinvn = jnp.exp(krspinvn)
    p2 = krspinvn * invn * invn
    p1 = krspinvn * p2
    a0 = (1.0 + (1.0 / 6.0) * (p1 + (1.0 - 5.0 * n) * p2)) * ekrspinvn
    a2 = (-0.5) * (p1 + (1.0 - 3.0 * n) * p2) / rs2 * ekrspinvn
    a3 = (1.0 / 3.0) * (p1 + (1.0 - 2.0 * n) * p2) / (rs2 * rs) * ekrspinvn

    # Outer profile: exp(-bn * (r/re)^(1/n))  –  two cases
    if n == 1.0:
        # Exponential: exp(-sqrt(ra))
        outer = jnp.exp(-jnp.sqrt(ra))
    else:
        # General Sérsic: exp(k * ra^hinvn)
        # Numerically stable: k * exp(log(ra) * hinvn), guarded for ra≤0
        safe_ra = jnp.where(ra > 0.0, ra, 1e-30)
        outer = jnp.exp(k * jnp.exp(jnp.log(safe_ra) * hinvn))

    # Smooth central polynomial (avoids catastrophic cancellation near ra=0)
    inner = a0 + ra * (a2 + a3 * jnp.sqrt(jnp.maximum(ra, 0.0)))

    # Select inner vs outer
    pix = jnp.where(ra < rs2, inner, outer)

    return pix.astype(jnp.float32)


def model_sersic(
    scale: float, aspect: float, posangle_deg: float,
    sersic_n: float,
    pixstep: float,
    width: int, height: int,
    typscale: float = 1.0,
) -> jnp.ndarray:
    """
    Render a Sérsic profile with arbitrary index *sersic_n*.

    Parameters
    ----------
    scale       : effective radius in pixels.
    aspect      : axis ratio b/a.
    posangle_deg: position angle (degrees E of N).
    sersic_n    : Sérsic index (0.5 ≤ n ≤ 10 recommended).
    pixstep     : ratio of model pixel scale to object pixel scale.
    width, height : output image dimensions (static).
    typscale    : normalisation scale (default 1.0).

    Returns
    -------
    pix : jnp.ndarray, shape (height, width), float32
    """
    n = float(abs(sersic_n))
    bn = (2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
          + 46.0 / (25515.0 * n * n) + 131.0 / (1148175.0 * n * n * n))
    cd11, cd12, cd21, cd22 = _cd_matrix(scale, aspect, posangle_deg,
                                         pixstep, typscale)
    xscale = abs((pixstep / typscale) / scale) if scale != 0 else 0.0
    yscale = abs((pixstep / typscale) / (scale * abs(aspect))) if scale * aspect != 0 else 0.0
    rs = PROFIT_SMOOTHR * max(xscale, yscale)
    if rs <= 0:
        rs = 1.0
    return _render_sersic_exponential(n, bn, cd11, cd12, cd21, cd22,
                                      width, height, rs)


def model_devaucouleurs(
    scale: float, aspect: float, posangle_deg: float,
    pixstep: float,
    width: int, height: int,
    typscale: float = 1.0,
) -> jnp.ndarray:
    """de Vaucouleurs profile (Sérsic n=4, bn=7.66924944)."""
    n = 4.0
    bn = 7.66924944
    cd11, cd12, cd21, cd22 = _cd_matrix(scale, aspect, posangle_deg,
                                         pixstep, typscale)
    xscale = abs((pixstep / typscale) / scale) if scale != 0 else 0.0
    yscale = abs((pixstep / typscale) / (scale * abs(aspect))) if scale * aspect != 0 else 0.0
    rs = PROFIT_SMOOTHR * max(xscale, yscale)
    if rs <= 0:
        rs = 1.0
    return _render_sersic_exponential(n, bn, cd11, cd12, cd21, cd22,
                                      width, height, rs)


def model_exponential(
    scale: float, aspect: float, posangle_deg: float,
    pixstep: float,
    width: int, height: int,
    typscale: float = 1.0,
) -> jnp.ndarray:
    """Exponential disk profile (Sérsic n=1, bn=1)."""
    n = 1.0
    bn = 1.0
    cd11, cd12, cd21, cd22 = _cd_matrix(scale, aspect, posangle_deg,
                                         pixstep, typscale)
    xscale = abs((pixstep / typscale) / scale) if scale != 0 else 0.0
    yscale = abs((pixstep / typscale) / (scale * abs(aspect))) if scale * aspect != 0 else 0.0
    rs = PROFIT_SMOOTHR * max(xscale, yscale)
    if rs <= 0:
        rs = 1.0
    return _render_sersic_exponential(n, bn, cd11, cd12, cd21, cd22,
                                      width, height, rs)


# ---------------------------------------------------------------------------
# Spiral arms model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_arms(
    cd11: float, cd12: float, cd21: float, cd22: float,
    featstart: float,
    featfrac: float,
    featpitch_deg: float,
    featposang_deg: float,
    width: int, height: int,
) -> jnp.ndarray:
    """
    Logarithmic spiral arm model (MODEL_ARMS).

    Replicates the MODEL_ARMS case in ``prof_add()``.
    """
    x1_grid, x2_grid = _make_coords(width, height)

    # Transform to profile internal coordinates
    x1t = cd11 * x1_grid + cd12 * x2_grid
    x2t = cd21 * x1_grid + cd22 * x2_grid

    r2 = x1t * x1t + x2t * x2t

    r2min = featstart * featstart
    r2minxin = r2min * (1.0 - PROFIT_BARXFADE) ** 2
    r2minxout = r2min * (1.0 + PROFIT_BARXFADE) ** 2
    invr2xdif = jnp.where(r2minxout > r2minxin,
                           1.0 / (r2minxout - r2minxin),
                           1.0)
    umin = 0.5 * jnp.log(r2minxin + 1e-5)

    arm2amp = featfrac
    armamp = 1.0 - arm2amp
    armrdphidr = 1.0 / jnp.tan(jnp.float32(featpitch_deg * DEG))
    posang = jnp.float32(featposang_deg * DEG)
    width_arm = 3.0  # hard-coded in C as well

    # Compute for all pixels
    u = 0.5 * jnp.log(r2 + 1e-5)
    theta = armrdphidr * u + posang
    ca = jnp.cos(theta)
    sa = jnp.sin(theta)
    x1in = x1t * ca - x2t * sa
    x2in = x1t * sa + x2t * ca

    amp_base = jnp.exp(-jnp.sqrt(r2 + 1e-30))
    fade = jnp.clip((r2 - r2minxin) * invr2xdif, 0.0, 1.0)
    amp = amp_base * jnp.where(r2 < r2minxout, fade, 1.0)

    safe_r2 = jnp.where(r2 > 0.0, r2, 1e-30)
    ra = x1in * x1in / safe_r2
    rb = x2in * x2in / safe_r2
    val = amp * (armamp * ra ** width_arm + arm2amp * rb ** width_arm)

    pix = jnp.where(r2 > r2minxin, val, 0.0)
    return pix.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Bar model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_bar(
    cd11: float, cd12: float, cd21: float, cd22: float,
    featstart: float,
    feataspect: float,
    featposang_deg: float,
    width: int, height: int,
) -> jnp.ndarray:
    """
    Galactic bar model (MODEL_BAR).

    Replicates the MODEL_BAR case in ``prof_add()``.
    """
    x1_grid, x2_grid = _make_coords(width, height)

    x1t = cd11 * x1_grid + cd12 * x2_grid
    x2t = cd21 * x1_grid + cd22 * x2_grid

    r2 = x1t * x1t + x2t * x2t

    r2min = featstart * featstart
    r2minxin = r2min * (1.0 - PROFIT_BARXFADE) ** 2
    r2minxout = r2min * (1.0 + PROFIT_BARXFADE) ** 2
    invr2xdif = jnp.where(r2minxout > r2minxin,
                           1.0 / (r2minxout - r2minxin),
                           1.0)

    invwidth2 = abs(1.0 / (featstart * feataspect + 1e-30))
    posang = float(featposang_deg * DEG)
    ca = math.cos(posang)
    sa = math.sin(posang)

    x1in = x1t * ca - x2t * sa
    x2in = x1t * sa + x2t * ca

    fade_in = jnp.clip((r2minxout - r2) * invr2xdif, 0.0, 1.0)
    amp = jnp.exp(-x1in * x1in * invwidth2) * jnp.where(r2 < r2minxout, fade_in, 0.0)

    return amp.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Inner ring model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_inring(
    cd11: float, cd12: float, cd21: float, cd22: float,
    featstart: float,
    featwidth: float,
    feataspect: float,
    width: int, height: int,
) -> jnp.ndarray:
    """
    Inner ring model (MODEL_INRING).
    """
    x1_grid, x2_grid = _make_coords(width, height)
    x1t = cd11 * x1_grid + cd12 * x2_grid
    x2t = cd21 * x1_grid + cd22 * x2_grid

    # Scale x2 by aspect to get ellipse
    x2e = x2t / (abs(feataspect) + 1e-30)
    r = jnp.sqrt(x1t * x1t + x2e * x2e)

    # Gaussian ring centred at featstart
    dr = r - featstart
    sigma = featwidth * featstart + 1e-30
    val = jnp.exp(-0.5 * (dr / sigma) ** 2)

    return val.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Outer ring model
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def model_outring(
    cd11: float, cd12: float, cd21: float, cd22: float,
    featstart: float,
    featwidth: float,
    width: int, height: int,
) -> jnp.ndarray:
    """
    Outer ring model (MODEL_OUTRING).
    """
    x1_grid, x2_grid = _make_coords(width, height)
    x1t = cd11 * x1_grid + cd12 * x2_grid
    x2t = cd21 * x1_grid + cd22 * x2_grid

    r = jnp.sqrt(x1t * x1t + x2t * x2t)
    dr = r - featstart
    sigma = featwidth * featstart + 1e-30
    val = jnp.exp(-0.5 * (dr / sigma) ** 2)

    return val.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Composite model: sum multiple profiles and convolve with PSF
# ---------------------------------------------------------------------------

def render_model(profiles, psf_ft, modpix_shape, pixstep: float):
    """
    Render a list of profile descriptors into a single model image,
    then convolve with the PSF.

    Parameters
    ----------
    profiles : list of dicts, each with keys:
        'type'   : one of 'back','dirac','sersic','devaucouleurs',
                          'exponential','arms','bar','inring','outring'
        'flux'   : float (total flux, used only for scaling outside)
        ... (profile-specific parameters; see individual renderers)
    psf_ft   : jnp.ndarray, shape (H, W//2+1), complex64
               Pre-computed FFT of the PSF (from jax_fft.fft_rtf).
    modpix_shape : (H, W) – shape of the model pixel array.
    pixstep  : pixel step (ratio of model to object pixel scale).

    Returns
    -------
    cmodpix : jnp.ndarray, shape (H, W), float32
              PSF-convolved composite model image.
    """
    from .jax_fft import fft_conv  # local import to avoid circular deps
    H, W = modpix_shape
    modpix = jnp.zeros((H, W), dtype=jnp.float32)

    for p in profiles:
        ptype = p['type']
        if ptype == 'back':
            modpix = modpix + model_back(p['flux'], W, H)
        elif ptype == 'dirac':
            modpix = modpix + p['flux'] * model_dirac(W, H)
        elif ptype == 'sersic':
            modpix = modpix + p['flux'] * model_sersic(
                p['scale'], p['aspect'], p['posangle'], p['sersic_n'],
                pixstep, W, H)
        elif ptype == 'devaucouleurs':
            modpix = modpix + p['flux'] * model_devaucouleurs(
                p['scale'], p['aspect'], p['posangle'],
                pixstep, W, H)
        elif ptype == 'exponential':
            modpix = modpix + p['flux'] * model_exponential(
                p['scale'], p['aspect'], p['posangle'],
                pixstep, W, H)
        elif ptype == 'arms':
            cd = _cd_matrix(p['scale'], p['aspect'], p['posangle'],
                            pixstep, p.get('typscale', 1.0))
            modpix = modpix + p['flux'] * model_arms(
                *cd, p['featstart'], p['featfrac'],
                p['featpitch'], p['featposang'], W, H)
        elif ptype == 'bar':
            cd = _cd_matrix(p['scale'], p['aspect'], p['posangle'],
                            pixstep, p.get('typscale', 1.0))
            modpix = modpix + p['flux'] * model_bar(
                *cd, p['featstart'], p['feataspect'], p['featposang'], W, H)
        elif ptype == 'inring':
            cd = _cd_matrix(p['scale'], p['aspect'], p['posangle'],
                            pixstep, p.get('typscale', 1.0))
            modpix = modpix + p['flux'] * model_inring(
                *cd, p['featstart'], p['featwidth'], p['feataspect'], W, H)
        elif ptype == 'outring':
            cd = _cd_matrix(p['scale'], p['aspect'], p['posangle'],
                            pixstep, p.get('typscale', 1.0))
            modpix = modpix + p['flux'] * model_outring(
                *cd, p['featstart'], p['featwidth'], W, H)

    # PSF convolution
    cmodpix = fft_conv(modpix, psf_ft, output_shape=(H, W))
    return cmodpix


# ---------------------------------------------------------------------------
# Residuals computation  (profit_compresi equivalent)
# ---------------------------------------------------------------------------

@jax.jit
def compute_residuals(lmodpix: jnp.ndarray,
                      objpix: jnp.ndarray,
                      objweight: jnp.ndarray,
                      dynparam: float = 0.0) -> jnp.ndarray:
    """
    Compute weighted residuals between model and data (profit_compresi).

    Parameters
    ----------
    lmodpix   : jnp.ndarray, shape (N,), float32  – resampled model pixels
    objpix    : jnp.ndarray, shape (N,), float32  – observed pixels
    objweight : jnp.ndarray, shape (N,), float32  – inverse-variance weights
    dynparam  : float  – dynamic compression parameter (0 = no compression)

    Returns
    -------
    resi  : jnp.ndarray, shape (<=N,), float32  – residuals for valid pixels
    chi2  : float  – sum of squared residuals
    """
    valid = objweight > 0.0

    if dynparam > 0.0:
        invsig = 1.0 / dynparam
        diff = (lmodpix - objpix) * objweight * invsig
        # Compressed: sign(diff) * log(1 + |diff|)
        val2 = jnp.where(diff > 0.0, jnp.log1p(diff), -jnp.log1p(-diff))
        resi_all = val2 * dynparam
        chi2 = jnp.sum(jnp.where(valid, val2 * val2, 0.0)) * dynparam * dynparam
    else:
        resi_all = (lmodpix - objpix) * objweight
        chi2 = jnp.sum(jnp.where(valid, resi_all * resi_all, 0.0))

    resi = jnp.where(valid, resi_all, 0.0)
    return resi, chi2


# ---------------------------------------------------------------------------
# Spiral index computation  (profit_spiralindex equivalent)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("width", "height"))
def compute_spiral_index(objpix: jnp.ndarray,
                         backsig: float,
                         guessradius: float,
                         mx: float, my: float,
                         width: int, height: int) -> float:
    """
    Compute the spiral index estimator (profit_spiralindex in profit.c).

    Uses FFT-based derivative filters then evaluates a curl-like measure.

    Parameters
    ----------
    objpix      : jnp.ndarray, shape (height, width), float32
    backsig     : float – background sigma
    guessradius : float – estimated object radius in pixels
    mx, my      : float – object centroid offsets
    width, height : int – image dimensions (static)

    Returns
    -------
    spirindex : float
    """
    from .jax_fft import fft_rtf, fft_conv  # local import

    fwhm = max(guessradius * 2.0 / 4.0, 2.0)
    sep = 2.0
    invtwosigma2 = -(2.35 * 2.35 / (2.0 * fwhm * fwhm))

    hw = width / 2.0
    hh = height / 2.0

    x1_grid, x2_grid = _make_coords(width, height)

    # Wrap coordinates (same periodicity trick as C code)
    x = jnp.where(x1_grid < -0.5, x1_grid + hw, x1_grid - (width - hw))
    y = jnp.where(x2_grid < -0.5, x2_grid + hh, x2_grid - (height - hh))

    # Derivative filters
    dx = (jnp.exp(invtwosigma2 * ((x + sep) ** 2 + y * y))
          - jnp.exp(invtwosigma2 * ((x - sep) ** 2 + y * y)))
    dy = (jnp.exp(invtwosigma2 * (x * x + (y + sep) ** 2))
          - jnp.exp(invtwosigma2 * (x * x + (y - sep) ** 2)))

    # Log-compress object pixels
    invsig = float(width * height) / (backsig + 1e-30)
    val = jnp.where(objpix > -1e29, objpix * invsig, 0.0)
    gdx = jnp.where(val > 0.0, jnp.log1p(val), -jnp.log1p(-val))
    gdy = gdx.copy()

    # Convolve with derivative filters (FFT)
    fdx = fft_rtf(dx)
    fdy = fft_rtf(dy)
    gdx = fft_conv(gdx, fdx, output_shape=(height, width))
    gdy = fft_conv(gdy, fdy, output_shape=(height, width))

    # Spiral index estimator
    invtwosigma2_r = -1.18 * 1.18 / (2.0 * guessradius * guessradius)
    xstart = -hw - mx + jnp.round(mx)
    ystart = -hh - my + jnp.round(my)

    xc = x1_grid + xstart + hw   # proper centred coords
    yc = x2_grid + ystart + hh

    r2 = xc * xc + yc * yc
    safe_r2 = jnp.where(r2 > 0.0, r2, 1.0)
    contrib = ((xc * yc * (gdx * gdx - gdy * gdy)
                + gdx * gdy * (yc * yc - xc * xc)) / safe_r2
               * jnp.exp(invtwosigma2_r * r2))
    spirindex = jnp.sum(jnp.where(r2 > 0.0, contrib, 0.0))
    return spirindex
