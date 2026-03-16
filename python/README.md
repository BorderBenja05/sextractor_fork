# sextractor_jax

JAX-accelerated computational kernels for SExtractor, plus an in-memory
Python interface that passes astropy HDUs directly to the `sex` binary
without writing image files to disk.

## Contents

- [Installation](#installation)
- [In-memory SExtractor interface](#in-memory-sextractor-interface)
  - [Quick-start](#quick-start)
  - [Weight maps](#weight-maps)
  - [WCS and world coordinates](#wcs-and-world-coordinates)
  - [Batch processing](#batch-processing)
  - [Memory backend](#memory-backend)
- [JAX kernels](#jax-kernels)
  - [FFT convolution](#fft-convolution)
  - [Image filtering](#image-filtering)
  - [Background estimation](#background-estimation)
  - [Aperture photometry](#aperture-photometry)
  - [Galaxy profile models](#galaxy-profile-models)
  - [Levenberg-Marquardt optimiser](#levenberg-marquardt-optimiser)
- [Running on GPU / TPU](#running-on-gpu--tpu)
- [Running the tests](#running-the-tests)

---

## Installation

```bash
# CPU-only (default)
pip install -e python/

# With CUDA 12 GPU support
pip install -e "python/[gpu]"

# With TPU support
pip install -e "python/[tpu]"

# Development extras (pytest, etc.)
pip install -e "python/[dev]"
```

The `sex` binary must also be on your `PATH`:

```bash
# Debian / Ubuntu
apt install source-extractor

# Conda
conda install -c conda-forge astromatic-source-extractor
```

---

## In-memory SExtractor interface

`sextractor_jax.SExtractor` wraps the `sex` command-line binary and passes
image data through RAM-backed file descriptors instead of writing temporary
FITS files to disk.  The result is an `astropy.table.Table`.

### Quick-start

```python
import numpy as np
from astropy.io import fits
from sextractor_jax import SExtractor, sex_run, memory_backend

# Which memory backend will be used on this system?
print(memory_backend())   # e.g. "memfd" / "/dev/shm" / "tempfile"

# --- Option A: one-liner convenience function ---
rng = np.random.default_rng(42)
data = rng.normal(1000.0, 15.0, (512, 512)).astype(np.float32)

cat = sex_run(data, config={"DETECT_THRESH": 3.0, "DETECT_MINAREA": 5})
print(cat["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])

# --- Option B: reusable SExtractor object ---
sx = SExtractor(
    config={
        "DETECT_THRESH":  3.0,
        "DETECT_MINAREA": 5,
        "BACK_SIZE":      64,
    },
    params=[
        "NUMBER",
        "X_IMAGE", "Y_IMAGE",
        "FLUX_AUTO", "FLUXERR_AUTO",
        "MAG_AUTO", "MAGERR_AUTO",
        "FWHM_IMAGE",
        "CLASS_STAR",
        "FLAGS",
    ],
)

hdu = fits.PrimaryHDU(data)
cat = sx.run(hdu)
```

### Weight maps

```python
science_hdu = fits.PrimaryHDU(data)

weight = np.ones_like(data)
weight[:, 120:125] = 0.0          # mask bad columns
weight_hdu = fits.PrimaryHDU(weight)

sx = SExtractor(config={"DETECT_THRESH": 2.5, "WEIGHT_TYPE": "MAP_WEIGHT"})
cat = sx.run(science_hdu, weight_hdu=weight_hdu)
```

### WCS and world coordinates

Pass an HDU with a complete WCS header and request `X_WORLD` / `Y_WORLD`
output columns:

```python
header = fits.Header()
header["CTYPE1"] = "RA---TAN"
header["CTYPE2"] = "DEC--TAN"
header["CRPIX1"] = 128.0
header["CRPIX2"] = 128.0
header["CRVAL1"] = 150.0
header["CRVAL2"] =   2.0
header["CD1_1"]  = -2.78e-4
header["CD2_2"]  =  2.78e-4

hdu = fits.PrimaryHDU(data, header=header)

sx = SExtractor(
    config={"PIXEL_SCALE": 0},    # 0 = read from WCS
    params=["X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD", "MAG_AUTO"],
)
cat = sx.run(hdu)
print(cat["X_WORLD", "Y_WORLD"])  # RA / Dec in degrees
```

### Batch processing

A `SExtractor` object can be reused across many images without reloading
configuration or spawning extra processes:

```python
sx = SExtractor(config={"DETECT_THRESH": 3.5})

for hdu in hdulist[1:]:          # iterate over multi-extension FITS
    cat = sx.run(hdu)
    print(f"{hdu.name}: {len(cat)} sources")
```

### Memory backend

On Linux ≥ 3.17 the interface uses `memfd_create(2)` to create anonymous
RAM-backed file descriptors.  The image bytes are written to the memfd and the
path `/proc/<pid>/fd/<n>` is passed to the `sex` subprocess, which opens it
via CFITSIO as a normal seekable file — no disk I/O occurs.

On systems without `memfd_create` (macOS, older kernels) the library falls
back automatically:

| Priority | Backend | Notes |
|----------|---------|-------|
| 1 | `memfd_create` | Linux ≥ 3.17; fully RAM-backed, no cleanup needed |
| 2 | `/dev/shm` | Linux RAM tmpfs; file deleted on context exit |
| 3 | `tempfile` | Standard temporary file on disk (any OS) |

`memory_backend()` returns a string identifying which backend will be used.

---

## JAX kernels

All functions accept `jnp.ndarray` (or plain `np.ndarray` / Python scalars
where noted) and are `jax.jit`-compatible.  Import from the top-level package:

```python
from sextractor_jax import (
    fft_rtf, fft_conv, fft_conv_batch,
    convolve_image, make_gaussian_mask,
    make_backmap, interp_backmap, subtract_background,
    compute_aperflux, flux_to_mag,
    model_sersic, render_model, compute_residuals,
    levmar_fit, PARFIT_LINBOUND,
)
```

### FFT convolution

Replaces `fft.c` / FFTW3.

```python
from sextractor_jax import fft_rtf, fft_conv, fft_conv_full, fft_conv_batch
import jax.numpy as jnp

# Pre-compute kernel FFT once, then reuse for many images
kernel_ft = fft_rtf(psf)                     # shape (H, W//2+1) complex64
result    = fft_conv(image, kernel_ft,        # shape (H, W) float32
                     output_shape=(H, W))

# Convenience: compute both FFTs internally
result = fft_conv_full(image, kernel)

# Batch: N images, one kernel
images   = jnp.stack([img1, img2, img3])      # (N, H, W)
results  = fft_conv_batch(images, kernel_ft)  # (N, H, W)
```

### Image filtering

Replaces `filter.c`.

```python
from sextractor_jax import (
    convolve_image, make_gaussian_mask, make_tophat_mask, make_mexhat_mask,
    apply_filter_batch,
)

# Build a kernel
gaussian = make_gaussian_mask(fwhm=3.0, size=9)   # (9, 9) float32, sums to 1
tophat   = make_tophat_mask(radius=3.0)
mexhat   = make_mexhat_mask(fwhm=2.0)

# Convolve (edge-padding, matching SExtractor C behaviour)
mh, mw   = gaussian.shape
filtered = convolve_image(image, gaussian, mh, mw, normalise=True)

# Batch
results  = apply_filter_batch(images, gaussian, mh, mw)  # (N, H, W)
```

### Background estimation

Replaces `back.c`.

```python
from sextractor_jax import (
    make_backmap, interp_backmap, subtract_background,
    backstat, backhisto, backguess,
)

# Full pipeline
back_map, sigma_map, grid = make_backmap(
    image,
    weight=None,    # or a weight array
    mesh_w=64,
    mesh_h=64,
)
# back_map : (ny, nx) float32 — background per mesh
# sigma_map: (ny, nx) float32 — noise sigma per mesh

# Interpolate to full image resolution
H, W      = image.shape
back_full = interp_backmap(back_map, H, W)   # (H, W)

# Subtract
cleaned   = subtract_background(image, back_full)
```

### Aperture photometry

Replaces `photom.c`.

```python
from sextractor_jax import (
    compute_aperflux, compute_autoflux, compute_kron_radius,
    flux_to_mag, flux_to_magerr,
)

# Circular aperture
flux, ferr, area = compute_aperflux(
    image,
    weight=None,
    mx=256.0,        # aperture centre x (pixels)
    my=256.0,        # aperture centre y
    raper=10.0,      # aperture radius (pixels)
    backsig=5.0,     # background sigma (for noise)
    gain=1.0,
    bkg=100.0,       # background level to subtract
)

# Convert to magnitudes
mag    = flux_to_mag(flux, zeropoint=25.0)
magerr = flux_to_magerr(flux, ferr)

# Kron auto-aperture (like MAG_AUTO)
r_kron              = compute_kron_radius(image, weight, mx, my, r_search=20.0)
auto_flux, auto_err = compute_autoflux(image, weight, mx, my, r_kron,
                                       backsig=5.0, gain=1.0, bkg=100.0)
```

### Galaxy profile models

Replaces `profit.c` profile rendering (`prof_add`).

```python
from sextractor_jax import (
    model_sersic, model_devaucouleurs, model_exponential,
    model_arms, model_bar, model_inring, model_outring,
    model_back, model_dirac,
    render_model, compute_residuals,
)

H, W = 64, 64

# Sérsic profile
sersic = model_sersic(
    scale=8.0,           # effective radius (pixels)
    aspect=0.7,          # minor/major axis ratio
    posangle_deg=30.0,   # position angle (degrees CCW from x-axis)
    sersic_n=2.0,        # Sérsic index (1=exponential, 4=de Vaucouleurs)
    pixstep=1.0,
    width=W, height=H,
)

# de Vaucouleurs (n=4) and exponential disk (n=1) shortcuts
bulge = model_devaucouleurs(scale=6.0, aspect=0.8, posangle_deg=0.0,
                             pixstep=1.0, width=W, height=H)
disk  = model_exponential(scale=12.0, aspect=0.9, posangle_deg=10.0,
                           pixstep=1.0, width=W, height=H)

# Spiral arms
from sextractor_jax.jax_models import _cd_matrix
cd = _cd_matrix(scale=10.0, aspect=0.6, posangle_deg=0.0, pixstep=1.0)
arms = model_arms(*cd,
                  featstart=3.0, featfrac=0.5,
                  featpitch_deg=20.0, featposang_deg=0.0,
                  width=W, height=H)

# Composite render with PSF convolution
from sextractor_jax import fft_rtf
psf_ft = fft_rtf(psf_image)
modpix = render_model(
    profiles=[("sersic", sersic_params), ("disk", disk_params)],
    psf_ft=psf_ft,
    modpix_shape=(H, W),
    pixstep=1.0,
)

# Residuals
lmodpix  = modpix.ravel()
objpix   = data_patch.ravel()
objweight = weight_patch.ravel()
residuals, chi2 = compute_residuals(lmodpix, objpix, objweight)
```

### Levenberg-Marquardt optimiser

Replaces the `levmar/` library used by `profit_minimize()`.

```python
from sextractor_jax import (
    levmar_fit, levmar_step,
    bounded_to_unbounded, unbounded_to_bounded, propagate_covar,
    PARFIT_FIXED, PARFIT_UNBOUND, PARFIT_LINBOUND, PARFIT_LOGBOUND,
)
import jax.numpy as jnp

# Define a residual function: returns a vector of residuals
def residuals(params, data, model_fn):
    model = model_fn(params)
    return (model - data).ravel()

p0 = jnp.array([8.0, 0.7, 30.0, 2.0])   # initial guess
lo = jnp.array([0.5, 0.1,  0.0, 0.5])   # lower bounds
hi = jnp.array([50., 1.0, 180., 8.0])   # upper bounds
ftypes = [PARFIT_LINBOUND] * 4           # all linearly bounded

result = levmar_fit(
    residuals, p0,
    args=(observed_data, model_fn),
    lo=lo, hi=hi, ftypes=ftypes,
    max_iter=200,
    tau=1e-3,
)

print(result["params"])   # converged parameters
print(result["chi2"])     # final chi²
print(result["niter"])    # iterations used
print(result["covar"])    # parameter covariance matrix
```

---

## Running on GPU / TPU

No code changes are required.  Install the appropriate `jaxlib` variant and
JAX will use the accelerator automatically:

```bash
# CUDA 12
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# TPU
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Force CPU execution for reproducibility:

```python
import jax
jax.config.update("jax_platform_name", "cpu")
```

---

## Running the tests

```bash
cd python

# Unit tests (no binary or GPU required)
pytest tests/ -v

# Include integration tests (requires sex binary in PATH)
pytest tests/ -v -m integration

# JAX kernel smoke tests only
pytest tests/test_jax_kernels.py -v

# SExtractor interface tests only
pytest tests/test_sextractor_interface.py -v
```
