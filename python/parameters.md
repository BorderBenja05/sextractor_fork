# sextractor_jax — API Reference

Full parameter reference for all public functions and classes.

## Contents

- [SExtractor interface](#sextractor-interface)
  - [SExtractor class](#sextractor-class)
  - [sex_run](#sex_run)
  - [memory_backend](#memory_backend)
  - [Low-level helpers](#low-level-helpers)
- [jax_fft](#jax_fft)
- [jax_filter](#jax_filter)
- [jax_back](#jax_back)
- [jax_photom](#jax_photom)
- [jax_models](#jax_models)
- [jax_optimize](#jax_optimize)
- [SExtractor config keys](#sextractor-config-keys)
- [SExtractor output parameters](#sextractor-output-parameters)

---

## SExtractor interface

### SExtractor class

```python
class SExtractor(
    binary: str = "sex",
    config: dict = None,
    params: list[str] = DEFAULT_PARAMS,
    extra_args: list[str] = None,
    timeout: float | None = None,
    raise_on_error: bool = True,
)
```

#### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `binary` | `str` | `"sex"` | Name or full path of the SExtractor executable. Tries `"sex"` then `"source-extractor"` automatically. |
| `config` | `dict` | `{}` | SExtractor configuration keys to override (see [Config keys](#sextractor-config-keys)). |
| `params` | `list[str]` | See below | Output catalog columns to request (see [Output parameters](#sextractor-output-parameters)). |
| `extra_args` | `list[str]` | `[]` | Extra CLI arguments appended verbatim to the `sex` command (e.g. `["-CHECKIMAGE_TYPE", "BACKGROUND"]`). |
| `timeout` | `float` or `None` | `None` | Subprocess timeout in seconds.  `None` = no limit. |
| `raise_on_error` | `bool` | `True` | Raise `RuntimeError` if `sex` exits with a non-zero return code. |

Default `params`:
```python
DEFAULT_PARAMS = [
    "NUMBER", "X_IMAGE", "Y_IMAGE",
    "FLUX_AUTO", "FLUXERR_AUTO",
    "MAG_AUTO", "MAGERR_AUTO",
    "FWHM_IMAGE", "FLAGS",
]
```

#### `SExtractor.run`

```python
def run(
    hdu: fits.HDU | fits.HDUList | np.ndarray,
    weight_hdu: fits.HDU | None = None,
    config: dict | None = None,
) -> astropy.table.Table
```

Run SExtractor on an in-memory image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `hdu` | `PrimaryHDU`, `ImageHDU`, `HDUList`, or `ndarray` | Science image.  Arrays are wrapped in a `PrimaryHDU` automatically. |
| `weight_hdu` | `HDU` or `None` | Optional weight map.  Must set `WEIGHT_TYPE` in `config` (e.g. `"MAP_WEIGHT"`, `"MAP_RMS"`, `"MAP_VAR"`). |
| `config` | `dict` or `None` | Per-call config overrides merged on top of the instance config. |

Returns an `astropy.table.Table` with one row per detected source and one column per requested output parameter.

Raises `RuntimeError` if `sex` exits non-zero and `raise_on_error=True`.

---

### sex_run

```python
def sex_run(
    hdu: fits.HDU | fits.HDUList | np.ndarray,
    config: dict | None = None,
    params: list[str] | None = None,
    **kwargs,
) -> astropy.table.Table
```

Module-level convenience function.  Creates a temporary `SExtractor` object and calls `.run()`.  Equivalent to:

```python
SExtractor(config=config, params=params, **kwargs).run(hdu)
```

---

### memory_backend

```python
def memory_backend() -> str
```

Return a string identifying the memory backend that will be used on this
system: `"memfd"`, `"/dev/shm"`, or `"tempfile"`.

---

### Low-level helpers

These are exported for testing and advanced use.

#### `_hdu_to_bytes`

```python
def _hdu_to_bytes(hdu: fits.HDU | fits.HDUList | np.ndarray) -> bytes
```

Serialise an HDU (or raw numpy array) to FITS-format bytes in memory.

#### `_memfd_available`

```python
def _memfd_available() -> bool
```

Return `True` if `memfd_create(2)` is available on this kernel.

#### `_memory_input_file`

```python
@contextmanager
def _memory_input_file(data: bytes, suffix: str = ".fits")
    -> Iterator[tuple[str, int | None]]
```

Context manager.  Writes `data` to a RAM-backed (or temp) file and yields
`(path, fd)`.  `path` is a string that CFITSIO can open; `fd` is the file
descriptor integer (or `None` for filesystem backends).  File is cleaned up
on context exit.

#### `_memory_output_file`

```python
@contextmanager
def _memory_output_file(suffix: str = ".cat")
    -> Iterator[tuple[str, Callable[[], bytes]]]
```

Context manager.  Yields `(path, read_fn)`.  After `sex` writes the catalog
to `path`, call `read_fn()` to retrieve the bytes.  Cleaned up on exit.

#### `_parse_ascii_head`

```python
def _parse_ascii_head(raw: bytes) -> astropy.table.Table
```

Parse a SExtractor `ASCII_HEAD` format catalog from raw bytes.

#### `_parse_fits_ldac`

```python
def _parse_fits_ldac(raw: bytes) -> astropy.table.Table
```

Parse a SExtractor `FITS_LDAC` format catalog from raw bytes.

---

## jax_fft

All functions are `@jax.jit`-compiled and differentiable.

### `fft_rtf`

```python
def fft_rtf(data: jnp.ndarray) -> jnp.ndarray
```

2-D real-to-complex FFT.  Equivalent to `fft_rtf()` in `fft.c`.

| Parameter | Shape | dtype | Description |
|-----------|-------|-------|-------------|
| `data` | `(H, W)` | float32 | Input image |
| **returns** | `(H, W//2+1)` | complex64 | Non-redundant FFT coefficients |

### `fft_conv`

```python
@partial(jax.jit, static_argnames=("output_shape",))
def fft_conv(
    data: jnp.ndarray,
    kernel_ft: jnp.ndarray,
    output_shape: tuple[int, int] | None = None,
) -> jnp.ndarray
```

Convolve `data` with a pre-computed Fourier-domain kernel.

| Parameter | Description |
|-----------|-------------|
| `data` | `(H, W)` float32 — input image |
| `kernel_ft` | `(H, W//2+1)` complex64 — pre-computed kernel FFT from `fft_rtf` |
| `output_shape` | `(H, W)` tuple, or `None` to use `data.shape` |
| **returns** | `(H, W)` float32 — convolved image |

### `fft_conv_full`

```python
def fft_conv_full(data: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray
```

Convenience: compute both FFTs and convolve.  Both arrays must have the same
shape.

### `fft_conv_batch`

```python
@partial(jax.jit, static_argnames=("output_shape",))
def fft_conv_batch(
    images: jnp.ndarray,
    kernel_ft: jnp.ndarray,
    output_shape: tuple[int, int] | None = None,
) -> jnp.ndarray
```

Batch convolution via `jax.vmap`.

| Parameter | Description |
|-----------|-------------|
| `images` | `(N, H, W)` float32 |
| `kernel_ft` | `(H, W//2+1)` complex64 |
| **returns** | `(N, H, W)` float32 |

---

## jax_filter

### `convolve_image`

```python
@partial(jax.jit, static_argnames=("mask_h", "mask_w", "normalise"))
def convolve_image(
    image: jnp.ndarray,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
    normalise: bool = True,
) -> jnp.ndarray
```

2-D convolution with edge padding (matches SExtractor C boundary behaviour).

| Parameter | Description |
|-----------|-------------|
| `image` | `(H, W)` float32 |
| `mask` | `(mask_h, mask_w)` float32 — convolution kernel |
| `mask_h`, `mask_w` | kernel dimensions (static for JIT) |
| `normalise` | divide mask by its sum before convolving |
| **returns** | `(H, W)` float32 |

### `convolve_scanline`

```python
def convolve_scanline(
    strip: jnp.ndarray,
    y: int,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
    strip_h: int,
) -> jnp.ndarray
```

Convolve a single scanline `y` within a strip buffer.  Streaming interface
compatible with the C `convolve()` API.

### `apply_filter_batch`

```python
def apply_filter_batch(
    images: jnp.ndarray,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
) -> jnp.ndarray
```

Apply the same filter to a batch of images. Returns `(N, H, W)`.

### `make_gaussian_mask`

```python
def make_gaussian_mask(fwhm: float, size: int | None = None) -> jnp.ndarray
```

Generate a normalised 2-D Gaussian kernel.

| Parameter | Description |
|-----------|-------------|
| `fwhm` | Full-width at half-maximum in pixels |
| `size` | Kernel side length (odd integer); defaults to `2 * ceil(2*sigma) + 1` |
| **returns** | `(size, size)` float32, sums to 1 |

### `make_tophat_mask`

```python
def make_tophat_mask(radius: float, size: int | None = None) -> jnp.ndarray
```

Generate a normalised circular top-hat kernel.

### `make_mexhat_mask`

```python
def make_mexhat_mask(fwhm: float, size: int | None = None) -> jnp.ndarray
```

Generate a normalised Mexican-hat (Laplacian-of-Gaussian) kernel.

---

## jax_back

### `backstat`

```python
def backstat(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mesh_w: int,
    mesh_h: int,
    wthresh: float = 0.0,
) -> tuple[list[dict], tuple[int, int]]
```

First-pass per-mesh statistics with 2σ clipping.  Replicates `backstat()` in
`back.c`.

| Parameter | Description |
|-----------|-------------|
| `image` | `(H, W)` float32 |
| `weight` | `(H, W)` float32 or `None` |
| `mesh_w`, `mesh_h` | mesh dimensions in pixels |
| `wthresh` | pixels with `weight < wthresh` are masked |
| **returns** | `(meshes, (ny, nx))` — list of mesh dicts, grid dimensions |

Each mesh dict has keys: `mean`, `sigma`, `lcut`, `hcut`, `nlevels`,
`qscale`, `qzero`, `npix`, `row0`, `col0`, `row1`, `col1`.

### `backhisto`

```python
def backhisto(
    image: jnp.ndarray,
    meshes: list[dict],
    weight: jnp.ndarray | None = None,
    wthresh: float = 0.0,
) -> list[dict]
```

Fill per-mesh integer histograms.  Adds a `histo` key (int32 array) to each
mesh dict.  Bad meshes get `histo = None`.

### `backguess`

```python
def backguess(bkg: dict) -> tuple[float, float]
```

Estimate robust background mean and sigma from a filled mesh dict using
iterative Gaussian fitting on the histogram.

Returns `(mean, sigma)`.  Returns `(-1e29, -1e29)` for bad meshes.

### `make_backmap`

```python
def make_backmap(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mesh_w: int,
    mesh_h: int,
    wthresh: float = 0.0,
    pearson: float = 0.3,
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[int, int]]
```

Full background pipeline: `backstat` → `backhisto` → `backguess`.

| Returns | Shape | Description |
|---------|-------|-------------|
| `back_map` | `(ny, nx)` float32 | Background level per mesh |
| `sigma_map` | `(ny, nx)` float32 | Noise sigma per mesh |
| `grid_shape` | `(ny, nx)` | Number of meshes |

### `interp_backmap`

```python
@partial(jax.jit, static_argnames=("image_h", "image_w"))
def interp_backmap(
    back_map: jnp.ndarray,
    image_h: int,
    image_w: int,
) -> jnp.ndarray
```

Bilinear interpolation of the coarse background map to full image resolution.
Returns `(image_h, image_w)` float32.

### `subtract_background`

```python
@jax.jit
def subtract_background(
    image: jnp.ndarray,
    back_full: jnp.ndarray,
) -> jnp.ndarray
```

Returns `image - back_full` as float32.

### `make_backmap_batch`

```python
def make_backmap_batch(
    images: jnp.ndarray,
    mesh_w: int,
    mesh_h: int,
    wthresh: float = 0.0,
) -> tuple[list, list]
```

Compute background maps for a stack of images `(N, H, W)`.
Returns `(back_maps, sigma_maps)` as lists of length N.

---

## jax_photom

### `compute_aperflux`

```python
def compute_aperflux(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float,
    my: float,
    raper: float,
    backsig: float,
    gain: float,
    bkg: float,
    oversamp: int = 5,
) -> tuple[float, float, float]
```

Circular aperture photometry with sub-pixel oversampling at the edge.

| Parameter | Description |
|-----------|-------------|
| `image` | `(H, W)` float32 |
| `weight` | `(H, W)` float32 or `None` |
| `mx`, `my` | aperture centre (pixels, 0-indexed) |
| `raper` | aperture radius (pixels) |
| `backsig` | background sigma (used for noise estimate) |
| `gain` | detector gain (electrons/ADU); 0 = ignore Poisson noise |
| `bkg` | background level to subtract from each pixel |
| `oversamp` | sub-pixel oversampling factor for edge pixels (default 5) |
| **returns** | `(flux, fluxerr, area)` floats |

### `compute_kron_radius`

```python
def compute_kron_radius(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float,
    my: float,
    r_search: float,
    bkg: float = 0.0,
) -> float
```

Compute the first-moment Kron radius within `r_search` pixels of `(mx, my)`.

### `compute_autoflux`

```python
def compute_autoflux(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float,
    my: float,
    r_kron: float,
    backsig: float,
    gain: float,
    bkg: float,
    kron_factor: float = 2.5,
    kron_min_radius: float = 3.5,
) -> tuple[float, float]
```

Kron auto-aperture flux (replicates `MAG_AUTO`).  Returns `(flux, fluxerr)`.

### `compute_petroflux`

```python
def compute_petroflux(
    image: jnp.ndarray,
    weight: jnp.ndarray | None,
    mx: float,
    my: float,
    r_max: float,
    backsig: float,
    gain: float,
    bkg: float,
    petro_factor: float = 2.0,
    eta: float = 0.2,
) -> tuple[float, float]
```

Petrosian aperture flux.  Returns `(flux, fluxerr)`.

### `flux_to_mag`

```python
def flux_to_mag(
    flux: jnp.ndarray,
    zeropoint: float = 0.0,
) -> jnp.ndarray
```

Convert flux to AB magnitude: `mag = -2.5 * log10(flux) + zeropoint`.
Returns `-99.0` for non-positive flux (matching SExtractor convention).

### `flux_to_magerr`

```python
def flux_to_magerr(
    flux: jnp.ndarray,
    fluxerr: jnp.ndarray,
) -> jnp.ndarray
```

Convert flux error to magnitude error: `magerr = 1.0857 * fluxerr / flux`.

---

## jax_models

### `model_back`

```python
def model_back(level: float, width: int, height: int) -> jnp.ndarray
```

Uniform background model.  Returns `(height, width)` float32 filled with
`level`.

### `model_dirac`

```python
def model_dirac(width: int, height: int) -> jnp.ndarray
```

Delta function at the image centre.  Returns `(height, width)` float32 with
1.0 at `(height//2, width//2)`.

### `model_sersic`

```python
@partial(jax.jit, static_argnames=("width", "height"))
def model_sersic(
    scale: float,
    aspect: float,
    posangle_deg: float,
    sersic_n: float,
    pixstep: float,
    width: int,
    height: int,
) -> jnp.ndarray
```

Sérsic profile `I ∝ exp(-b_n ((r/r_e)^(1/n) - 1))`.

| Parameter | Description |
|-----------|-------------|
| `scale` | Effective (half-light) radius in pixels |
| `aspect` | Minor-to-major axis ratio (0 < aspect ≤ 1) |
| `posangle_deg` | Position angle in degrees, CCW from x-axis |
| `sersic_n` | Sérsic index (1 = exponential, 4 = de Vaucouleurs) |
| `pixstep` | Pixel scale (arcsec/pixel or 1.0 for dimensionless) |
| `width`, `height` | Output image dimensions (static) |
| **returns** | `(height, width)` float32, unnormalised |

### `model_devaucouleurs`

```python
def model_devaucouleurs(
    scale: float, aspect: float, posangle_deg: float,
    pixstep: float, width: int, height: int,
) -> jnp.ndarray
```

de Vaucouleurs profile (Sérsic n=4, b_n=7.66924944).

### `model_exponential`

```python
def model_exponential(
    scale: float, aspect: float, posangle_deg: float,
    pixstep: float, width: int, height: int,
) -> jnp.ndarray
```

Exponential disk profile (Sérsic n=1, b_n=1).

### `model_arms`

```python
def model_arms(
    cd11: float, cd12: float, cd21: float, cd22: float,
    featstart: float,
    featfrac: float,
    featpitch_deg: float,
    featposang_deg: float,
    width: int,
    height: int,
) -> jnp.ndarray
```

Logarithmic spiral arm perturbation added on top of a disk model.

| Parameter | Description |
|-----------|-------------|
| `cd11..cd22` | 2×2 coordinate transformation matrix (from `_cd_matrix`) |
| `featstart` | Inner radius where arms begin (pixels) |
| `featfrac` | Fraction of disk flux in the arms (0–1) |
| `featpitch_deg` | Spiral pitch angle (degrees) |
| `featposang_deg` | Starting position angle of arms (degrees) |

### `model_bar`

```python
def model_bar(
    cd11, cd12, cd21, cd22,
    featstart: float,
    feataspect: float,
    featposang_deg: float,
    width: int,
    height: int,
) -> jnp.ndarray
```

Bar perturbation (elongated Gaussian).

### `model_inring` / `model_outring`

```python
def model_inring(
    cd11, cd12, cd21, cd22,
    featstart: float,
    featwidth: float,
    feataspect: float,
    width: int,
    height: int,
) -> jnp.ndarray

def model_outring(
    cd11, cd12, cd21, cd22,
    featstart: float,
    featwidth: float,
    feataspect: float,
    width: int,
    height: int,
) -> jnp.ndarray
```

Inner and outer ring perturbations.

### `render_model`

```python
def render_model(
    profiles: list[tuple[str, jnp.ndarray]],
    psf_ft: jnp.ndarray,
    modpix_shape: tuple[int, int],
    pixstep: float,
) -> jnp.ndarray
```

Composite model renderer.  Sums profile images, convolves with PSF (via
`fft_conv`), and returns the model pixel array.

### `compute_residuals`

```python
def compute_residuals(
    lmodpix: jnp.ndarray,
    objpix: jnp.ndarray,
    objweight: jnp.ndarray,
    dynparam: float = 1.0,
) -> tuple[jnp.ndarray, float]
```

Returns `(residuals, chi2)` where `residuals = (lmodpix - objpix) * objweight`
and `chi2 = sum(residuals**2)`.

### `compute_spiral_index`

```python
def compute_spiral_index(
    image: jnp.ndarray,
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
) -> float
```

Compute a logarithmic spiral index for a galaxy stamp.

---

## jax_optimize

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `PARFIT_FIXED` | 0 | Parameter is held fixed |
| `PARFIT_UNBOUND` | 1 | No bounds |
| `PARFIT_LINBOUND` | 2 | Linear (logit) bounds between `lo` and `hi` |
| `PARFIT_LOGBOUND` | 3 | Logarithmic bounds |

### `levmar_fit`

```python
def levmar_fit(
    residual_fn: Callable,
    params0: jnp.ndarray,
    args: tuple = (),
    lo: jnp.ndarray | None = None,
    hi: jnp.ndarray | None = None,
    ftypes: list[int] | None = None,
    max_iter: int = 200,
    tau: float = 1e-3,
    eps1: float = 1e-8,
    eps2: float = 1e-8,
) -> dict
```

Levenberg-Marquardt non-linear least squares.

| Parameter | Description |
|-----------|-------------|
| `residual_fn` | `f(params, *args) -> jnp.ndarray` of residuals |
| `params0` | Initial parameter vector |
| `args` | Extra positional arguments to `residual_fn` |
| `lo`, `hi` | Lower/upper bounds (required when `ftypes` includes `PARFIT_LINBOUND`) |
| `ftypes` | List of bound type codes, length `len(params0)` |
| `max_iter` | Maximum LM iterations |
| `tau` | Initial damping factor scale |
| `eps1`, `eps2` | Convergence tolerances on gradient and step size |

Returns a dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `params` | `jnp.ndarray` | Converged parameters |
| `chi2` | `float` | Final `sum(residuals**2)` |
| `niter` | `int` | Iterations used |
| `covar` | `jnp.ndarray` | Parameter covariance matrix `(n, n)` |
| `converged` | `bool` | `True` if a convergence criterion was met |

### `levmar_step`

```python
def levmar_step(
    J: jnp.ndarray,
    r: jnp.ndarray,
    mu: float,
) -> jnp.ndarray
```

Compute a single LM parameter update: solves `(J^T J + μI) Δp = J^T r`.

| Parameter | Description |
|-----------|-------------|
| `J` | Jacobian `(m, n)` float64 |
| `r` | Residual vector `(m,)` |
| `mu` | Damping parameter |
| **returns** | Parameter update `Δp`, shape `(n,)` |

### `levmar_fit_batch`

```python
def levmar_fit_batch(
    residual_fn: Callable,
    params_batch: jnp.ndarray,
    args_batch: tuple = (),
    **kwargs,
) -> list[dict]
```

Run `levmar_fit` independently for each row of `params_batch`.  Returns a
list of result dicts.

### `bounded_to_unbounded`

```python
def bounded_to_unbounded(
    params: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    ftypes: list[int],
) -> jnp.ndarray
```

Map parameters from bounded space to the full real line (logit transform for
`PARFIT_LINBOUND`, log transform for `PARFIT_LOGBOUND`).

### `unbounded_to_bounded`

```python
def unbounded_to_bounded(
    dparams: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    ftypes: list[int],
) -> jnp.ndarray
```

Inverse of `bounded_to_unbounded`.

### `propagate_covar`

```python
def propagate_covar(
    covar_in: jnp.ndarray,
    jac: jnp.ndarray,
) -> jnp.ndarray
```

Propagate covariance through a linear transformation: `jac @ covar_in @ jac.T`.

---

## SExtractor config keys

Commonly used keys that can be passed in the `config` dict.  This is a subset
of the full SExtractor configuration; see the
[SExtractor documentation](https://sextractor.readthedocs.io/) for the
complete list.

### Detection

| Key | Default | Description |
|-----|---------|-------------|
| `DETECT_TYPE` | `"CCD"` | Detector type (`"CCD"` or `"PHOTO"`) |
| `DETECT_MINAREA` | `5` | Minimum number of pixels above threshold |
| `DETECT_THRESH` | `1.5` | Detection threshold (σ above background) |
| `ANALYSIS_THRESH` | `1.5` | Analysis threshold |
| `FILTER` | `"Y"` | Filter image before detection |
| `FILTER_NAME` | `"default.conv"` | Convolution filter file |
| `DEBLEND_NTHRESH` | `32` | Number of deblending sub-thresholds |
| `DEBLEND_MINCONT` | `0.005` | Minimum contrast for deblending |
| `CLEAN` | `"Y"` | Clean spurious detections |
| `CLEAN_PARAM` | `1.0` | Cleaning efficiency |
| `MASK_TYPE` | `"CORRECT"` | Masking type |

### Photometry

| Key | Default | Description |
|-----|---------|-------------|
| `PHOT_APERTURES` | `5` | Aperture diameter(s) in pixels |
| `PHOT_AUTOPARAMS` | `"2.5,3.5"` | Kron factor and minimum radius for `MAG_AUTO` |
| `PHOT_PETROPARAMS` | `"2.0,3.5"` | Petrosian factor and minimum radius |
| `PHOT_AUTOAPERS` | `"0.0,0.0"` | Minimum apertures for `MAG_AUTO` (pixels) |
| `PHOT_FLUXFRAC` | `"0.5"` | Fraction(s) of total flux for `FLUX_RADIUS` |
| `SATUR_LEVEL` | `50000.0` | Saturation level (ADU) |
| `MAG_ZEROPOINT` | `0.0` | Photometric zero-point |
| `MAG_GAMMA` | `4.0` | Emulsion gamma (for `PHOTO` type) |
| `GAIN` | `0.0` | Detector gain (e⁻/ADU); 0 = ignore Poisson |
| `PIXEL_SCALE` | `1.0` | Pixel scale in arcsec (0 = read from WCS) |

### Star/galaxy separation

| Key | Default | Description |
|-----|---------|-------------|
| `SEEING_FWHM` | `1.2` | FWHM of stellar PSF (arcsec) for star-galaxy classifier |
| `STARNNW_NAME` | `"default.nnw"` | Neural network weights file |

### Background

| Key | Default | Description |
|-----|---------|-------------|
| `BACK_TYPE` | `"AUTO"` | Background type (`"AUTO"` or `"MANUAL"`) |
| `BACK_VALUE` | `0.0` | Manual background value |
| `BACK_SIZE` | `64` | Background mesh size (pixels) |
| `BACK_FILTERSIZE` | `3` | Background map filter size (meshes) |
| `BACKPHOTO_TYPE` | `"GLOBAL"` | `"GLOBAL"` or `"LOCAL"` photometric background |
| `BACKPHOTO_THICK` | `24` | Local background annulus thickness (pixels) |

### Weight map

| Key | Default | Description |
|-----|---------|-------------|
| `WEIGHT_TYPE` | `"NONE"` | `"NONE"`, `"BACKGROUND"`, `"MAP_RMS"`, `"MAP_VAR"`, `"MAP_WEIGHT"` |
| `WEIGHT_THRESH` | *(none)* | Bad-pixel threshold on weight map |
| `WEIGHT_GAIN` | `"Y"` | Use weight map for gain correction |

### Output

| Key | Default | Description |
|-----|---------|-------------|
| `CATALOG_TYPE` | `"ASCII_HEAD"` | `"ASCII_HEAD"`, `"FITS_LDAC"`, etc. |
| `VERBOSE_TYPE` | `"NORMAL"` | `"QUIET"`, `"NORMAL"`, `"FULL"` |
| `CHECKIMAGE_TYPE` | `"NONE"` | e.g. `"BACKGROUND"`, `"SEGMENTATION"` |
| `CHECKIMAGE_NAME` | `"check.fits"` | Filename for check image |

---

## SExtractor output parameters

Commonly used output columns for the `params` list.

### Position

| Column | Unit | Description |
|--------|------|-------------|
| `NUMBER` | — | Running object number |
| `X_IMAGE` | px | Object position along x |
| `Y_IMAGE` | px | Object position along y |
| `X_WORLD` | deg | Right ascension (requires WCS) |
| `Y_WORLD` | deg | Declination (requires WCS) |
| `ALPHA_J2000` | deg | Right ascension (J2000) |
| `DELTA_J2000` | deg | Declination (J2000) |
| `XMIN_IMAGE`, `XMAX_IMAGE` | px | Bounding box |
| `YMIN_IMAGE`, `YMAX_IMAGE` | px | Bounding box |
| `X2_IMAGE`, `Y2_IMAGE`, `XY_IMAGE` | px² | Second-order moments |

### Shape

| Column | Unit | Description |
|--------|------|-------------|
| `A_IMAGE`, `B_IMAGE` | px | Semi-major / semi-minor axis |
| `THETA_IMAGE` | deg | Position angle (CCW from x) |
| `ELLIPTICITY` | — | `1 - B_IMAGE/A_IMAGE` |
| `FWHM_IMAGE` | px | FWHM of PSF-fitted profile |
| `FWHM_WORLD` | arcsec | FWHM in world coordinates |
| `ISOAREA_IMAGE` | px² | Isophotal area |
| `CLASS_STAR` | — | Star-galaxy classifier output (0=galaxy, 1=star) |

### Photometry

| Column | Unit | Description |
|--------|------|-------------|
| `FLUX_AUTO` | counts | Kron auto-aperture flux |
| `FLUXERR_AUTO` | counts | Kron auto-aperture flux error |
| `MAG_AUTO` | mag | Kron auto-aperture magnitude |
| `MAGERR_AUTO` | mag | Kron auto-aperture magnitude error |
| `FLUX_APER` | counts | Fixed circular aperture flux |
| `FLUXERR_APER` | counts | Fixed aperture flux error |
| `MAG_APER` | mag | Fixed aperture magnitude |
| `MAGERR_APER` | mag | Fixed aperture magnitude error |
| `FLUX_ISO` | counts | Isophotal flux |
| `MAG_ISO` | mag | Isophotal magnitude |
| `FLUX_PETRO` | counts | Petrosian flux |
| `MAG_PETRO` | mag | Petrosian magnitude |
| `FLUX_RADIUS` | px | Fraction-of-light radius |
| `KRON_RADIUS` | px | Kron radius used for `FLUX_AUTO` |

### Quality

| Column | Unit | Description |
|--------|------|-------------|
| `FLAGS` | — | Extraction flags (bitmask; see below) |
| `IMAFLAGS_ISO` | — | Flag from external flag image |
| `SNR_WIN` | — | Signal-to-noise ratio in windowed aperture |

**FLAGS bitmask:**

| Bit | Value | Meaning |
|-----|-------|---------|
| 0 | 1 | Object has neighbours; bright contamination |
| 1 | 2 | Object was blended with another |
| 2 | 4 | At least one pixel is saturated |
| 3 | 8 | Object truncated by image boundary |
| 4 | 16 | Aperture data incomplete or corrupted |
| 5 | 32 | Isophotal data incomplete or corrupted |
| 6 | 64 | Memory overflow during deblending |
| 7 | 128 | Memory overflow during extraction |
