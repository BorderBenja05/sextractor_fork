"""
jax_fft.py
----------
JAX-accelerated FFT convolution routines replacing fft.c.

The original fft.c uses FFTW3 single-precision real-to-complex transforms.
Here we replicate the same semantics with jax.numpy.fft, which also runs on
GPU/TPU transparently.

Public API
----------
fft_conv(data, kernel_ft, shape)
    Convolve *data* with a pre-computed Fourier-domain kernel.

fft_rtf(data)
    Compute and return the (compressed) real-to-Fourier transform of *data*.

fft_conv_full(data, kernel)
    High-level convenience: convolve two real 2-D arrays via FFT.

All functions are JAX-jit-compilable and differentiable (where meaningful).
"""

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

@jax.jit
def _rfft2(arr: jnp.ndarray) -> jnp.ndarray:
    """Forward 2-D real FFT, row-major (same axis ordering as FFTW r2c)."""
    return jnp.fft.rfft2(arr)


@jax.jit
def _irfft2(arr: jnp.ndarray, s=None) -> jnp.ndarray:
    """Inverse 2-D real FFT."""
    return jnp.fft.irfft2(arr, s=s)


# ---------------------------------------------------------------------------
# fft_rtf  –  compute Fourier transform of a real 2-D array
# ---------------------------------------------------------------------------

@jax.jit
def fft_rtf(data: jnp.ndarray) -> jnp.ndarray:
    """
    Equivalent of fft_rtf() in fft.c.

    Compute the 2-D real-to-complex FFT of *data* and return the compressed
    (non-redundant) complex array with shape ``(*data.shape[:-1],
    data.shape[-1]//2 + 1)``.

    Parameters
    ----------
    data : jnp.ndarray, shape (H, W), float32
        Input image (may be corrupted on return by original C; here we are
        non-destructive).

    Returns
    -------
    fdata : jnp.ndarray, shape (H, W//2+1), complex64
    """
    return _rfft2(data.astype(jnp.float32))


# ---------------------------------------------------------------------------
# fft_conv  –  convolve image with pre-computed Fourier kernel
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("output_shape",))
def fft_conv(data: jnp.ndarray,
             kernel_ft: jnp.ndarray,
             output_shape=None) -> jnp.ndarray:
    """
    Equivalent of fft_conv() in fft.c.

    Convolve *data* (real 2-D) with a PSF whose Fourier transform
    *kernel_ft* has already been computed via :func:`fft_rtf`.

    The original C code:
        1. Forward FFT data  → fdata1
        2. Fourier product   fdata1 *= fdata2 / npix   (complex multiply)
        3. Inverse FFT       → data  (in-place)

    Parameters
    ----------
    data : jnp.ndarray, shape (H, W), float32
        Input image (model pixels).
    kernel_ft : jnp.ndarray, shape (H, W//2+1), complex64
        Pre-computed FFT of the PSF (from :func:`fft_rtf`).
    output_shape : tuple (H, W) or None
        Used to set the irfft2 output size; defaults to *data.shape*.

    Returns
    -------
    result : jnp.ndarray, shape (H, W), float32
        Convolved image.
    """
    if output_shape is None:
        output_shape = data.shape

    H, W = output_shape
    npix = H * W

    # Forward real-to-complex FFT
    fdata = _rfft2(data.astype(jnp.float32))

    # Complex multiply with normalisation (matches C: fac = 1/npix)
    fdata = fdata * kernel_ft / jnp.float32(npix)

    # Inverse FFT back to real space
    result = _irfft2(fdata, s=(H, W))
    return result.astype(jnp.float32)


# ---------------------------------------------------------------------------
# fft_conv_full  –  convolve two real arrays (computes kernel FFT on the fly)
# ---------------------------------------------------------------------------

@jax.jit
def fft_conv_full(data: jnp.ndarray,
                  kernel: jnp.ndarray) -> jnp.ndarray:
    """
    Convolve *data* with *kernel* using FFT.

    Both arrays must have the same shape.  This is a convenience wrapper for
    use when the kernel FFT has not been pre-computed.

    Parameters
    ----------
    data   : jnp.ndarray, shape (H, W), float32
    kernel : jnp.ndarray, shape (H, W), float32

    Returns
    -------
    result : jnp.ndarray, shape (H, W), float32
    """
    kernel_ft = fft_rtf(kernel)
    return fft_conv(data, kernel_ft, output_shape=data.shape)


# ---------------------------------------------------------------------------
# Batched version – convolve N images with the same kernel (vmap)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("output_shape",))
def fft_conv_batch(images: jnp.ndarray,
                   kernel_ft: jnp.ndarray,
                   output_shape=None) -> jnp.ndarray:
    """
    Convolve a batch of images with the same pre-computed kernel FFT.

    Parameters
    ----------
    images    : jnp.ndarray, shape (N, H, W), float32
    kernel_ft : jnp.ndarray, shape (H, W//2+1), complex64
    output_shape : (H, W) or None

    Returns
    -------
    results : jnp.ndarray, shape (N, H, W), float32
    """
    def _single(img):
        return fft_conv(img, kernel_ft, output_shape=output_shape)

    return jax.vmap(_single)(images)
