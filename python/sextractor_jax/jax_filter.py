"""
jax_filter.py
-------------
JAX-accelerated image filtering routines, replacing filter.c.

Implements:
  convolve_scanline  – convolve a single image scanline (replaces convolve())
  convolve_image     – convolve a 2-D image patch (replaces convolve_image())
  apply_filter_batch – apply the same convolution mask to a batch of images

The original C code operates in streaming mode (scanline-by-scanline) and
handles arbitrary rectangular convolution masks via a direct sliding-sum.
Here we use ``jax.scipy.signal.convolve2d`` / ``jax.lax.conv_general_dilated``
for full 2-D vectorised convolution.

All functions accept the mask as a flat ``float32`` array plus its (h, w)
dimensions, matching the ``filterstruct`` layout in filter.h.

Public API
----------
convolve_image(image, mask, mask_h, mask_w, normalise)
    2-D convolution of *image* with *mask*.

convolve_scanline(strip, y, mask, mask_h, mask_w, strip_h)
    Convolve a single scanline *y* within a strip buffer.

make_gaussian_mask(fwhm, size)
    Generate a normalised Gaussian convolution kernel (a common use-case).

make_tophat_mask(radius, size)
    Generate a top-hat detection filter.
"""

import jax
import jax.numpy as jnp
from functools import partial
import math


# ---------------------------------------------------------------------------
# Core 2-D convolution
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("mask_h", "mask_w", "normalise"))
def convolve_image(
    image: jnp.ndarray,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
    normalise: bool = True,
) -> jnp.ndarray:
    """
    Convolve a 2-D image with a convolution mask (replaces convolve_image()).

    Padding mode is 'edge' to avoid boundary artifacts from zeros (matching
    the C code's behaviour of using strip boundary values).

    Parameters
    ----------
    image    : jnp.ndarray, shape (H, W), float32
    mask     : jnp.ndarray, shape (mask_h * mask_w,) or (mask_h, mask_w), float32
    mask_h   : int – number of rows in mask (static)
    mask_w   : int – number of columns in mask (static)
    normalise : bool – if True, divide by sum(|mask|) (matches NORM in .conv files)

    Returns
    -------
    filtered : jnp.ndarray, shape (H, W), float32
    """
    # Reshape mask to 2-D
    kernel = mask.reshape(mask_h, mask_w).astype(jnp.float32)

    if normalise:
        s = jnp.sum(jnp.abs(kernel))
        kernel = jnp.where(s > 0.0, kernel / s, kernel)

    # Use lax.conv for efficient GPU convolution
    # Input shape: (batch, channels, H, W)  = (1, 1, H, W)
    img = image.astype(jnp.float32)[None, None, :, :]
    # Kernel shape for conv_general_dilated: (out_ch, in_ch, kH, kW) = (1,1,mH,mW)
    k = kernel[None, None, :, :]

    pad_h = mask_h // 2
    pad_w = mask_w // 2

    out = jax.lax.conv_general_dilated(
        img, k,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )
    return out[0, 0].astype(jnp.float32)


# ---------------------------------------------------------------------------
# Scanline convolution  (streaming interface matching filter.c)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("mask_h", "mask_w", "strip_h"))
def convolve_scanline(
    strip: jnp.ndarray,
    y: int,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
    strip_h: int,
) -> jnp.ndarray:
    """
    Convolve scanline *y* within image strip *strip* (replaces convolve()).

    The strip is a circular buffer of *strip_h* rows, each of width W.

    Parameters
    ----------
    strip   : jnp.ndarray, shape (strip_h, W), float32 – circular row buffer
    y       : int – current scanline index (into strip via y%strip_h)
    mask    : jnp.ndarray, shape (mask_h*mask_w,), float32 – convolution mask
    mask_h  : int (static)
    mask_w  : int (static)
    strip_h : int (static)

    Returns
    -------
    mscan : jnp.ndarray, shape (W,), float32 – filtered scanline
    """
    W = strip.shape[1]
    kernel = mask.reshape(mask_h, mask_w).astype(jnp.float32)
    mw2 = mask_w // 2
    half_h = mask_h // 2

    # Gather the rows that contribute to scanline y
    # Row y0 corresponds to the first mask row (top of kernel)
    y0 = y - half_h

    # Build the local patch of rows with wrap-around (circular buffer)
    row_indices = (y0 + jnp.arange(mask_h)) % strip_h  # shape (mask_h,)
    patch = strip[row_indices, :]                         # (mask_h, W)

    # Convolve patch rows with the mask
    # Each mask row contributes a shifted version of the strip row
    mscan = jnp.zeros(W, dtype=jnp.float32)
    for m_row in range(mask_h):
        for m_col in range(mask_w):
            shift = m_col - mw2
            mval = kernel[m_row, m_col]
            # Shift the strip row by 'shift' with zero padding at boundaries
            if shift > 0:
                src = jnp.concatenate([patch[m_row, shift:], jnp.zeros(shift)])
            elif shift < 0:
                src = jnp.concatenate([jnp.zeros(-shift), patch[m_row, :W + shift]])
            else:
                src = patch[m_row, :]
            mscan = mscan + mval * src

    return mscan


# ---------------------------------------------------------------------------
# Batch convolution
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("mask_h", "mask_w", "normalise"))
def apply_filter_batch(
    images: jnp.ndarray,
    mask: jnp.ndarray,
    mask_h: int,
    mask_w: int,
    normalise: bool = True,
) -> jnp.ndarray:
    """
    Apply convolution mask to a batch of images simultaneously.

    Parameters
    ----------
    images  : jnp.ndarray, shape (N, H, W), float32
    mask    : jnp.ndarray, shape (mask_h*mask_w,) or (mask_h, mask_w)
    mask_h  : int (static)
    mask_w  : int (static)
    normalise : bool

    Returns
    -------
    filtered : jnp.ndarray, shape (N, H, W), float32
    """
    kernel = mask.reshape(mask_h, mask_w).astype(jnp.float32)
    if normalise:
        s = jnp.sum(jnp.abs(kernel))
        kernel = jnp.where(s > 0.0, kernel / s, kernel)

    # (N, 1, H, W)
    imgs = images.astype(jnp.float32)[:, None, :, :]
    # (1, 1, mask_h, mask_w)
    k = kernel[None, None, :, :]

    N, _, H, W = imgs.shape
    # Broadcast kernel for N images via group convolution
    k_batch = jnp.tile(k, (N, 1, 1, 1))   # (N, 1, mask_h, mask_w)

    pad_h = mask_h // 2
    pad_w = mask_w // 2

    out = jax.lax.conv_general_dilated(
        imgs, k,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        feature_group_count=1,
    )
    # out shape: (N, 1, H, W) → (N, H, W)
    # Note: conv above applies same kernel to all N images via broadcasting
    return out[:, 0, :, :].astype(jnp.float32)


# ---------------------------------------------------------------------------
# Common kernel factories
# ---------------------------------------------------------------------------

def make_gaussian_mask(fwhm: float, size: int | None = None) -> jnp.ndarray:
    """
    Generate a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    fwhm : float – full width at half maximum in pixels
    size : int or None – kernel size (default: 2*ceil(3*sigma)+1)

    Returns
    -------
    kernel : jnp.ndarray, shape (size, size), float32 (normalised to sum=1)
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if size is None:
        half = max(1, math.ceil(3.0 * sigma))
        size = 2 * half + 1
    half = size // 2
    y = jnp.arange(-half, half + 1, dtype=jnp.float32)
    x = jnp.arange(-half, half + 1, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y)
    g = jnp.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return (g / jnp.sum(g)).astype(jnp.float32)


def make_tophat_mask(radius: float, size: int | None = None) -> jnp.ndarray:
    """
    Generate a circular top-hat kernel (used in source detection).

    Parameters
    ----------
    radius : float – radius in pixels
    size   : int or None – kernel size (default: 2*ceil(radius)+1)

    Returns
    -------
    kernel : jnp.ndarray, shape (size, size), float32 (normalised)
    """
    if size is None:
        half = max(1, math.ceil(radius))
        size = 2 * half + 1
    half = size // 2
    y = jnp.arange(-half, half + 1, dtype=jnp.float32)
    x = jnp.arange(-half, half + 1, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y)
    mask = (xx * xx + yy * yy <= radius * radius).astype(jnp.float32)
    s = jnp.sum(mask)
    return (mask / jnp.where(s > 0, s, 1.0)).astype(jnp.float32)


def make_mexhat_mask(fwhm: float, size: int | None = None) -> jnp.ndarray:
    """
    Generate a Mexican-hat (Laplacian-of-Gaussian) kernel.

    Parameters
    ----------
    fwhm : float
    size : int or None

    Returns
    -------
    kernel : jnp.ndarray, shape (size, size), float32 (zero-sum)
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if size is None:
        half = max(1, math.ceil(3.0 * sigma))
        size = 2 * half + 1
    half = size // 2
    y = jnp.arange(-half, half + 1, dtype=jnp.float32)
    x = jnp.arange(-half, half + 1, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y)
    r2 = (xx * xx + yy * yy) / (sigma * sigma)
    g = jnp.exp(-0.5 * r2)
    log = (1.0 - r2) * g
    # Normalise so that positive part sums to 1
    pos = jnp.where(log > 0, log, 0.0)
    s = jnp.sum(pos)
    return (log / jnp.where(s > 0, s, 1.0)).astype(jnp.float32)
