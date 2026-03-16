"""
sextractor_jax
==============
JAX-accelerated computational kernels for SExtractor.

This package provides drop-in JAX replacements for the most
computationally intensive C modules in SExtractor 2.29.

Modules
-------
jax_fft      – FFT convolution (replaces fft.c / FFTW3)
jax_models   – Galaxy profile rendering (replaces profit.c prof_add)
jax_photom   – Aperture photometry (replaces photom.c)
jax_back     – Background estimation (replaces back.c)
jax_optimize – Levenberg-Marquardt optimiser (replaces levmar/)
jax_filter   – Image convolution filters (replaces filter.c)

Quick-start
-----------
>>> import jax.numpy as jnp
>>> from sextractor_jax import fft_conv, fft_rtf
>>> from sextractor_jax import model_sersic, compute_residuals
>>> from sextractor_jax import compute_aperflux
>>> from sextractor_jax import make_backmap, interp_backmap
>>> from sextractor_jax import levmar_fit
>>> from sextractor_jax import convolve_image, make_gaussian_mask

All functions are ``jax.jit``-compatible and run transparently on CPU,
GPU (CUDA/ROCm), and TPU via JAX's XLA backend.
"""

# FFT convolution
from .jax_fft import (
    fft_rtf,
    fft_conv,
    fft_conv_full,
    fft_conv_batch,
)

# Galaxy profile models
from .jax_models import (
    model_back,
    model_dirac,
    model_sersic,
    model_devaucouleurs,
    model_exponential,
    model_arms,
    model_bar,
    model_inring,
    model_outring,
    render_model,
    compute_residuals,
    compute_spiral_index,
)

# Photometry
from .jax_photom import (
    compute_aperflux,
    compute_kron_radius,
    compute_autoflux,
    compute_petroflux,
    flux_to_mag,
    flux_to_magerr,
)

# Background estimation
from .jax_back import (
    backstat,
    backhisto,
    backguess,
    make_backmap,
    interp_backmap,
    subtract_background,
    make_backmap_batch,
)

# Levenberg-Marquardt optimiser
from .jax_optimize import (
    levmar_fit,
    levmar_fit_batch,
    levmar_step,
    bounded_to_unbounded,
    unbounded_to_bounded,
    propagate_covar,
    PARFIT_FIXED,
    PARFIT_UNBOUND,
    PARFIT_LINBOUND,
    PARFIT_LOGBOUND,
)

# Image filtering
from .jax_filter import (
    convolve_image,
    convolve_scanline,
    apply_filter_batch,
    make_gaussian_mask,
    make_tophat_mask,
    make_mexhat_mask,
)

# In-memory SExtractor interface
from .sextractor import (
    SExtractor,
    run as sex_run,
    memory_backend,
)

__version__ = "0.1.0"
__all__ = [
    # fft
    "fft_rtf", "fft_conv", "fft_conv_full", "fft_conv_batch",
    # models
    "model_back", "model_dirac", "model_sersic", "model_devaucouleurs",
    "model_exponential", "model_arms", "model_bar", "model_inring",
    "model_outring", "render_model", "compute_residuals",
    "compute_spiral_index",
    # photometry
    "compute_aperflux", "compute_kron_radius", "compute_autoflux",
    "compute_petroflux", "flux_to_mag", "flux_to_magerr",
    # background
    "backstat", "backhisto", "backguess", "make_backmap",
    "interp_backmap", "subtract_background", "make_backmap_batch",
    # optimiser
    "levmar_fit", "levmar_fit_batch", "levmar_step",
    "bounded_to_unbounded", "unbounded_to_bounded", "propagate_covar",
    "PARFIT_FIXED", "PARFIT_UNBOUND", "PARFIT_LINBOUND", "PARFIT_LOGBOUND",
    # filter
    "convolve_image", "convolve_scanline", "apply_filter_batch",
    "make_gaussian_mask", "make_tophat_mask", "make_mexhat_mask",
    # in-memory SExtractor interface
    "SExtractor", "sex_run", "memory_backend",
]
