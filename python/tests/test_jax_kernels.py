"""
tests/test_jax_kernels.py
--------------------------
Smoke tests for the sextractor_jax package.

Verifies that all public functions run without error and produce
numerically sensible results.  Does not require a GPU.
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow running from the python/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from sextractor_jax import (
    # fft
    fft_rtf, fft_conv, fft_conv_full, fft_conv_batch,
    # models
    model_back, model_dirac, model_sersic, model_devaucouleurs,
    model_exponential, model_arms, model_bar, model_inring, model_outring,
    compute_residuals,
    # photometry
    compute_aperflux, flux_to_mag, flux_to_magerr,
    # background
    backstat, backguess, make_backmap, interp_backmap, subtract_background,
    # optimiser
    levmar_fit, levmar_step, PARFIT_LINBOUND, PARFIT_UNBOUND,
    # filter
    convolve_image, make_gaussian_mask, make_tophat_mask,
)

# Use CPU backend for CI reproducibility
jax.config.update("jax_platform_name", "cpu")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_image(rng):
    return jnp.array(rng.normal(100.0, 10.0, (64, 64)).astype(np.float32))


# ---------------------------------------------------------------------------
# FFT tests
# ---------------------------------------------------------------------------

class TestFFT:
    def test_fft_rtf_shape(self, small_image):
        fdata = fft_rtf(small_image)
        H, W = small_image.shape
        assert fdata.shape == (H, W // 2 + 1)
        assert jnp.iscomplexobj(fdata)

    def test_fft_conv_identity(self, small_image):
        """Convolving with delta function returns the original image."""
        H, W = small_image.shape
        delta = jnp.zeros((H, W), dtype=jnp.float32)
        delta = delta.at[H // 2, W // 2].set(1.0)
        delta_ft = fft_rtf(delta)
        result = fft_conv(small_image, delta_ft, output_shape=(H, W))
        # Should be close to original (up to float32 precision)
        np.testing.assert_allclose(
            np.array(result), np.array(small_image), rtol=1e-4, atol=1e-2)

    def test_fft_conv_full(self, small_image):
        H, W = small_image.shape
        kernel = make_gaussian_mask(3.0, size=9)
        kernel_padded = jnp.zeros((H, W), dtype=jnp.float32)
        kh, kw = kernel.shape
        kernel_padded = kernel_padded.at[:kh, :kw].set(kernel)
        result = fft_conv_full(small_image, kernel_padded)
        assert result.shape == (H, W)
        assert not jnp.any(jnp.isnan(result))

    def test_fft_conv_batch(self, small_image):
        H, W = small_image.shape
        batch = jnp.stack([small_image] * 4)
        delta = jnp.zeros((H, W), dtype=jnp.float32).at[H//2, W//2].set(1.0)
        delta_ft = fft_rtf(delta)
        results = fft_conv_batch(batch, delta_ft, output_shape=(H, W))
        assert results.shape == (4, H, W)


# ---------------------------------------------------------------------------
# Galaxy model tests
# ---------------------------------------------------------------------------

class TestModels:
    SIZE = 64

    def test_model_back(self):
        pix = model_back(5.0, self.SIZE, self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        np.testing.assert_allclose(float(pix.mean()), 5.0, rtol=1e-5)

    def test_model_dirac(self):
        pix = model_dirac(self.SIZE, self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        assert float(pix.sum()) == pytest.approx(1.0)
        assert float(pix[self.SIZE // 2, self.SIZE // 2]) == pytest.approx(1.0)

    def test_model_sersic_positive(self):
        pix = model_sersic(
            scale=8.0, aspect=0.7, posangle_deg=30.0,
            sersic_n=1.5, pixstep=1.0,
            width=self.SIZE, height=self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        assert float(jnp.min(pix)) >= 0.0
        assert float(jnp.max(pix)) > 0.0

    def test_model_sersic_symmetry(self):
        """Sérsic profile with aspect=1, posangle=0 should be rotationally symmetric."""
        pix = model_sersic(
            scale=8.0, aspect=1.0, posangle_deg=0.0,
            sersic_n=2.0, pixstep=1.0,
            width=self.SIZE, height=self.SIZE)
        centre = self.SIZE // 2
        # Value at (cy, cx+r) should equal value at (cy+r, cx)
        r = 5
        np.testing.assert_allclose(
            float(pix[centre, centre + r]),
            float(pix[centre + r, centre]),
            rtol=0.05)

    def test_model_devaucouleurs(self):
        pix = model_devaucouleurs(
            scale=6.0, aspect=0.8, posangle_deg=0.0,
            pixstep=1.0, width=self.SIZE, height=self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        assert float(jnp.max(pix)) > 0.0

    def test_model_exponential(self):
        pix = model_exponential(
            scale=10.0, aspect=1.0, posangle_deg=0.0,
            pixstep=1.0, width=self.SIZE, height=self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        # Exponential profile should peak at centre
        centre = self.SIZE // 2
        assert float(pix[centre, centre]) == float(jnp.max(pix))

    def test_model_bar(self):
        from sextractor_jax.jax_models import _cd_matrix
        cd = _cd_matrix(10.0, 0.5, 0.0, 1.0)
        pix = model_bar(*cd, featstart=3.0, feataspect=0.3,
                        featposang_deg=0.0,
                        width=self.SIZE, height=self.SIZE)
        assert pix.shape == (self.SIZE, self.SIZE)
        assert not jnp.any(jnp.isnan(pix))

    def test_compute_residuals_chi2(self):
        lmodpix = jnp.ones(100, dtype=jnp.float32) * 2.0
        objpix = jnp.ones(100, dtype=jnp.float32)
        objweight = jnp.ones(100, dtype=jnp.float32)
        resi, chi2 = compute_residuals(lmodpix, objpix, objweight)
        # residual = (2-1)*1 = 1 for each of 100 pixels
        np.testing.assert_allclose(float(chi2), 100.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Photometry tests
# ---------------------------------------------------------------------------

class TestPhotometry:
    def test_aperflux_flat_field(self):
        """Flat image: aperture flux ≈ π*r² * (value - bkg)."""
        H, W = 64, 64
        value = 10.0
        bkg = 5.0
        raper = 8.0
        image = jnp.full((H, W), value, dtype=jnp.float32)
        flux, ferr, area = compute_aperflux(
            image, None,
            mx=W / 2.0, my=H / 2.0,
            raper=raper, backsig=1.0, gain=0.0, bkg=bkg)
        expected = math.pi * raper ** 2 * (value - bkg)
        assert abs(flux - expected) / expected < 0.05, \
            f"flux={flux:.2f} expected≈{expected:.2f}"

    def test_aperflux_zero_bkg(self):
        H, W = 32, 32
        image = jnp.zeros((H, W), dtype=jnp.float32)
        flux, ferr, area = compute_aperflux(
            image, None, mx=16.0, my=16.0,
            raper=5.0, backsig=1.0, gain=0.0, bkg=0.0)
        assert abs(flux) < 1.0

    def test_flux_to_mag(self):
        flux = jnp.array([1.0, 100.0, 1e4], dtype=jnp.float32)
        mag = flux_to_mag(flux, zeropoint=25.0)
        # mag = -2.5*log10(flux) + 25
        expected = -2.5 * np.log10([1, 100, 1e4]) + 25.0
        np.testing.assert_allclose(np.array(mag), expected, rtol=1e-5)

    def test_flux_to_magerr(self):
        flux = jnp.array([1000.0], dtype=jnp.float32)
        ferr = jnp.array([10.0], dtype=jnp.float32)
        magerr = flux_to_magerr(flux, ferr)
        expected = 1.086 * 10.0 / 1000.0
        np.testing.assert_allclose(float(magerr[0]), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Background tests
# ---------------------------------------------------------------------------

class TestBackground:
    def test_backstat_uniform(self):
        """Uniform image: all meshes should have mean ≈ value, sigma ≈ 0."""
        image = jnp.full((128, 128), 50.0, dtype=jnp.float32)
        meshes, grid = backstat(image, None, mesh_w=32, mesh_h=32)
        assert grid == (4, 4)
        for m in meshes:
            if m['mean'] > -1e28:
                assert abs(m['mean'] - 50.0) < 1.0

    def test_backstat_noisy(self):
        rng = np.random.default_rng(0)
        image = jnp.array(rng.normal(100.0, 5.0, (128, 128)).astype(np.float32))
        meshes, grid = backstat(image, None, mesh_w=32, mesh_h=32)
        for m in meshes:
            if m['mean'] > -1e28:
                assert 90.0 < m['mean'] < 110.0
                assert 1.0 < m['sigma'] < 15.0

    def test_make_backmap_shape(self):
        rng = np.random.default_rng(1)
        image = jnp.array(rng.normal(100.0, 5.0, (64, 64)).astype(np.float32))
        back_map, sigma_map, grid = make_backmap(image, None, 16, 16)
        assert back_map.shape == (4, 4)
        assert sigma_map.shape == (4, 4)

    def test_interp_backmap_shape(self):
        back_map = jnp.ones((4, 4), dtype=jnp.float32) * 100.0
        full = interp_backmap(back_map, 64, 64)
        assert full.shape == (64, 64)
        np.testing.assert_allclose(float(full.mean()), 100.0, rtol=0.01)

    def test_subtract_background(self):
        image = jnp.full((32, 32), 110.0, dtype=jnp.float32)
        back = jnp.full((32, 32), 100.0, dtype=jnp.float32)
        result = subtract_background(image, back)
        np.testing.assert_allclose(float(result.mean()), 10.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Optimiser tests
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_levmar_step_identity(self):
        """With μ→0, LM step should approach least-squares solution."""
        J = jnp.eye(3, dtype=jnp.float64)
        r = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        dp = levmar_step(J, r, mu=1e-10)
        np.testing.assert_allclose(np.array(dp), [1.0, 2.0, 3.0], rtol=1e-4)

    def test_levmar_fit_quadratic(self):
        """Minimise (x - 3)² + (y - 5)²  → minimum at (3, 5)."""
        def residuals(p, *_):
            return jnp.array([p[0] - 3.0, p[1] - 5.0], dtype=jnp.float32)

        p0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
        result = levmar_fit(residuals, p0, max_iter=50)
        params = np.array(result['params'])
        np.testing.assert_allclose(params, [3.0, 5.0], atol=1e-3)
        assert result['chi2'] < 1e-6

    def test_levmar_fit_bounded(self):
        """With bounds, solution should be clipped."""
        def residuals(p, *_):
            return jnp.array([p[0] - 10.0], dtype=jnp.float32)  # wants x=10

        p0 = jnp.array([0.5], dtype=jnp.float32)
        lo = jnp.array([0.0])
        hi = jnp.array([2.0])
        ftypes = [PARFIT_LINBOUND]
        result = levmar_fit(residuals, p0, lo=lo, hi=hi, ftypes=ftypes,
                            max_iter=100)
        # Solution should be clamped near hi=2.0
        assert float(result['params'][0]) < 2.0 + 0.1


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestFilter:
    def test_gaussian_mask_sum(self):
        mask = make_gaussian_mask(3.0, size=11)
        np.testing.assert_allclose(float(jnp.sum(mask)), 1.0, rtol=1e-5)

    def test_tophat_mask_sum(self):
        mask = make_tophat_mask(3.0)
        np.testing.assert_allclose(float(jnp.sum(mask)), 1.0, rtol=1e-5)

    def test_convolve_image_preserves_shape(self, small_image):
        mask = make_gaussian_mask(2.0, size=5)
        H, W = small_image.shape
        mh, mw = mask.shape
        result = convolve_image(small_image, mask, mh, mw, normalise=False)
        assert result.shape == (H, W)

    def test_convolve_image_no_nan(self, small_image):
        mask = make_gaussian_mask(2.0, size=5)
        mh, mw = mask.shape
        result = convolve_image(small_image, mask, mh, mw)
        assert not jnp.any(jnp.isnan(result))

    def test_convolve_image_delta(self):
        """Convolving with a delta kernel should return the original image."""
        H, W = 32, 32
        image = jnp.arange(H * W, dtype=jnp.float32).reshape(H, W)
        delta = jnp.zeros((1, 1), dtype=jnp.float32).at[0, 0].set(1.0)
        result = convolve_image(image, delta, 1, 1, normalise=False)
        np.testing.assert_allclose(
            np.array(result), np.array(image), rtol=1e-5)

    def test_convolve_smoothing(self, small_image):
        """Gaussian smoothing should reduce image variance."""
        mask = make_gaussian_mask(5.0, size=15)
        mh, mw = mask.shape
        result = convolve_image(small_image, mask, mh, mw)
        assert float(jnp.std(result)) < float(jnp.std(small_image))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
