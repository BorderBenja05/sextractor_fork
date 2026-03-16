"""
tests/test_sextractor_interface.py
-----------------------------------
Tests for the in-memory SExtractor Python interface (sextractor.py).

These tests focus on the memory-buffer machinery — they do NOT require
a real SExtractor binary to be installed.  Tests that need the binary are
marked with ``@pytest.mark.integration`` and skipped by default.
"""

import io
import os
import sys
import tempfile
import unittest.mock as mock

import numpy as np
import pytest

# Allow running from python/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import astropy.io.fits as fits
from astropy.table import Table

from sextractor_jax.sextractor import (
    SExtractor,
    run,
    memory_backend,
    _hdu_to_bytes,
    _memfd_available,
    _memory_input_file,
    _memory_output_file,
    _parse_ascii_head,
    _parse_fits_ldac,
    DEFAULT_PARAMS,
    DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_hdu():
    """A 64×64 float32 image with a single bright source at the centre."""
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 10.0, (64, 64)).astype(np.float32)
    # Plant a faint point source
    cy, cx = 32, 32
    y, x = np.mgrid[:64, :64]
    data += 500.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0 ** 2))
    return fits.PrimaryHDU(data.astype(np.float32))


@pytest.fixture
def simple_hdulist(simple_hdu):
    return fits.HDUList([simple_hdu])


# ---------------------------------------------------------------------------
# FITS serialisation
# ---------------------------------------------------------------------------

class TestHDUToBytes:
    def test_primary_hdu(self, simple_hdu):
        raw = _hdu_to_bytes(simple_hdu)
        assert isinstance(raw, bytes)
        assert raw[:6] == b"SIMPLE"  # FITS magic

    def test_hdulist(self, simple_hdulist):
        raw = _hdu_to_bytes(simple_hdulist)
        assert raw[:6] == b"SIMPLE"

    def test_numpy_array(self):
        arr = np.ones((32, 32), dtype=np.float32)
        raw = _hdu_to_bytes(arr)
        assert raw[:6] == b"SIMPLE"
        # Deserialise and check shape
        buf = io.BytesIO(raw)
        with fits.open(buf) as hdl:
            assert hdl[0].data.shape == (32, 32)

    def test_roundtrip_values(self, simple_hdu):
        raw = _hdu_to_bytes(simple_hdu)
        buf = io.BytesIO(raw)
        with fits.open(buf) as hdl:
            np.testing.assert_array_equal(
                hdl[0].data, simple_hdu.data)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            _hdu_to_bytes("not_an_hdu")


# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------

class TestMemoryBackend:
    def test_memory_backend_returns_string(self):
        s = memory_backend()
        assert isinstance(s, str)
        assert len(s) > 0

    @pytest.mark.skipif(sys.platform != "linux", reason="memfd is Linux-only")
    def test_memfd_available_linux(self):
        # On a modern Linux we expect memfd_create to be available
        result = _memfd_available()
        assert isinstance(result, bool)


class TestMemoryInputFile:
    def test_file_is_readable(self, simple_hdu):
        data = _hdu_to_bytes(simple_hdu)
        with _memory_input_file(data) as (path, fd):
            assert os.path.exists(path)
            with open(path, "rb") as f:
                content = f.read()
            assert content == data

    def test_file_supports_seek(self, simple_hdu):
        """The path must resolve to a seekable file (fseek support)."""
        data = _hdu_to_bytes(simple_hdu)
        with _memory_input_file(data) as (path, fd):
            with open(path, "rb") as f:
                f.seek(0, 2)                    # seek to end
                size = f.tell()
                f.seek(0)                       # back to beginning
                first_bytes = f.read(6)
            assert size == len(data)
            assert first_bytes == b"SIMPLE"

    def test_fits_readable_via_path(self, simple_hdu):
        """Astropy (and therefore CFITSIO) can open the path and read the HDU."""
        data = _hdu_to_bytes(simple_hdu)
        with _memory_input_file(data) as (path, fd):
            with fits.open(path) as hdl:
                np.testing.assert_array_equal(
                    hdl[0].data, simple_hdu.data)

    def test_cleanup_after_context(self):
        """The temporary resource should be cleaned up on exit."""
        data = b"X" * 1024
        with _memory_input_file(data) as (path, fd):
            # Determine if it's a real filesystem path (not memfd)
            is_filesystem = not path.startswith("/proc/")
        # After context: real-file paths should no longer exist
        if is_filesystem:
            assert not os.path.exists(path)


class TestMemoryOutputFile:
    def test_write_then_read(self):
        test_data = b"# SExtractor catalog\n1 2 3\n"
        with _memory_output_file() as (path, read_fn):
            with open(path, "wb") as f:
                f.write(test_data)
            result = read_fn()
        assert result == test_data

    def test_empty_file(self):
        with _memory_output_file() as (path, read_fn):
            # Don't write anything
            result = read_fn()
        assert result == b""

    def test_large_output(self):
        large = b"A" * (5 * 1024 * 1024)   # 5 MB
        with _memory_output_file() as (path, read_fn):
            with open(path, "wb") as f:
                f.write(large)
            result = read_fn()
        assert result == large


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------

_ASCII_HEAD_SAMPLE = b"""\
# 1 NUMBER Running object number
# 2 X_IMAGE Object position along x [pixel]
# 3 Y_IMAGE Object position along y [pixel]
# 4 FLUX_AUTO Flux within a Kron-like elliptical aperture [count]
# 5 FLAGS Extraction flags
1 32.1 32.3 10234.5 0
2 10.0 20.0 543.2 0
"""


class TestCatalogParsing:
    def test_parse_ascii_head(self):
        tbl = _parse_ascii_head(_ASCII_HEAD_SAMPLE)
        assert isinstance(tbl, Table)
        assert "X_IMAGE" in tbl.colnames
        assert "FLUX_AUTO" in tbl.colnames
        assert len(tbl) == 2
        assert float(tbl["X_IMAGE"][0]) == pytest.approx(32.1, abs=0.01)

    def test_parse_fits_ldac(self):
        # Build a minimal FITS_LDAC in memory
        col_num = fits.Column(name="NUMBER", format="J", array=np.array([1, 2]))
        col_x = fits.Column(name="X_IMAGE", format="E", array=np.array([32.1, 10.0]))
        col_y = fits.Column(name="Y_IMAGE", format="E", array=np.array([32.3, 20.0]))
        tbl_hdu = fits.BinTableHDU.from_columns([col_num, col_x, col_y])
        tbl_hdu.name = "LDAC_OBJECTS"

        primary = fits.PrimaryHDU()
        ldac_imhead = fits.BinTableHDU(name="LDAC_IMHEAD")

        hdul = fits.HDUList([primary, ldac_imhead, tbl_hdu])
        buf = io.BytesIO()
        hdul.writeto(buf)
        raw = buf.getvalue()

        tbl = _parse_fits_ldac(raw)
        assert isinstance(tbl, Table)
        assert "X_IMAGE" in tbl.colnames
        assert len(tbl) == 2


# ---------------------------------------------------------------------------
# SExtractor command building
# ---------------------------------------------------------------------------

class TestCommandBuilding:
    def setup_method(self):
        self.sx = SExtractor(config={"DETECT_THRESH": 2.0})

    def test_command_starts_with_binary(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", None,
            "/proc/123/fd/4", "/tmp/sex.param",
            self.sx.config,
        )
        assert cmd[0] == "sex"
        assert cmd[1] == "/proc/123/fd/3"

    def test_command_contains_catalog_name(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", None,
            "/tmp/sex.cat", "/tmp/sex.param",
            self.sx.config,
        )
        assert "-CATALOG_NAME" in cmd
        idx = cmd.index("-CATALOG_NAME")
        assert cmd[idx + 1] == "/tmp/sex.cat"

    def test_command_contains_params_file(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", None,
            "/tmp/sex.cat", "/tmp/sex.param",
            self.sx.config,
        )
        assert "-PARAMETERS_NAME" in cmd

    def test_command_includes_weight_image(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", "/proc/123/fd/5",
            "/tmp/sex.cat", "/tmp/sex.param",
            self.sx.config,
        )
        assert "-WEIGHT_IMAGE" in cmd
        idx = cmd.index("-WEIGHT_IMAGE")
        assert cmd[idx + 1] == "/proc/123/fd/5"

    def test_command_omits_weight_if_none(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", None,
            "/tmp/sex.cat", "/tmp/sex.param",
            self.sx.config,
        )
        assert "-WEIGHT_IMAGE" not in cmd

    def test_config_overrides_appear(self):
        cmd = self.sx._build_command(
            "/proc/123/fd/3", None,
            "/tmp/sex.cat", "/tmp/sex.param",
            {"DETECT_THRESH": 3.5, "VERBOSE_TYPE": "QUIET",
             "CATALOG_TYPE": "ASCII_HEAD"},
        )
        assert "-DETECT_THRESH" in cmd
        idx = cmd.index("-DETECT_THRESH")
        assert float(cmd[idx + 1]) == pytest.approx(3.5)

    def test_extra_args_appended(self):
        sx = SExtractor(extra_args=["-CHECKIMAGE_TYPE", "BACKGROUND"])
        cmd = sx._build_command(
            "img.fits", None, "cat.fits", "p.param", sx.config)
        assert "-CHECKIMAGE_TYPE" in cmd


# ---------------------------------------------------------------------------
# SExtractor.run() with mocked subprocess
# ---------------------------------------------------------------------------

class TestSExtractorRun:
    """Tests that mock the subprocess to avoid needing a real SExtractor binary."""

    def _make_catalog_bytes(self, n=5) -> bytes:
        """Build a minimal ASCII_HEAD catalog."""
        lines = [
            b"# 1 NUMBER Running object number",
            b"# 2 X_IMAGE Object position along x [pixel]",
            b"# 3 Y_IMAGE Object position along y [pixel]",
            b"# 4 FLUX_AUTO Flux [count]",
            b"# 5 FLAGS Flags",
        ]
        for i in range(1, n + 1):
            lines.append(f"{i} {10.0*i:.2f} {20.0*i:.2f} {1000.0*i:.1f} 0".encode())
        return b"\n".join(lines) + b"\n"

    def _run_with_mock(self, hdu, n_sources=5, returncode=0, config=None):
        """Run sx.run() with the subprocess mocked out."""
        sx = SExtractor(config=config or {})
        cat_bytes = self._make_catalog_bytes(n_sources)

        original_run = sx._run_subprocess

        def fake_subprocess(cmd):
            # Find -CATALOG_NAME and write our fake catalog there
            idx = cmd.index("-CATALOG_NAME")
            cat_path = cmd[idx + 1]
            with open(cat_path, "wb") as f:
                f.write(cat_bytes)
            result = mock.MagicMock()
            result.returncode = returncode
            result.stderr = b""
            return result

        sx._run_subprocess = fake_subprocess
        return sx.run(hdu)

    def test_returns_table(self, simple_hdu):
        tbl = self._run_with_mock(simple_hdu)
        assert isinstance(tbl, Table)

    def test_expected_columns(self, simple_hdu):
        tbl = self._run_with_mock(simple_hdu)
        assert "X_IMAGE" in tbl.colnames
        assert "Y_IMAGE" in tbl.colnames
        assert "FLUX_AUTO" in tbl.colnames

    def test_correct_number_of_sources(self, simple_hdu):
        tbl = self._run_with_mock(simple_hdu, n_sources=7)
        assert len(tbl) == 7

    def test_numpy_array_input(self):
        arr = np.random.default_rng(0).normal(1000, 10, (32, 32)).astype(np.float32)
        tbl = self._run_with_mock(arr, n_sources=3)
        assert len(tbl) == 3

    def test_hdulist_input(self, simple_hdulist):
        tbl = self._run_with_mock(simple_hdulist)
        assert isinstance(tbl, Table)

    def test_nonzero_returncode_raises(self, simple_hdu):
        with pytest.raises(RuntimeError, match="SExtractor exited"):
            self._run_with_mock(simple_hdu, returncode=1)

    def test_raise_on_error_false(self, simple_hdu):
        """raise_on_error=False should not raise even on nonzero exit."""
        sx = SExtractor(raise_on_error=False)
        cat_bytes = self._make_catalog_bytes(2)

        def fake_subprocess(cmd):
            idx = cmd.index("-CATALOG_NAME")
            with open(cmd[idx + 1], "wb") as f:
                f.write(cat_bytes)
            r = mock.MagicMock()
            r.returncode = 42
            r.stderr = b"some warning"
            return r

        sx._run_subprocess = fake_subprocess
        tbl = sx.run(simple_hdu)   # should not raise
        assert isinstance(tbl, Table)

    def test_module_level_run(self, simple_hdu):
        """Module-level run() convenience function."""
        sx_obj = SExtractor()
        cat_bytes = self._make_catalog_bytes(4)

        def fake_subprocess(cmd):
            idx = cmd.index("-CATALOG_NAME")
            with open(cmd[idx + 1], "wb") as f:
                f.write(cat_bytes)
            r = mock.MagicMock()
            r.returncode = 0
            r.stderr = b""
            return r

        with mock.patch.object(SExtractor, "_run_subprocess", fake_subprocess):
            tbl = run(simple_hdu)

        assert len(tbl) == 4

    def test_image_path_is_readable_fits(self, simple_hdu):
        """Verify that the path we pass to SExtractor is a real, openable FITS file."""
        captured_cmd = []

        def fake_subprocess(cmd):
            captured_cmd.extend(cmd)
            idx = cmd.index("-CATALOG_NAME")
            with open(cmd[idx + 1], "wb") as f:
                f.write(self._make_catalog_bytes(1))
            r = mock.MagicMock()
            r.returncode = 0
            r.stderr = b""
            return r

        sx = SExtractor()
        sx._run_subprocess = fake_subprocess
        sx.run(simple_hdu)

        image_path = captured_cmd[1]  # second arg is the image path
        assert os.path.exists(image_path), \
            f"Image path {image_path!r} does not exist while subprocess is running"
        with fits.open(image_path) as hdl:
            assert hdl[0].data.shape == simple_hdu.data.shape


# ---------------------------------------------------------------------------
# Integration test (requires real SExtractor binary)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Requires ``sex`` (or ``source-extractor``) in PATH."""

    def test_real_run(self):
        rng = np.random.default_rng(99)
        data = rng.normal(1000.0, 10.0, (256, 256)).astype(np.float32)
        # Plant 5 obvious point sources
        for cx, cy in [(50, 50), (100, 100), (150, 200), (200, 50), (30, 200)]:
            y, x = np.mgrid[:256, :256]
            data += 2000.0 * np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5 ** 2))

        hdu = fits.PrimaryHDU(data)
        sx = SExtractor(config={"DETECT_THRESH": 5.0, "DETECT_MINAREA": 5})

        tbl = sx.run(hdu)
        assert isinstance(tbl, Table)
        assert len(tbl) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not integration"])
