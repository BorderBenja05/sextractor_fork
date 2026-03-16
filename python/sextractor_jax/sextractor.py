"""
sextractor.py
-------------
Python interface for running SExtractor on an in-memory FITS HDU without
writing image data to disk.

How it works
------------
SExtractor's bundled FITS library opens image files with ``fopen()`` followed
by ``fseek()`` (random access required — named pipes would fail here).

On Linux ≥ 3.17 we use ``memfd_create(2)`` to create an **anonymous,
RAM-backed file descriptor**.  The kernel exposes it at
``/proc/<pid>/fd/<n>``, which looks like a regular seekable file to any
process running as the same user.  We write the serialised FITS bytes into
this fd, then pass the path string to SExtractor.  **No bytes ever touch a
storage device.**

Fallback chain (used when ``memfd_create`` is unavailable, e.g. on macOS or
kernels < 3.17):
  1. ``/dev/shm/<name>``  – RAM-backed tmpfs on Linux (no disk write)
  2. ``tempfile.NamedTemporaryFile``  – regular temp file (disk write)

The output catalog is handled symmetrically:
  • By default SExtractor is asked to write ASCII_HEAD to ``/dev/stdout``,
    which is captured via ``subprocess.PIPE`` — no output file needed.
  • Alternatively, ``catalog_type='FITS_LDAC'`` writes to a second memfd
    and the result is returned as an ``astropy.table.Table``.

Public API
----------
SExtractor(binary, default_config, ...)
    Persistent runner object.  Call ``runner.run(hdu)`` repeatedly.

run(hdu, ...)
    Module-level convenience function for one-off calls.

Examples
--------
>>> from astropy.io import fits
>>> import numpy as np
>>> from sextractor_jax.sextractor import SExtractor
>>>
>>> data = np.random.normal(1000, 30, (512, 512)).astype(np.float32)
>>> hdu = fits.PrimaryHDU(data)
>>>
>>> sx = SExtractor(config={'DETECT_MINAREA': 5, 'DETECT_THRESH': 1.5})
>>> cat = sx.run(hdu)
>>> print(cat['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO'])
"""

from __future__ import annotations

import ctypes
import io
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import astropy.io.fits as fits
    from astropy.table import Table
    import astropy.io.ascii as ascii_reader
    _ASTROPY = True
except ImportError:
    fits = None  # type: ignore
    Table = None  # type: ignore
    _ASTROPY = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None  # type: ignore
    _NUMPY = False


# ---------------------------------------------------------------------------
# memfd_create wrapper
# ---------------------------------------------------------------------------

_libc = None
_MEMFD_CLOEXEC = 1      # kernel flag: close-on-exec (we clear it later)
_MFD_ALLOW_SEALING = 2  # optional, not required


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # Declare the prototype
        _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        _libc.memfd_create.restype = ctypes.c_int
    return _libc


def _memfd_create(name: str) -> int:
    """
    Create an anonymous, RAM-backed file descriptor via ``memfd_create(2)``.

    Returns the file descriptor number.
    Raises ``OSError`` if the syscall fails (e.g. kernel < 3.17).
    """
    libc = _get_libc()
    fd = libc.memfd_create(name.encode(), 0)   # 0 = no flags (inheritable)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), f"memfd_create({name!r})")
    return fd


def _memfd_available() -> bool:
    """Return True if memfd_create is available on this system."""
    if sys.platform != "linux":
        return False
    try:
        fd = _memfd_create("_probe")
        os.close(fd)
        return True
    except (OSError, AttributeError):
        return False


# Cache availability check
_USE_MEMFD: bool | None = None


def _use_memfd() -> bool:
    global _USE_MEMFD
    if _USE_MEMFD is None:
        _USE_MEMFD = _memfd_available()
    return _USE_MEMFD


# ---------------------------------------------------------------------------
# Memory file context managers
# ---------------------------------------------------------------------------

@contextmanager
def _memory_input_file(data: bytes, suffix: str = ".fits"):
    """
    Context manager that exposes *data* as a seekable file path.

    Yields ``(path: str, fd: int | None)``.

    Priority:
      1. memfd_create  → ``/proc/<pid>/fd/<n>``   (no disk write)
      2. /dev/shm      → RAM tmpfs                 (no disk write)
      3. tempfile      → regular temp file         (disk write fallback)
    """
    if _use_memfd():
        # --- memfd approach ---
        fd = _memfd_create("sex_input")
        try:
            os.write(fd, data)
            # Do NOT close or rewind here; SExtractor will open the path
            # independently via fopen(), which creates a new file description
            # at offset 0.
            path = f"/proc/{os.getpid()}/fd/{fd}"
            yield path, fd
        finally:
            os.close(fd)

    elif os.path.isdir("/dev/shm"):
        # --- /dev/shm RAM tmpfs approach ---
        tmp = tempfile.NamedTemporaryFile(
            dir="/dev/shm", suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            yield tmp.name, None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    else:
        # --- Regular tempfile fallback ---
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            yield tmp.name, None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@contextmanager
def _memory_output_file(suffix: str = ".cat"):
    """
    Context manager that provides a writable path for SExtractor's output
    catalog, and yields the path together with a way to read the bytes back.

    Yields ``(path: str, read_fn: callable() -> bytes)``.
    """
    if _use_memfd():
        fd = _memfd_create("sex_output")
        try:
            path = f"/proc/{os.getpid()}/fd/{fd}"

            def _read() -> bytes:
                os.lseek(fd, 0, os.SEEK_SET)
                chunks = []
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)

            yield path, _read
        finally:
            os.close(fd)

    elif os.path.isdir("/dev/shm"):
        tmp = tempfile.NamedTemporaryFile(
            dir="/dev/shm", suffix=suffix, delete=False)
        tmp.close()
        try:
            def _read() -> bytes:
                with open(tmp.name, "rb") as f:
                    return f.read()

            yield tmp.name, _read
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    else:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        try:
            def _read() -> bytes:
                with open(tmp.name, "rb") as f:
                    return f.read()

            yield tmp.name, _read
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FITS serialisation helpers
# ---------------------------------------------------------------------------

def _hdu_to_bytes(hdu_or_array) -> bytes:
    """
    Convert an astropy HDU / HDUList / numpy array to raw FITS bytes in RAM.
    Never touches disk.
    """
    if not _ASTROPY:
        raise ImportError("astropy is required: pip install astropy")

    buf = io.BytesIO()

    if isinstance(hdu_or_array, fits.HDUList):
        hdu_or_array.writeto(buf)
    elif isinstance(hdu_or_array, (fits.PrimaryHDU,
                                   fits.ImageHDU,
                                   fits.CompImageHDU)):
        fits.HDUList([hdu_or_array]).writeto(buf)
    elif _NUMPY and isinstance(hdu_or_array, np.ndarray):
        fits.HDUList([fits.PrimaryHDU(hdu_or_array)]).writeto(buf)
    else:
        raise TypeError(
            f"Expected astropy HDU/HDUList or numpy array, got {type(hdu_or_array)}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Catalog parsing helpers
# ---------------------------------------------------------------------------

def _parse_ascii_head(raw: bytes) -> "Table":
    """Parse SExtractor ASCII_HEAD catalog bytes into an astropy Table."""
    text = raw.decode(errors="replace")
    return ascii_reader.read(text, format="sextractor")


def _parse_fits_ldac(raw: bytes) -> "Table":
    """Parse a FITS_LDAC catalog from raw bytes into an astropy Table."""
    buf = io.BytesIO(raw)
    with fits.open(buf) as hdl:
        # LDAC format: PRIMARY + LDAC_IMHEAD (BinTable) + LDAC_OBJECTS (BinTable)
        for ext in hdl:
            if hasattr(ext, "columns") and ext.name not in ("PRIMARY", "LDAC_IMHEAD"):
                return Table(ext.data)
        # Fallback: last BinTable
        for ext in reversed(hdl):
            if hasattr(ext, "columns"):
                return Table(ext.data)
    raise ValueError("Could not find object table in FITS_LDAC catalog")


def _parse_catalog(raw: bytes, catalog_type: str) -> "Table | None":
    """Dispatch to the appropriate catalog parser."""
    if not raw:
        return Table() if _ASTROPY else None
    if catalog_type.upper().startswith("FITS"):
        return _parse_fits_ldac(raw)
    else:
        return _parse_ascii_head(raw)


# ---------------------------------------------------------------------------
# Default SExtractor parameters
# ---------------------------------------------------------------------------

# Columns to request when the caller doesn't specify a param file.
DEFAULT_PARAMS = [
    "NUMBER",
    "X_IMAGE", "Y_IMAGE",
    "X_WORLD", "Y_WORLD",
    "FLUX_AUTO", "FLUXERR_AUTO",
    "MAG_AUTO", "MAGERR_AUTO",
    "FLUX_APER", "FLUXERR_APER",
    "A_IMAGE", "B_IMAGE",
    "THETA_IMAGE",
    "FWHM_IMAGE",
    "ELLIPTICITY",
    "FLAGS",
    "CLASS_STAR",
]

# Minimal default configuration (override with the config= argument)
DEFAULT_CONFIG: dict[str, Any] = {
    "DETECT_TYPE":    "CCD",
    "DETECT_MINAREA": 3,
    "DETECT_THRESH":  1.5,
    "ANALYSIS_THRESH": 1.5,
    "FILTER":         "Y",
    "DEBLEND_NTHRESH": 32,
    "DEBLEND_MINCONT": 0.005,
    "CLEAN":          "Y",
    "CLEAN_PARAM":    1.0,
    "MASK_TYPE":      "CORRECT",
    "PHOT_APERTURES": 10.0,
    "PHOT_AUTOPARAMS": "2.5,3.5",
    "SATUR_LEVEL":    50000.0,
    "MAG_ZEROPOINT":  0.0,
    "PIXEL_SCALE":    0.0,        # 0 = read from WCS
    "BACK_SIZE":      64,
    "BACK_FILTERSIZE": 3,
    "BACKPHOTO_TYPE": "GLOBAL",
    "VERBOSE_TYPE":   "QUIET",
    "CATALOG_TYPE":   "ASCII_HEAD",
}


# ---------------------------------------------------------------------------
# SExtractor runner class
# ---------------------------------------------------------------------------

class SExtractor:
    """
    Run SExtractor on an in-memory FITS HDU without writing the image to disk.

    Parameters
    ----------
    binary : str
        Path to the ``sex`` (or ``source-extractor``) executable.
        Defaults to ``sex`` (found via PATH).
    config : dict, optional
        SExtractor configuration key-value pairs.  These are passed as
        ``-KEY VALUE`` arguments on the command line, overriding defaults.
    params : list of str, optional
        Output columns to request (written to a temporary param file).
        Defaults to :data:`DEFAULT_PARAMS`.
    extra_args : list of str, optional
        Additional raw command-line arguments appended verbatim.
    timeout : float, optional
        Subprocess timeout in seconds (default: 300).
    raise_on_error : bool
        If True (default), raise ``RuntimeError`` when SExtractor exits
        non-zero.

    Examples
    --------
    >>> sx = SExtractor(config={'DETECT_THRESH': 2.0})
    >>> cat = sx.run(my_hdu)
    >>> cat['FLUX_AUTO']
    """

    def __init__(
        self,
        binary: str = "sex",
        config: dict[str, Any] | None = None,
        params: list[str] | None = None,
        extra_args: list[str] | None = None,
        timeout: float = 300.0,
        raise_on_error: bool = True,
    ):
        self.binary = binary
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update({k.upper(): v for k, v in config.items()})
        self.params = list(params) if params else list(DEFAULT_PARAMS)
        self.extra_args = list(extra_args) if extra_args else []
        self.timeout = timeout
        self.raise_on_error = raise_on_error

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        hdu,
        weight_hdu=None,
        config: dict[str, Any] | None = None,
    ) -> "Table":
        """
        Run SExtractor on *hdu* and return the source catalog.

        Parameters
        ----------
        hdu : astropy HDU, HDUList, or numpy ndarray
            The detection/measurement image.  May be a ``PrimaryHDU``,
            ``ImageHDU``, ``HDUList``, or a plain numpy array.
        weight_hdu : astropy HDU or numpy array, optional
            Weight / variance / RMS map passed via ``-WEIGHT_IMAGE``.
        config : dict, optional
            Per-call config overrides (merged on top of instance config).

        Returns
        -------
        catalog : astropy.table.Table
        """
        # Merge per-call config
        effective_config = dict(self.config)
        if config:
            effective_config.update({k.upper(): v for k, v in config.items()})

        # Serialise images to bytes in RAM
        image_bytes = _hdu_to_bytes(hdu)
        weight_bytes = _hdu_to_bytes(weight_hdu) if weight_hdu is not None else None

        catalog_type = effective_config.get("CATALOG_TYPE", "ASCII_HEAD").upper()

        # Use a temp file for the param list (small, text only)
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".param", delete=False) as pf:
            pf.write("\n".join(self.params) + "\n")
            param_path = pf.name

        try:
            with _memory_input_file(image_bytes) as (image_path, _image_fd):
                # Weight image (optional)
                if weight_bytes is not None:
                    weight_ctx = _memory_input_file(weight_bytes)
                else:
                    weight_ctx = _null_ctx()

                with weight_ctx as (weight_path, _wfd):
                    # Output catalog
                    with _memory_output_file(
                            suffix=".cat" if "FITS" not in catalog_type
                            else ".fits") as (cat_path, read_catalog):

                        cmd = self._build_command(
                            image_path, weight_path,
                            cat_path, param_path,
                            effective_config,
                        )

                        result = self._run_subprocess(cmd)

                        cat_bytes = read_catalog()

            catalog = _parse_catalog(cat_bytes, catalog_type)
            return catalog

        finally:
            try:
                os.unlink(param_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(
        self,
        image_path: str,
        weight_path: str | None,
        catalog_path: str,
        param_path: str,
        config: dict,
    ) -> list[str]:
        """Assemble the SExtractor command-line arguments."""
        cmd = [self.binary, image_path]

        # Pass every config key as -KEY VALUE (avoids needing a .sex file)
        for key, val in config.items():
            if key in ("CATALOG_NAME", "PARAMETERS_NAME",
                       "WEIGHT_IMAGE", "WEIGHT_TYPE"):
                continue   # handled explicitly below
            cmd += [f"-{key}", str(val)]

        # Catalog output
        cmd += ["-CATALOG_NAME", catalog_path]

        # Parameters file
        cmd += ["-PARAMETERS_NAME", param_path]

        # Weight image
        if weight_path is not None:
            cmd += ["-WEIGHT_IMAGE", weight_path]
            if "WEIGHT_TYPE" not in config:
                cmd += ["-WEIGHT_TYPE", "MAP_WEIGHT"]

        # Suppress the filter file requirement if not provided
        if "FILTER_NAME" not in config:
            cmd += ["-FILTER", "N"]

        # Extra raw arguments
        cmd += self.extra_args

        return cmd

    def _run_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """Execute the SExtractor subprocess, capturing stderr for diagnostics."""
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SExtractor binary not found: {self.binary!r}. "
                "Install source-extractor or set the binary= argument."
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"SExtractor timed out after {self.timeout}s"
            ) from e

        if self.raise_on_error and result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(
                f"SExtractor exited with code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{stderr}"
            )

        return result


# ---------------------------------------------------------------------------
# Null context manager (for optional weight image)
# ---------------------------------------------------------------------------

@contextmanager
def _null_ctx():
    """Context manager yielding (None, None) when no weight image is provided."""
    yield None, None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run(
    hdu,
    weight_hdu=None,
    binary: str = "sex",
    config: dict[str, Any] | None = None,
    params: list[str] | None = None,
    **kwargs,
) -> "Table":
    """
    Run SExtractor on *hdu* and return the catalog.

    Convenience wrapper around :class:`SExtractor` for one-off calls.

    Parameters
    ----------
    hdu : astropy HDU / HDUList / numpy array
    weight_hdu : optional weight image
    binary : str – path to the ``sex`` executable
    config : dict – SExtractor configuration overrides
    params : list of str – output columns to request
    **kwargs : passed to :class:`SExtractor`

    Returns
    -------
    catalog : astropy.table.Table

    Examples
    --------
    >>> from astropy.io import fits
    >>> from sextractor_jax.sextractor import run as sex_run
    >>> cat = sex_run(fits.open('image.fits')[0],
    ...               config={'DETECT_THRESH': 3.0})
    """
    sx = SExtractor(binary=binary, config=config, params=params, **kwargs)
    return sx.run(hdu, weight_hdu=weight_hdu)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def memory_backend() -> str:
    """Return which memory backend will be used on the current system."""
    if _use_memfd():
        return "memfd_create (zero disk writes, RAM-backed, seekable)"
    elif os.path.isdir("/dev/shm"):
        return "/dev/shm (RAM tmpfs, no disk writes)"
    else:
        return "tempfile (disk write, fallback)"
