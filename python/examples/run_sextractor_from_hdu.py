"""
run_sextractor_from_hdu.py
--------------------------
Examples of running SExtractor on an in-memory FITS HDU —
no image files are written to disk.

Requirements
------------
    pip install astropy numpy
    # and SExtractor installed:  apt install source-extractor
    #                         or: conda install -c conda-forge astromatic-source-extractor

Usage
-----
    python run_sextractor_from_hdu.py
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table

from sextractor_jax.sextractor import SExtractor, run, memory_backend


# ---------------------------------------------------------------------------
# 0.  Show which memory backend will be used (memfd / /dev/shm / tempfile)
# ---------------------------------------------------------------------------

print("Memory backend:", memory_backend())
print()


# ---------------------------------------------------------------------------
# 1.  Minimal one-liner  –  run() convenience function
# ---------------------------------------------------------------------------

def example_one_liner():
    """Simplest possible call: numpy array in, astropy Table out."""
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 15.0, (512, 512)).astype(np.float32)

    # Plant three obvious point sources
    for cx, cy, peak in [(100, 150, 3000), (300, 200, 5000), (400, 400, 2000)]:
        y, x = np.mgrid[:512, :512]
        data += peak * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 2.5**2))

    cat = run(data, config={"DETECT_THRESH": 3.0, "DETECT_MINAREA": 5})

    print("=== Example 1: one-liner ===")
    print(f"Found {len(cat)} sources")
    if len(cat):
        print(cat["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])
    print()


# ---------------------------------------------------------------------------
# 2.  Using an astropy PrimaryHDU with WCS header
# ---------------------------------------------------------------------------

def example_with_wcs_header():
    """Pass a proper HDU so SExtractor picks up WCS and outputs RA/Dec."""
    rng = np.random.default_rng(7)
    data = rng.normal(500.0, 8.0, (256, 256)).astype(np.float32)

    # Add a few sources
    for cx, cy in [(60, 80), (180, 120), (200, 200)]:
        y, x = np.mgrid[:256, :256]
        data += 1500.0 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 2.0**2))

    # Minimal TAN-projection WCS header
    header = fits.Header()
    header["NAXIS"]  = 2
    header["NAXIS1"] = 256
    header["NAXIS2"] = 256
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 128.0
    header["CRPIX2"] = 128.0
    header["CRVAL1"] = 150.0        # RA  of reference pixel (degrees)
    header["CRVAL2"] =   2.0        # Dec of reference pixel (degrees)
    header["CD1_1"]  = -2.78e-4     # ~1 arcsec/pixel
    header["CD1_2"]  =  0.0
    header["CD2_1"]  =  0.0
    header["CD2_2"]  =  2.78e-4

    hdu = fits.PrimaryHDU(data, header=header)

    sx = SExtractor(
        config={
            "DETECT_THRESH":  3.0,
            "DETECT_MINAREA": 5,
            "PIXEL_SCALE":    0,    # read from WCS
        },
        params=[
            "NUMBER",
            "X_IMAGE", "Y_IMAGE",
            "X_WORLD", "Y_WORLD",   # RA / Dec (requires WCS)
            "FLUX_AUTO", "FLUXERR_AUTO",
            "MAG_AUTO",
            "FWHM_IMAGE",
            "FLAGS",
        ],
    )

    cat = sx.run(hdu)

    print("=== Example 2: HDU with WCS header ===")
    print(f"Found {len(cat)} sources")
    if len(cat):
        print(cat["X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD", "MAG_AUTO"])
    print()


# ---------------------------------------------------------------------------
# 3.  Passing a weight map alongside the science image
# ---------------------------------------------------------------------------

def example_weight_map():
    """Pass a weight map alongside the science image."""
    rng = np.random.default_rng(99)
    data = rng.normal(1000.0, 10.0, (256, 256)).astype(np.float32)

    # Add sources
    for cx, cy in [(80, 80), (160, 160)]:
        y, x = np.mgrid[:256, :256]
        data += 2000.0 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 3.0**2))

    science_hdu = fits.PrimaryHDU(data)

    # Weight map: uniform except for a bad column
    weight = np.ones((256, 256), dtype=np.float32)
    weight[:, 120:125] = 0.0          # bad columns → zero weight
    weight_hdu = fits.PrimaryHDU(weight)

    sx = SExtractor(
        config={
            "DETECT_THRESH": 2.5,
            "WEIGHT_TYPE":   "MAP_WEIGHT",
        },
    )

    cat = sx.run(science_hdu, weight_hdu=weight_hdu)

    print("=== Example 3: weight map ===")
    print(f"Found {len(cat)} sources")
    if len(cat):
        print(cat["X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLAGS"])
    print()


# ---------------------------------------------------------------------------
# 4.  Re-using the SExtractor object across many HDUs
#     (e.g. iterating over extensions in a multi-extension FITS file)
# ---------------------------------------------------------------------------

def example_batch_extensions():
    """Process every image extension in an HDUList without reloading config."""
    rng = np.random.default_rng(0)

    # Build a mock multi-extension file (e.g. a mosaic camera)
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    for chip in range(4):
        data = rng.normal(800.0, 12.0, (128, 128)).astype(np.float32)
        cx, cy = rng.integers(20, 108, size=2)
        y, x = np.mgrid[:128, :128]
        data += 2500.0 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 2.0**2))
        hdulist.append(fits.ImageHDU(data, name=f"CHIP{chip}"))

    sx = SExtractor(config={"DETECT_THRESH": 3.5, "DETECT_MINAREA": 4})

    print("=== Example 4: batch over extensions ===")
    all_cats = []
    for ext in hdulist[1:]:            # skip empty Primary
        cat = sx.run(ext)              # SExtractor object is reused
        cat["CHIP"] = ext.name         # tag with chip name
        all_cats.append(cat)
        print(f"  {ext.name}: {len(cat)} sources")

    if all_cats:
        from astropy.table import vstack
        full_cat = vstack(all_cats)
        print(f"  Total: {len(full_cat)} sources across {len(all_cats)} chips")
    print()


# ---------------------------------------------------------------------------
# 5.  Reading a real FITS file from disk, then passing the HDU in-memory
#     (demonstrates the typical use-case: you already have the file open)
# ---------------------------------------------------------------------------

def example_from_open_file(fits_path: str):
    """
    Read an existing FITS file and extract sources without writing a temp copy.

    Parameters
    ----------
    fits_path : str – path to a real FITS image on disk
    """
    with fits.open(fits_path, memmap=False) as hdul:
        # Pick the first image extension
        for ext in hdul:
            if isinstance(ext, (fits.PrimaryHDU, fits.ImageHDU)) \
                    and ext.data is not None and ext.data.ndim == 2:
                hdu = ext
                break
        else:
            print(f"No 2-D image extension found in {fits_path}")
            return

    sx = SExtractor(
        config={
            "DETECT_THRESH":  2.0,
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

    cat = sx.run(hdu)

    print(f"=== Example 5: from open file ({fits_path}) ===")
    print(f"Found {len(cat)} sources")
    if len(cat):
        print(cat[:10])   # first 10 rows
    print()

    return cat


# ---------------------------------------------------------------------------
# Run all self-contained examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_one_liner()
    example_with_wcs_header()
    example_weight_map()
    example_batch_extensions()

    # example_from_open_file() requires a real FITS file; uncomment to use:
    # example_from_open_file("/path/to/your/image.fits")
