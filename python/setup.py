"""
Setup script for sextractor_jax – JAX acceleration for SExtractor.
"""

from setuptools import setup, find_packages

setup(
    name="sextractor_jax",
    version="0.1.0",
    description="JAX-accelerated computational kernels for SExtractor",
    long_description=open("../README.md").read() if __import__("os").path.exists("../README.md") else "",
    author="SExtractor Contributors",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.1",
        "jaxlib>=0.4.1",
        "numpy>=1.24",
    ],
    extras_require={
        "gpu": ["jax[cuda12_pip]"],
        "tpu": ["jax[tpu]"],
        "dev": ["pytest", "numpy", "astropy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
