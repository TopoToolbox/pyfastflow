from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyfastflow",
    version="0.0.1",
    author="Boris Gailleton",
    author_email="boris.gailleton@univ-rennes.fr",
    description="GPU geomorphological and hydraulic flow routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bgailleton/fastflow_taichi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.9",
    install_requires=[
        "taichi>=1.4.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "click>=7.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "topotoolbox": [
            "topotoolbox",
        ],
        "visugl": [
            "moderngl>=5.8.0",
            "moderngl-window>=2.4.0",
            "imgui[glfw]>=1.4.0",
            "glfw>=2.5.0",
        ],
        "3dterrain": [
            "moderngl>=5.8.0",
            "moderngl-window>=2.4.0",
            "imgui[glfw]>=1.4.0",
            "glfw>=2.5.0",
            "rasterio>=1.3.0",
            "scipy>=1.7.0",
        ],
    },
    keywords="geomorphology hydraulics flow routing GPU taichi",
    project_urls={
        "Bug Reports": "https://github.com/bgailleton/fastflow_taichi/issues",
        "Source": "https://github.com/bgailleton/fastflow_taichi",
    },
    entry_points={
        "console_scripts": [
            "pff-raster2npy=pyfastflow.cli.raster_commands:raster2npy",
            "pff-upscale=pyfastflow.cli.rastermanip_commands:raster_upscale",
            "pff-downscale=pyfastflow.cli.rastermanip_commands:raster_downscale",
            "pff-dem2png=pyfastflow.cli.dem2png_commands:dem2png",
            "pff-boundary-gui=pyfastflow.cli.grid_commands:boundary_gui",
            "pff-precip-gui=pyfastflow.cli.precip_commands:precipitation_gui",
            "pff-terrain3d=pyfastflow.cli.terrain3d_cli:main",
        ],
    },
)
