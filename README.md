# PyFastFlow

**GPU-accelerated geomorphological and hydraulic flow modeling powered by Taichi**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Taichi](https://img.shields.io/badge/taichi-≥1.6.0-orange.svg)](https://github.com/taichi-dev/taichi)
[![License](https://img.shields.io/badge/license-Custom-red.svg)](./LICENSE)

## Overview

**A lot of work will be put into finalising v1.0 in Q1 2026 + paper**


PyFastFlow is a high-performance Python package for general geomorphological and hydrodynamic computations on GPU. Built on the Taichi programming language with bindings to `pytopotoolbox`, it provides efficient, portable parallel algorithms for flow accumulation, depression filling, (simple) shallow water flow modeling, and landscape evolution simulations.

The fast flow routines are implemented following **Jain et al., 2024** [📝](https://www-sop.inria.fr/reves/Basilic/2024/JKGFC24/FastFlowPG2024_Author_Version.pdf), delivering state-of-the-art performance for GPU-oriented flow computation (flow accumulation and local minima handling).

The flooding/hydraulic modelling implements a GPU version of GraphFlood (Gailleton et al., 2024) for stationary solutions and an implementation of Bates et al., 2010 and inertial flow.

## 🚀 Key Features

### **Flow Routing & Hydrology**
- **GPU-accelerated flow routing**: Steepest descent algorithms with multiple boundary conditions
- **Advanced depression filling**: Priority flood and carving algorithms for handling closed basins
- **Flow accumulation**: Efficient rake-and-compress algorithms for parallel tree traversal
- **Boundary conditions**: Normal, periodic (EW/NS), and custom per-node boundary handling

### **2D Shallow Water Flow (Flood Modeling)**
- **LisFlood implementation**: Bates et al. 2010 explicit finite difference scheme
- **GraphFlood**: Fast approximation of the 2D shallow water 2D stationary solution
- **Manning's friction**: Configurable roughness coefficients
- **Precipitation input**: Rainfall and boundary conditions

### **Landscape Evolution**
- **Refactor in progress, not usable right now**
- **Stream Power Law (SPL)**: Bedrock erosion with detachment and transport-limited models
- **Sediment transport**: Erosion-deposition coupling with transport capacity
- **Tectonic uplift**: Block and spatially-varying uplift patterns
- **Non linear hillslope**: Based on Carretier et al. 2016

### **Visualization & Analysis**
- **Hillshading**: GPU-accelerated terrain shading with multiple illumination models
- **Real-time 3D visualization**: Interactive terrain rendering with using mordernGL (Experimental)

### **Performance & Memory**
- **Field pooling system**: Efficient GPU memory management for taichi with automatic field reuse/clean
- **General Parallel algorithms**: Utilities written in taichi (e.g. Blelloch parallel scan, ping-pong, swap, ...)
- **Scalable**: Handles large grids (up to hundreds of millions of nodes) efficiently

## 📦 Installation

### From PyPI (will be available at launch)
```bash
pip install pyfastflow
```

### From Source
```bash
git clone https://github.com/bgailleton/pyfastflow.git
cd pyfastflow
pip install -e .
```

### Requirements
- **Python** ≥ 3.9
- **Taichi** ≥ 1.4.0
- **NumPy** ≥ 1.20.0
- **Matplotlib** ≥ 3.3.0 (for visualization)

## 🏃‍♂️ Quick Start

WIP


## 🔬 Scientific Background

### Flow Routing Algorithms

- **Flow routing through local minimas**: Improved depression handling with Cordonnier et al. (2019) adapted to GPU by Jain et al. (2024)
- **Rake-and-Compress**: Parallel flow accumulation following Jain et al. (2024)

### Shallow Water Flow
- **LisFlood**: Explicit finite difference scheme (Bates et al., 2010)
- **GraphFlood**: Graph-based implicit flow routing (Gaileton et al., 2024)

### Landscape Evolution
- **Stream Power Law**: E = K × A^m × S^n erosion model (Howard and Kerby 1983)
- **Transport-Limited**: Erosion-transport-deposition coupling (Davy and Lague, 2009 style - WIP)
- **Non linear hillslope**: adapted from CIDRE with a semi-implicit scheme (Carretier et al., 2016 - WIP)


## 📄 License

CeCILL v2.1

## 📖 Citation

If you use PyFastFlow in your research, please contact me.

### Related Publications
- Jain, A., et al. (2024). "Fast Flow Computation using GPU". *Proceedings of Graphics Interface 2024*.
- Bates, P. D., et al. (2010). "A simple inertial formulation of the shallow water equations". *Journal of Hydrology*.

## 👥 Authors

**Main Authors:**
- **Boris Gailleton** - Géosciences Rennes - boris.gailleton@univ-rennes.fr
- **Guillaume Cordonnier** - INRIA Sophia Antipolis

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/bgailleton/pyfastflow/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/bgailleton/pyfastflow/discussions)
- **Documentation**: [Read the Docs](https://pyfastflow.readthedocs.io/) *(coming soon - hopefully)*

## 🔗 Links

- **Repository**: https://github.com/bgailleton/pyfastflow
- **Documentation**: https://pyfastflow.readthedocs.io/ *(coming soon)*
- **PyPI Package**: https://pypi.org/project/pyfastflow/ *(coming soon)*
- **Jain et al. 2024 Paper**: [PDF](https://www-sop.inria.fr/reves/Basilic/2024/JKGFC24/FastFlowPG2024_Author_Version.pdf)

---
