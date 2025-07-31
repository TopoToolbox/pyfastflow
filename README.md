# PyFastFlow

**GPU-accelerated geomorphological and hydraulic flow modeling powered by Taichi**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Taichi](https://img.shields.io/badge/taichi-≥1.4.0-orange.svg)](https://github.com/taichi-dev/taichi)
[![License](https://img.shields.io/badge/license-Custom-red.svg)](./LICENSE)

## Overview

PyFastFlow is a high-performance Python package for geomorphological and hydraulic flow routing computations on GPU. Built on the Taichi programming language, it provides efficient parallel algorithms for flow accumulation, depression filling, shallow water flow modeling, and landscape evolution simulations.

The fast flow routines are implemented following **Jain et al., 2024** [📝](https://www-sop.inria.fr/reves/Basilic/2024/JKGFC24/FastFlowPG2024_Author_Version.pdf), delivering state-of-the-art performance for large-scale geomorphological modeling.

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
- **Stream Power Law (SPL)**: Bedrock erosion with detachment and transport-limited models
- **Sediment transport**: Erosion-deposition coupling with transport capacity
- **Tectonic uplift**: Block and spatially-varying uplift patterns
- **Implicit solvers**: Stable numerical schemes for large time steps

### **Visualization & Analysis**
- **Hillshading**: GPU-accelerated terrain shading with multiple illumination models
- **Real-time 3D visualization**: Interactive terrain rendering with Taichi GGUI (WiP)

### **Performance & Memory**
- **Field pooling system**: Efficient GPU memory management with automatic field reuse
- **Parallel algorithms**: Optimized for modern GPU architectures
- **Scalable**: Handles large grids (millions of nodes) efficiently

## 📦 Installation

### From PyPI (recommended - not ready yet though)
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

### Basic Flow Routing
```python
import pyfastflow as pf
import numpy as np
import taichi as ti

# Initialize Taichi (GPU backend)
ti.init(ti.gpu)

# Create synthetic topography
nx, ny = 512, 512
dx = 10.0  # 10 meter resolution
elevation = np.random.rand(ny, nx) * 100

# Create grid and flow router
grid = pf.flow.GridField(nx, ny, dx)
grid.set_z(elevation)

router = pf.flow.FlowRouter(grid)

# Compute flow routing
router.compute_receivers()           # Steepest descent
router.reroute_flow()               # Handle depressions
router.accumulate_constant_Q(1.0)   # Flow accumulation

# Get results
flow_accumulation = router.get_Q()   # 2D numpy array
drainage_area = flow_accumulation * dx * dx
```

### GraphFlood Modeling
```python
# Advanced flood modeling with diffusion
flooder.run_graphflood(
    N=10,              # Iterations
    N_stochastic=4,    # Stochastic flow paths
    N_diffuse=2,       # Diffusion steps
    temporal_filtering=0.1  # Temporal smoothing
)
```

### 2D Flood Modeling
```python
# Create flood model
flooder = pf.flood.Flooder(
    router, 
    precipitation_rates=10e-3/3600,  # 10 mm/hr
    manning=0.033,                   # Manning's n
    dt_hydro=1e-3                    # Time step
)

# Run LisFlood simulation
flooder.run_LS(N=1000)  # 1000 time steps

# Get flood depths
water_depth = flooder.get_h()        # 2D numpy array
discharge_x = flooder.get_qx()       # x-direction discharge
discharge_y = flooder.get_qy()       # y-direction discharge
```

### Landscape Evolution
```python
# Setup SPL erosion model
alpha = ti.field(ti.f32, shape=(nx*ny,))
alpha_ = ti.field(ti.f32, shape=(nx*ny,))
alpha.fill(1e-5)   # Erosion coefficient
alpha_.fill(1e-5)

# Run erosion time step
pf.erodep.SPL(router, alpha, alpha_)

# Get eroded topography
new_elevation = router.get_Z()
```

### Real-time Visualization
```python
# 3D terrain visualization
viewer = pf.visu.SurfaceViewer(elevation)
viewer.run()

# Hillshading
hillshade = pf.visu.hillshade_numpy(
    elevation, 
    altitude_deg=45, 
    azimuth_deg=315
)

# Multidirectional hillshading
multi_hs = pf.visu.hillshade_multidirectional_numpy(elevation)
```

### Memory Management
```python
# Automatic field pooling (recommended)
with pf.pool.temp_field(ti.f32, (nx*ny,)) as temp:
    # Use temp field...
    some_computation(temp)
# Field automatically released

# Manual field management
temp = pf.pool.get_temp_field(ti.f32, (nx*ny,))
# Use temp field...
pf.pool.release_temp_field(temp)

# Pool statistics
stats = pf.pool.pool_stats()
print(f"Pool usage: {stats['in_use']}/{stats['total']}")
```

## 📚 Advanced Usage

### Custom Boundary Conditions
```python
# Create custom boundary mask
boundaries = np.ones((ny, nx), dtype=np.uint8)
boundaries[0, :] = 3    # Top edge: can leave domain
boundaries[-1, :] = 3   # Bottom edge: can leave domain
boundaries[:, 0] = 1    # Left edge: cannot leave
boundaries[:, -1] = 1   # Right edge: cannot leave

grid = pf.flow.GridField(nx, ny, dx, boundary_mode='custom')
grid.set_boundaries(boundaries)
```

### Stochastic Flow Routing
```python
# Enable stochastic receivers
router.compute_stochastic_receivers()

# Stochastic flow accumulation (multiple realizations)
router.accumulate_constant_Q_stochastic(
    value=1.0, 
    area=True, 
    N=10  # 10 stochastic realizations
)
```

## 🏗️ Package Structure

```
pyfastflow/
├── constants.py          # Global constants and configuration
├── flow/              # Flow routing algorithms
│   ├── flowfields.py  # FlowRouter class
│   ├── receivers.py   # Steepest descent algorithms
│   ├── lakeflow.py    # Depression handling
│   └── fill_topo.py   # Topographic filling
├── flood/             # 2D shallow water flow
│   ├── gf_fields.py   # Flooder class
│   └── gf_ls.py       # LisFlood kernels
├── erodep/            # Erosion and landscape evolution
│   └── SPL.py         # Stream Power Law models
├── visu/              # Visualization tools
│   ├── live.py        # 3D real-time visualization
│   └── hillshading.py # Terrain shading algorithms
├── pool/              # Memory management
│   └── pool.py        # Field pooling system
└── general_algorithms/ # Fundamental algorithms
    ├── parallel_scan.py  # Parallel prefix sum
    └── pingpong.py       # Double buffering utilities
```

## 🔬 Scientific Background

### Flow Routing Algorithms
- **Steepest Descent**: O'Callaghan & Mark (1984) with GPU optimization
- **Lake Flow**: Improved depression handling with priority flood algorithms
- **Rake-and-Compress**: Parallel flow accumulation following Jain et al. (2024)

### Shallow Water Flow
- **LisFlood**: Explicit finite difference scheme (Bates et al., 2010)
- **GraphFlood**: Graph-based implicit flow routing (Gaileton et al., 2024)

### Landscape Evolution
- **Stream Power Law**: E = K × A^m × S^n erosion model
- **Transport-Limited**: Erosion-transport-deposition coupling
- **Implicit Schemes**: Stable numerical methods for large time steps

## 🎯 Performance

PyFastFlow achieves significant speedups over traditional CPU implementations:

- **Flow routing**: 10-100× faster than equivalent CPU algorithms
- **Flood modeling**: Real-time simulation for grids up to 1M+ nodes
- **Memory efficiency**: Advanced pooling system minimizes GPU memory usage
- **Scalability**: Near-linear scaling with grid size on modern GPUs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/bgailleton/pyfastflow.git
cd pyfastflow
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

This project uses a **restrictive license** for commercial applications. See [LICENSE](LICENSE) for details.

## 📖 Citation

If you use PyFastFlow in your research, please cite:

```bibtex
@software{pyfastflow,
  title = {PyFastFlow: GPU geomorphological and hydraulic flow routines},
  author = {Gailleton, Boris and Cordonnier, Guillaume},
  year = {2024},
  url = {https://github.com/bgailleton/pyfastflow}
}
```

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
