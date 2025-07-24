# PyFastFlow

GPU geomorphological and hydraulic flow routines powered by Taichi.

## Description

PyFastFlow is a high-performance Python package for geomorphological and hydraulic flow routing computations on GPU. 
The fast flow routines are implemented following Jain et al., 2024 [📝](https://www-sop.inria.fr/reves/Basilic/2024/JKGFC24/FastFlowPG2024_Author_Version.pdf)
It leverages the Taichi programming language for efficient parallel computation of flow accumulation, depression filling, and related hydrological algorithms.

## Features

- **GPU-accelerated flow routing**: Efficient steepest descent and flow accumulation algorithms
- **Depression filling**: Multiple algorithms for creating depressionless surfaces
- **Boundary conditions**: Support for periodic and custom boundary conditions
- **Parallel algorithms**: Rake-and-compress algorithms for efficient parallel processing
- **Lake flow handling**: Advanced algorithms for depression and lake flow computation

## Installation

```bash
pip install pyfastflow
```

Or install from source:

```bash
git clone https://github.com/bgailleton/fastflow_taichi.git
cd fastflow_taichi
pip install -e .
```

## Requirements

- Python >= 3.9
- Taichi >= 1.4.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

## Quick Start

TODO

## License

## Authors

Boris Gailleton - Géosciences Rennes - boris.gailleton@univ-rennes.fr
Guillaume Cordonnier - INRIA Sofia Antipolis Nice
<!-- 
## Citation

If you use PyFastFlow in your research, please cite:

```bibtex
@software{pyfastflow,
  title = {PyFastFlow: GPU geomorphological and hydraulic flow routines},
  author = {Gailleton, Boris},
  year = {2024},
  url = {https://github.com/bgailleton/fastflow_taichi}
}
``` -->
