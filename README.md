# PyFastFlow

> ⚠️ **Experimental.** This describes the in-progress v1 core. The API is
> settling and will change.

**GPU routines for Earth-surface processes — portable across backends.**

Two things in one:

- **A portable GPU-routine engine.** Compose parameters, helpers and
  multi-kernel *routines* once, and run them on **Taichi**, **Quadrants**,
  or **CuPy** — same composition model, backend-native kernels.
- **A geomorphology toolbox built on it.** Flow routing, flooding, and
  landscape evolution, engineered for grids of hundreds of millions of nodes.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CeCILL%20v2.1-red.svg)](./LICENSE)

## Why

Writing flexible GPU kernels for physics simulation can be time-consuming. Performance on GPU is highly affected by memory layout. Hence simple things such as a constant _vs_ a 2D varying parameter, or stencil operations (periodic boundaries, no-data, outlets, ...), each push you toward a different hand-tuned kernel — multiplied by every backend you want to run on. PyFastFlow lets you write a routine once and *bind* how each parameter and stencil is realised, then specialises it per backend and layout.

## Example

```python
import taichi as ti; ti.init(arch=ti.gpu)
from pyfastflow.core.context.taichi_backend import (
    TaichiKernelBuilder, TaichiParameter,
)
from pyfastflow.core.pool.taichi_pool import TaichiPool

pool = TaichiPool()
D = TaichiParameter("D", ti.f32, mode="scalar", value=1e-2, pool=pool)

def diffuse(z_out: ti.template(), z_in: ti.template()):
    for i, j in z_in:
        z_out[i, j] = z_in[i, j] + D.get(0) * (
            z_in[i-1, j] + z_in[i+1, j] + z_in[i, j-1] + z_in[i, j+1] - 4.0 * z_in[i, j])

step = TaichiKernelBuilder().bind("D", D).ingest(diffuse).compile()
```

The parameter / helper / routine API is identical on every backend; only the
kernel body changes — a Python `def` on Taichi and Quadrants, CUDA source on
CuPy. The same landscape-evolution model, built once per backend, lives in
[`examples/core/lem/`](./examples/core/lem/).

## Install

```bash
pip install -e .            # from source
```

Requires Python ≥ 3.9, and the backend(s) you target (Taichi, Quadrants, CuPy).

## Status

Core interface under active refactor toward v1.0 (+ paper). Flow/flood routines
are being ported onto the new core; landscape evolution is WIP.

## License & authors

CeCILL v2.1 — Boris Gailleton (Géosciences Rennes) · Guillaume Cordonnier (INRIA).
