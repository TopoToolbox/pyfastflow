"""
The one piece of closure-backend block plumbing every flow section needs:
the data-argument annotation a kernel template declares for a Tensor-typed
parameter on Taichi vs Quadrants. Kept as its own module rather than
duplicated (or homed arbitrarily) in one of _closure_receivers.py/
_closure_accum.py/_closure_depressions.py/_closure_reconstruct.py, which all
import it from here.

Author: B.G (07/2026)
"""


def _tensor_annotation(backend_mod, backend: str):
    """
    The data-argument annotation a kernel template needs on this closure
    backend: `ti.template()` for Taichi, `qd.Tensor` for Quadrants - mirrors
    ../ops/_closure_blocks.py's _tensor_annotation.

    Author: B.G (07/2026)
    """
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor
