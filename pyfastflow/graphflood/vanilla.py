"""GraphFloodVanilla: the iterative GraphFlood (CuPy only).

Shared config, inputs and initial surfaces: see ``_base``.

``run(n)``: n steps of MFD topology on z+h (local minima handled by
``mfd_local_minima``), rain accumulation into Qi, then
h += (Qi - Qo) dt / dx^2 with Qo the friction outflow at the frozen steepest
slope.

``run_transient(n)``: n local transient steps on the raw hydraulic surface
(no depression conditioning): MFD weights from the live drops, Manning Qo
limited to what the cell holds over dt plus its rain, Qi from the donors'
Qo, h += (Qi - Qo) dt / dx^2.

``dt`` (s, const or scalar) is the time step of both.

Convergence checks (``convergence()``, ``run_until_converged``): see
``_convergence``.

Typical use::

    gf = GraphFloodVanilla(be, nx=nx, ny=ny, dx=dx, precipitation=rain)
    gf.z.from_numpy(z)
    gf.fill_topography()
    gf.dt.set(1.0)
    gf.run(1000)

Author: B.G (10/2026)
"""

from ._base import TOPOLOGY, finish_program, graphflood_builder
from ._convergence import add_convergence, attach_convergence
from ._vanilla import (
    _transient_factory, _transient_plan, _update_factory, _update_plan,
)


def build_graphflood_vanilla() -> type:
    """Build the GraphFloodVanilla class."""
    b = add_convergence(graphflood_builder("GraphFloodVanilla"))
    b.param("dt", "auto", "f32", value=1.0e-3)
    b.add("update_depth", _update_factory, bind=_update_plan)
    b.add("transport_transient", _transient_factory, bind=_transient_plan)
    b.pipeline("run", TOPOLOGY + ("update_depth",))
    b.pipeline("run_transient", ("make_surface", "transport_transient"))
    return attach_convergence(finish_program(b.freeze()))


GraphFloodVanilla = build_graphflood_vanilla()

__all__ = ["GraphFloodVanilla", "build_graphflood_vanilla"]
