import taichi as ti
import numpy as np
import pyfastflow as pf


@ti.kernel
def iterate_deposition_ptr_jmp_cte_kd(z:ti.template(), Qs:ti.template(), Q:ti.template(), rec_:ti.template(), rec__:ti.template()):
    """
    in log2(N) operations it should have traversed the whole landscape

    Potential optimisation to test one day: per-cell ping-pong read and write for rec (see general algorithms)
        
    Author: B.G.
    """

    for i in Qs:
        
        if(rec_[i] == rec_[rec_[i]] or pf.flow.neighbourer_flat.can_leave_domain(i) or Qs[i] <= 0):
            continue

        tzhs = ti.math.min(pf.constants.KD * Qs[i]/Q[rec_[i]], Qs[i] / pf.constants.DX**2)

        Qs[i] -= tzhs * pf.constants.DX**2

        ti.atomic_add(z[rec_[i]], tzhs * pf.constants.DT_SPL)
        rec__[i] = rec_[rec_[i]]

    for i in rec_:
        rec_[i] = rec__[i]