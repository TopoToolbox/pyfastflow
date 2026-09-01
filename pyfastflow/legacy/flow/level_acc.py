import taichi as ti

import pyfastflow.grid.neighbourer_flat as nei

from .. import general_algorithms as gena


@ti.kernel
def ndon_MFD(z: ti.template(), ndons: ti.template()):
    for i in z:
        for k in ti.static(range(4)):
            j = nei.neighbour(i, k)
            if j > -1:
                if z[j] > z[i]:
                    ndons[i] += 1


@ti.kernel
def iteration_accumulate_flow_MFD(
    Q: ti.template(), z: ti.template(), ndons: ti.template()
) -> bool:
    changed: ti.u1 = False

    for i in Q:
        if ndons[i] == 0:
            changed = True

            ndons[i] = -1

            sumslope = gena.sumslope_downstream_node(z, i)

            for k in ti.static(range(4)):
                j = nei.neighbour(i, k)
                if j > -1:
                    if z[j] < z[i]:
                        tS = gena.slope_dir(z, i, k)

                        ti.atomic_add(Q[j], Q[i] * tS / sumslope)

                        # Absolutely at the end to avoid RAW in case j is now ready to be processed and picked up by another thread
                        ndons[j] -= 1
    return changed
