"""


Author: B.G.
"""

import taichi as ti

import pyfastflow.flow as flow

from .. import constants as cte


@ti.func
def slope_pos(a, b):
    return ti.max(0.0, (a - b) / cte.DX)

@ti.kernel
def build_S(S: ti.template()):
    for i in S:
        S[i] = cte.PREC * cte.DX * cte.DX   # stationary source

@ti.kernel
def sweep_color(Q: ti.template(), zh: ti.template(), S: ti.template(),
                parity: ti.i32):
    for lin in zh:
        if flow.neighbourer_flat.can_leave_domain(lin) or flow.neighbourer_flat.nodata(lin):
            continue
        iy,ix = flow.neighbourer_flat.rc_from_i(lin)
        if ((ix + iy) & 1) != parity:   # cheap parity
            continue

        acc = S[lin]  # <-- source added every sweep

        # gather influx from opposite-color neighbors
        has_hz = False
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(lin, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            # sums_j = total positive out-slope from j
            sums_j = 0.0
            zj = zh[j]
            if zj < zh[lin]:
            	has_hz = True
            for kk in range(4):
                jj = flow.neighbourer_flat.neighbour(j, kk)
                if jj == -1 or flow.neighbourer_flat.nodata(jj): 
                    continue
                sums_j += slope_pos(zj, zh[jj])

            if sums_j > 0.0:
                acc += slope_pos(zj, zh[lin]) / sums_j * Q[j]
        if has_hz == False:
        	zh[lin] += 2e-3 + ti.random() * 1e-2

        # Optional SOR to converge faster (omega in (1,2))
        omega = 1.
        Q[lin] = (1.0 - omega) * Q[lin] + omega * acc