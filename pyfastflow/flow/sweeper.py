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
def build_S_var(S: ti.template(), var:ti.template()):
    for i in S:
        S[i] = var[i]   # stationary source


@ti.kernel
def sweep_color(Q: ti.template(), zh: ti.template(), S: ti.template(),
                parity: ti.i32, omega:ti.f32):
    for i in zh:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        iy,ix = flow.neighbourer_flat.rc_from_i(i)
        if ((ix + iy) & 1) != parity:   # cheap parity
            continue

        acc = S[i]  # <-- source added every sweep

        # gather influx from opposite-color neighbors
        has_hz = False
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            # sums_j = total positive out-slope from j
            sums_j = 0.0
            zj = zh[j]
            if zj < zh[i]:
            	has_hz = True
            for kk in range(4):
                jj = flow.neighbourer_flat.neighbour(j, kk)
                if jj == -1 or flow.neighbourer_flat.nodata(jj): 
                    continue
                sums_j += slope_pos(zj, zh[jj])

            if sums_j > 0.0:
                acc += slope_pos(zj, zh[i]) / sums_j * Q[j]

        # if has_hz == False:
        # 	zh[i] += 1e-3 + ti.random() * 1e-3

        # Optional SOR to converge faster (omega in (1,2))
        # omega = 0.2
        Q[i] = (1.0 - omega) * Q[i] + omega * acc


@ti.kernel
def sweep_Qapp_tiled_init(Q: ti.template(), Qapp: ti.template(), z:ti.template(), h: ti.template(), S: ti.template(), tyler:ti.template()):

    for i in z:
        Qapp[i] = S[i]  # <-- Local source added every sweep/acc
        

    for i in z:

        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        

        sums = 0. 

        # gather influx from opposite-color neighbors
        has = False
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            ts = slope_pos(z[i]+h[i], z[j]+h[j])

            if(tyler[j] != tyler[i] and ts > 0):
                has = True
            sums += ti.max(ts,0.)

        if(has and sums>0):

            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if j == -1 or flow.neighbourer_flat.nodata(j): 
                    continue
                ts = slope_pos(z[i]+h[i], z[j]+h[j])

                if(tyler[j] != tyler[i] and ts > 0):
                    ti.atomic_add(Qapp[j], Q[i]*ts/sums)

@ti.kernel
def sweep_Qapp_tiled_iter(Q: ti.template(), Qapp: ti.template(), Qtemp: ti.template(), z:ti.template(), h: ti.template(), tyler:ti.template()):

    for i in Qtemp:
        Qtemp[i] = Qapp[i]

    for i in z:

        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        
        sums = 0. 

        # gather influx from opposite-color neighbors
        has = False
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            ts = slope_pos(z[i]+h[i], z[j]+h[j])
            sums += ti.max(ts,0.)

        if(sums>0):

            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if j == -1 or flow.neighbourer_flat.nodata(j): 
                    continue
                ts = slope_pos(z[i]+h[i], z[j]+h[j])

                if(tyler[j] == tyler[i] and ts > 0):
                    ti.atomic_add(Qtemp[j], Q[i]*ts/sums)
                else:
                    ti.atomic_add(Qtemp[i], -Q[i]*ts/sums)
        else:
            h[i] += 1e-3 + ti.random()*5e-3
            ti.atomic_add(Qtemp[i], -Q[i])

    for i in Qtemp:
        Q[i] = Qtemp[i]


#NEXT STEP: SWEEP PER TILE
# 


@ti.kernel
def sweep_sweep(Q: ti.template(), Q_: ti.template(), zh: ti.template(), S: ti.template(), omega:ti.f32):

    for i in zh:
        Q_[i] = S[i]

    for i in zh:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
            
        sums_j = 0.
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            sums_j += slope_pos(zh[i],zh[j])

        if sums_j > 0.0:
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if j == -1 or flow.neighbourer_flat.nodata(j): 
                    continue
                ti.atomic_add(Q_[j], slope_pos(zh[i],zh[j]) / sums_j * Q[i])

        # if has_hz == False:
        else:
            zh[i] += 1e-3 + ti.random() * 1e-3

        # Optional SOR to converge faster (omega in (1,2))
        # omega = 0.2
        
    for i in zh:
        Q[i] = (1.0 - omega) * Q[i] + omega * Q_[i]

@ti.kernel
def sweep_sweep_tiled_iter(Q: ti.template(), Q_: ti.template(), zh: ti.template(), S: ti.template(), tyler:ti.template(), omega:ti.f32):

    for i in zh:
        Q_[i] = S[i]

    for i in zh:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
            
        sums_j = 0.
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            sums_j += slope_pos(zh[i],zh[j])

        if sums_j > 0.0:
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if j == -1 or flow.neighbourer_flat.nodata(j): 
                    continue
                if tyler[j] == tyler[i]:
                    ti.atomic_add(Q_[j], slope_pos(zh[i],zh[j]) / sums_j * Q[i])


        # if has_hz == False:
        else:
            zh[i] += 1e-3 + ti.random() * 1e-3

        # Optional SOR to converge faster (omega in (1,2))
        # omega = 0.2
        
    for i in zh:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
            
        sums_j = 0.
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            sums_j += slope_pos(zh[i],zh[j])

        if sums_j > 0.0:
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if j == -1 or flow.neighbourer_flat.nodata(j): 
                    continue
                if tyler[j] != tyler[i]:
                    Q[i]-= slope_pos(zh[i],zh[j]) / sums_j * Q[i]


        

    for i in zh:
        Q[i] = (1.0 - omega) * Q[i] + omega * Q_[i]

