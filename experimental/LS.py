import taichi as ti
import taichi.math as tm

# Global constants
MANNINGS = 0.033
GRAVITY = 9.81
HFLOW_THRESHOLD = 1e-3
DX = 1.0
FROUDE_LIMIT = 1.0
EDGESLOPE = 1e-2
NX = 0
NY = 0


@ti.kernel
def flow_route(
    hw: ti.template(),
    elev: ti.template(),
    qy: ti.template(),
    qx: ti.template(),
    flow_timestep: float,
):
    for i, j in elev:
        if i == 0 or j == 0:
            continue

        if elev[i, j] > -9999:  # to stop moving water into -9999's on elev
            # routing in y direction
            if (hw[i, j] > 0 or hw[i - 1, j] > 0) and elev[i - 1, j] > -9999:
                hflow = tm.max(
                    elev[i, j] + hw[i, j], elev[i - 1, j] + hw[i - 1, j]
                ) - tm.max(elev[i - 1, j], elev[i, j])

                if hflow > HFLOW_THRESHOLD:
                    tempslope = (
                        (elev[i - 1, j] + hw[i - 1, j]) - (elev[i, j] + hw[i, j])
                    ) / DX

                    if i == elev.shape[0] - 1:  # imax equivalent
                        tempslope = EDGESLOPE
                    if i <= 2:
                        tempslope = 0 - EDGESLOPE

                    qy[i, j] = (
                        qy[i, j] - (GRAVITY * hflow * flow_timestep * tempslope)
                    ) / (
                        1
                        + GRAVITY
                        * hflow
                        * flow_timestep
                        * (MANNINGS * MANNINGS)
                        * ti.abs(qy[i, j])
                        / ti.pow(hflow, (10.0 / 3.0))
                    )

                    # FROUDE NUMBER CHECKS
                    if (
                        qy[i, j] > 0
                        and (qy[i, j] / hflow) / ti.sqrt(GRAVITY * hflow) > FROUDE_LIMIT
                    ):
                        qy[i, j] = hflow * (ti.sqrt(GRAVITY * hflow) * FROUDE_LIMIT)

                    if (
                        qy[i, j] < 0
                        and ti.abs(qy[i, j] / hflow) / ti.sqrt(GRAVITY * hflow)
                        > FROUDE_LIMIT
                    ):
                        qy[i, j] = 0 - (
                            hflow * (ti.sqrt(GRAVITY * hflow) * FROUDE_LIMIT)
                        )

                    # DISCHARGE MAGNITUDE/TIMESTEP CHECKS
                    if qy[i, j] > 0 and (qy[i, j] * flow_timestep / DX) > (
                        hw[i, j] / 4
                    ):
                        qy[i, j] = ((hw[i, j] * DX) / 5) / flow_timestep

                    if qy[i, j] < 0 and ti.abs(qy[i, j] * flow_timestep / DX) > (
                        hw[i - 1, j] / 4
                    ):
                        qy[i, j] = 0 - ((hw[i - 1, j] * DX) / 5) / flow_timestep
                else:
                    qy[i, j] = 0

            # routing in the x direction
            if (hw[i, j] > 0 or hw[i, j - 1] > 0) and elev[i, j - 1] > -9999:
                hflow = tm.max(
                    elev[i, j] + hw[i, j], elev[i, j - 1] + hw[i, j - 1]
                ) - tm.max(elev[i, j], elev[i, j - 1])

                if hflow > HFLOW_THRESHOLD:
                    tempslope = (
                        (elev[i, j - 1] + hw[i, j - 1]) - (elev[i, j] + hw[i, j])
                    ) / DX

                    if j == elev.shape[1] - 1:  # jmax equivalent
                        tempslope = EDGESLOPE
                    if j <= 2:
                        tempslope = 0 - EDGESLOPE

                    qx[i, j] = (
                        qx[i, j] - (GRAVITY * hflow * flow_timestep * tempslope)
                    ) / (
                        1
                        + GRAVITY
                        * hflow
                        * flow_timestep
                        * (MANNINGS * MANNINGS)
                        * ti.abs(qx[i, j])
                        / ti.pow(hflow, (10.0 / 3.0))
                    )

                    # FROUDE NUMBER CHECKS
                    if (
                        qx[i, j] > 0
                        and (qx[i, j] / hflow) / ti.sqrt(GRAVITY * hflow) > FROUDE_LIMIT
                    ):
                        qx[i, j] = hflow * (ti.sqrt(GRAVITY * hflow) * FROUDE_LIMIT)

                    if (
                        qx[i, j] < 0
                        and ti.abs(qx[i, j] / hflow) / ti.sqrt(GRAVITY * hflow)
                        > FROUDE_LIMIT
                    ):
                        qx[i, j] = 0 - (
                            hflow * (ti.sqrt(GRAVITY * hflow) * FROUDE_LIMIT)
                        )

                    # DISCHARGE MAGNITUDE/TIMESTEP CHECKS
                    if qx[i, j] > 0 and (qx[i, j] * flow_timestep / DX) > (
                        hw[i, j] / 4
                    ):
                        qx[i, j] = ((hw[i, j] * DX) / 5) / flow_timestep

                    if qx[i, j] < 0 and ti.abs(qx[i, j] * flow_timestep / DX) > (
                        hw[i, j - 1] / 4
                    ):
                        qx[i, j] = 0 - ((hw[i, j - 1] * DX) / 5) / flow_timestep
                else:
                    qx[i, j] = 0


@ti.kernel
def depth_update(
    hw: ti.template(),
    elev: ti.template(),
    qy: ti.template(),
    qx: ti.template(),
    flow_timestep: float,
):
    for i, j in elev:
        if i == NY - 1 or j == NX - 1:
            continue

        # update water depths
        hw[i, j] += (
            flow_timestep * (qy[i + 1, j] - qy[i, j] + qx[i, j + 1] - qx[i, j]) / DX
        )
