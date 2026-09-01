"""


Author: B.G.
"""

import taichi as ti

import pyfastflow.flow as flow

from .. import constants as cte

@ti.kernel
def zh_to_z_h(zh:ti.template(), z:ti.template(), h:ti.template()):
    '''
    Converts back an added h to z into a z and h

    Authors:
        B.G.
    '''

    for i in zh:

        h[i] = zh[i] - z[i]

        if(h[i] < 0):
            z[i] -= h[i]
            h[i] = 0.


@ti.kernel
def locarve(zh: ti.template(), srecs:ti.template()):

    for i in zh:

        if flow.neighbourer_flat.nodata(i) or  flow.neighbourer_flat.can_leave_domain(i):
            continue

        has_hz = False
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j): 
                continue
            zj = zh[j]
            if zj < zh[i]:
                has_hz = True

        if(has_hz):
            continue

        refzh = zh[i]
        lzh   = zh[i]
        N     = 0
        node  = i

        while True:
            N+=1
            node = srecs[node]
            if(node == srecs[node]):
            #     if flow.neighbourer_flat.can_leave_domain(node)==False:
            #         print('???')

                break

            if zh[node] < lzh:
                break

            zh[node] = refzh - N * 2e-3 
            lzh = zh[node]