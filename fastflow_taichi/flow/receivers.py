import taichi as ti
from . import constants as cte
from . import neighbourer_flat as nei

@ti.kernel
def compute_receivers(z: ti.template(), receivers: ti.template(), gradient: ti.template()):
    """
    Compute steepest descent receivers for each node in the grid.
    
    Args:
        z: Elevation field
        receivers: Output array for receiver indices
        gradient: Output array for steepest gradients
        
    Author: B.G.
    """
    for i in z:
        
        r:ti.i32 = i
        sr:ti.f32 = 0.
        for k in ti.static(range(4)):
            # Fetch the receiver and compute gradient
            tr    :ti.i32  = nei.neighbour(i,k)
            valid :ti.u1   = (tr != -1)
            tsr   :ti.f32  = (z[i]-z[tr])/cte.DX if valid else -1e9
            
            # Apply stocastic neighbouring if needed
            if(ti.static(cte.RAND_RCV==1)):
                tsr *= ti.random()

            # Keep if steepest
            valid  = (valid and tsr>sr)
            sr     = tsr if valid else sr 
            r      = tr if valid else r

        # Register
        receivers[i] = r
        gradient[i]  = sr
