import numpy as np
import taichi as ti

from fastflow.unified_fields import UnifiedFlowFields
from fastflow.compute_receivers import compute_receivers


def check_tree_accum_down(uf):

    print("Check check_tree_accum_down")
    compute_receivers(uf)

def check_tree_accum_up(uf):
    pass

def main():

    ti.init(arch=ti.gpu, debug = False)


    print("Generating terrain...")
    z = np.load("pnoise.npy")[:1000, :1000]
    z = (z - z.min()) / (z.max() - z.min()) * 1000

    # Boundary conditions
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 0.15
    z = np.maximum(z, 0.01)

    unified_fields = UnifiedFlowFields(z.size)
    unified_fields.load_terrain(z)
    unified_fields.set_boundary_edges()

    
    check_tree_accum_down(unified_fields)
    



if __name__ == "__main__":
    main()