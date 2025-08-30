import random

import dg2
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

import pyfastflow as pf

ti.init(
    arch=ti.gpu, cpu_max_num_threads=1, debug=False
)  # Using default ti.i32 to suppress warnings


# --- Main execution ---
nx, ny = 1024, 1024
N = nx * ny
dx = 100.0
pf.constants.KR = 4e-5
pf.constants.KD = 100.0
Urate = 1e-3
pf.constants.DT_SPL = 1e3


# pf.erodep.SPL.DT = dt

# np.random.seed(42)

print("Generating terrain...")
noise = dg2.PerlinNoiseF32(
    frequency=0.1, amplitude=1.0, octaves=6, seed=random.randint(1, 32000000)
)
z_np = noise.create_noise_grid(nx, ny, 0, 0, 100, 100).as_numpy()
# z_np    +=  np.random.rand(ny,nx)
z_np = (z_np - z_np.min()) / (z_np.max() - z_np.min()) * 100
z_np[[0, -1], :] = -5


noiseU = dg2.PerlinNoiseF32(
    frequency=0.02, amplitude=1.0, octaves=3, seed=random.randint(1, 32000000)
)
U = noiseU.create_noise_grid(nx, ny, 0, 0, 100, 100).as_numpy()
U -= U.min()
U /= U.max()
# U-=0.2
U *= 1e-3
U[: round(ny / 4), :] = -1e-4
U[round(3 * ny / 4) :, :] = -1e-4
# U-=2e-4
# U[U<0] = 0

# plt.imshow(z_np)
# plt.colorbar()
# plt.show()
# # quit()


router = pf.flow.FlowRouter(
    nx,
    ny,
    dx,
    boundary_mode="periodic_EW",
    boundaries=None,
    lakeflow=True,
    stochastic_receivers=False,
)
router.set_z(z_np.ravel())

alpha_ = ti.field(ti.f32, shape=(nx * ny))
alpha__ = ti.field(ti.f32, shape=(nx * ny))
Qs = ti.field(ti.f32, shape=(nx * ny))
Urate = ti.field(ti.f32, shape=(nx * ny))
Urate.from_numpy(U.ravel())

fig, ax = plt.subplots()

im = ax.imshow(router.get_Z(), cmap="terrain", vmax=50.0)
# im = ax.imshow(router.get_Z(), cmap = 'Blues', vmax = 6)
# im = ax.imshow(router.get_Z(), cmap = 'RdBu_r', vmin = -50, vmax = 50)
# im = ax.imshow(router.get_Z(), cmap = 'RdBu_r',vmin=-1e-3, vmax = 1e-3)

# fig.show()

viewer = pf.visu.SurfaceViewer(z_np)

cumdz = np.zeros_like(U)
it = 0
while True:
    it += 1
    for i in range(10):
        router.compute_receivers()
        router.reroute_flow()
        router.fill_z()
        router.accumulate_constant_Q(1.0)
        # pf.erodep.block_uplift(router.z,Urate)
        pf.erodep.ext_uplift_bl(router.z, Urate)

        # pf.erodep.SPL(router, alpha_, alpha__, kr)
        # beef = router.get_Z()
        pf.erodep.SPL_transport(router, alpha_, alpha__, Qs)
        # cumdz+=router.get_Z() - beef

    tZ = router.get_Z()
    # tZ = np.log10(router.get_Q())

    viewer.update_surface(tZ)
    viewer.render_frame()

    im.set_data(tZ)
    im.set_clim(tZ.min(), tZ.max())
    # im.set_data(router.get_Z() - beef)
    # fig.canvas.draw_idle()
    # fig.canvas.start_event_loop(0.01)

    if it == 500:
        print("saving")
        np.save("test.npy", tZ)
