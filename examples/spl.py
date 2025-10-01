import pyfastflow as pff
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np

ti.init(ti.gpu)

@ti.kernel
def init_k(a:ti.template(), b:ti.template(), z:ti.template(), dt:ti.template(), K:ti.template(), dx:ti.template(),A:ti.template(), m:ti.template()):
	for i in a:
		k = K*dt*(A[i]**m)/dx
		a[i] = k/(1 + k)
		b[i] = z[i]/(1 + k)

@ti.kernel
def uplift(z:ti.template(), rec:ti.template(), rate:ti.f32, dt:ti.f32):

	for i in z:
		if rec[i] != i:
			z[i] += rate*dt

# ti.init(ti.gpu)

nx, ny = 512,512
a = pff.pool.get_temp_field(ti.f32, (ny*nx))
b = pff.pool.get_temp_field(ti.f32, (ny*nx))

dx = 32.
dt = 1e3
K = 1e-3
m = 0.45

znp = pff.noise.perlin_noise(nx, ny, frequency = 4.0, octaves = 12, persistence = 0.4, amplitude = 100.0, seed = 42)

# plt.imshow(znp, cmap = 'terrain')
# plt.show()

grid = pff.grid.Grid(nx,ny,dx,znp)

router = pff.flow.FlowRouter(grid)
for i in range(50):
	router.compute_receivers()
	router.reroute_flow()
	router.fill_z()
	uplift(grid.z.field, router.receivers.field,1e-3,dt)
	router.accumulate_constant_Q(1.)

	init_k(a.field, b.field, grid.z.field, dt, K, dx, router.Q.field, m)
	# b.field.copy_from(grid.z.field)

	router.propagate_upstream_affine_var(a, b, grid.z)

fig,ax = plt.subplots(1,2)
ax[0].imshow(znp, cmap = 'terrain')
ax[1].imshow(grid.get_z(), cmap = 'terrain')
plt.show()


# plt.imshow(np.log10(router.get_Q()), cmap = 'magma')
# plt.imshow(grid.get_z(), cmap = 'terrain')
# plt.show()
