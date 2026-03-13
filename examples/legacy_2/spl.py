import pyfastflow as pff
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np
import time

ti.init(ti.gpu, debug=False)

@ti.kernel
def init_k(a:ti.template(), b:ti.template(), z:ti.template(), dt:ti.template(), K:ti.template(), dx:ti.template(), 
	A:ti.template(), m:ti.template(), rec:ti.template()):
	for i in a:
		k = K*dt*(A[i]**m)/dx
		a[i] = k/(1 + k)
		# if rec[i] != i or True:
		b[i] = z[i]/(1 + k)
		# else:
		# 	b[i] = z[i]

@ti.kernel
def init_k_h(a:ti.template(), b:ti.template(), z:ti.template(), dt:ti.template(), kappa:ti.template(), dx:ti.template(), rec:ti.template()):
	for i in a:
		k = kappa * dt/dx
		a[i] = k/(1 + k)
		# if rec[i] != i:
		b[i] = z[i]/(1 + k)
		# else:
		# 	b[i] = z[i]

@ti.kernel
def uplift(z:ti.template(), rate:ti.template(), dt:ti.f32):

	for i in z:
		z[i] += rate[i]*dt



# ti.init(ti.gpu)

nx, ny = 1024,1024
a = pff.pool.get_temp_field(ti.f32, (ny*nx))
b = pff.pool.get_temp_field(ti.f32, (ny*nx))

dx = 32.
dt = 1e2
K = 1e-5
hillslope = False
kappa = 1e-2
Sc = 0.56
m = 0.45
Urate = np.full((ny,nx), 1e-3, dtype = np.float32)
Urate[[0,-1],:] = 0
rate = ti.field(ti.f32, shape = (ny*nx))
rate.from_numpy(Urate.ravel())

znp = pff.noise.perlin_noise(nx, ny, frequency = 4.0, octaves = 12, persistence = 0.4, amplitude = 10.0, seed = 42)
znp -= znp.min()
znp /= znp.max()
znp*=100
rng = np.random.default_rng(42455896)

# znp = rng.random((ny, nx), dtype=np.float32)

# plt.imshow(znp, cmap = 'terrain')
# plt.show()

grid = pff.grid.Grid(nx,ny,dx,znp, boundary_mode='periodic_EW')

router = pff.flow.FlowRouter(grid)
# router.compute_receivers()
# router.reroute_flow()
# pff.flow.fill_topo.topobehead(router,S_c = 0.1)
# pff.flow.fill_topo.topofill(router,epsilon = 0.001)

# recs = router.receivers.field.to_numpy()
# grad_mag = np.zeros_like(znp)
# mask = recs!=np.arange(ny*nx)
# grad_mag.ravel()[mask] = (grid.get_z().ravel()[mask] - grid.get_z().ravel()[recs[mask]])/dx
# magnitude of gradient (steepest ascent)
# grad_mag = np.sqrt(df_dx**2 + df_dy**2)

# steepest descent value = negative of that
# steepest_descent = np.ab
# print(np.argmax(grad_mag))

# fig,ax = plt.subplots(1,3)
# ax[0].imshow(znp, cmap = 'terrain')
# ax[1].imshow(grid.get_z(), cmap = 'terrain')
# ax[2].imshow(grad_mag, cmap = 'magma', vmin = 0, vmax = 0.6)
# plt.show()
# quit()

#warmup (compilation)
# for i in range(5):
# 	router.compute_receivers()
# 	router.reroute_flow()
# 	router.fill_z()
# 	uplift(grid.z.field, router.receivers.field,Urate,dt)
# 	router.accumulate_constant_Q(1.)
# 	init_k(a.field, b.field, grid.z.field, dt, K, dx, router.Q.field, m,router.receivers.field)
# 	# b.field.copy_from(grid.z.field)
# 	router.propagate_upstream_affine_var(a, b)
# 	grid.z.copy_from(b)
# 	router.fill_z()

# 	init_k_h(a.field, b.field, grid.z.field, dt, kappa, dx, router.receivers.field)
# 	router.propagate_upstream_affine_var(a, b)
# 	grid.z.copy_from(b)
	
# ti.sync()
# print('GO')

fig,ax = plt.subplots()
im = ax.imshow(grid.get_z(), cmap = 'terrain')
plt.colorbar(im, label = 'elevation')
fig.show()

router.compute_receivers()
# recs = router.receivers.to_numpy()
# recs[:nx] = np.arange(nx)
# recs[(nx-1)*ny:nx*ny] = np.arange((nx-1)*ny,nx*ny)
# router.receivers.field.from_numpy(recs)
router.reroute_flow()
router.fill_z()
uplift(grid.z.field, rate ,dt)
router.accumulate_constant_Q(1.)

init_k(a.field, b.field, grid.z.field, dt, K, dx, router.Q.field, m, router.receivers.field)
# b.field.copy_from(grid.z.field)

router.propagate_upstream_affine_var(a, b)
grid.z.copy_from(b)
if (hillslope):
	router.fill_z()
	init_k_h(a.field, b.field, grid.z.field, dt, kappa, dx, router.receivers.field)
	router.propagate_upstream_affine_var(a, b)
	grid.z.copy_from(b)
	pff.flow.fill_topo.topobehead(router,S_c = Sc)

st = time.perf_counter()
for i in range (10):
	for i in range(100):
		router.compute_receivers()
		# recs = router.receivers.to_numpy()
		# recs[:nx] = np.arange(nx)
		# recs[(nx-1)*ny:nx*ny] = np.arange((nx-1)*ny,nx*ny)
		# router.receivers.field.from_numpy(recs)
		router.reroute_flow()
		router.fill_z()
		uplift(grid.z.field, rate ,dt)
		router.accumulate_constant_Q(1.)

		init_k(a.field, b.field, grid.z.field, dt, K, dx, router.Q.field, m, router.receivers.field)
		# b.field.copy_from(grid.z.field)

		router.propagate_upstream_affine_var(a, b)
		grid.z.copy_from(b)
		if (hillslope):
			router.fill_z()
			init_k_h(a.field, b.field, grid.z.field, dt, kappa, dx, router.receivers.field)
			router.propagate_upstream_affine_var(a, b)
			grid.z.copy_from(b)
			pff.flow.fill_topo.topobehead(router,S_c = Sc)

	tz = grid.get_z()
	im.set_data(tz)
	im.set_clim(tz.min(),tz.max())

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.01)

ti.sync()
print('took', - st + time.perf_counter(), "s")


fig,ax = plt.subplots(1,2)
ax[0].imshow(znp, cmap = 'terrain')
ax[1].imshow(grid.get_z(), cmap = 'terrain',vmin = 0)
plt.show()


# plt.imshow(np.log10(router.get_Q()), cmap = 'magma')
# plt.imshow(grid.get_z(), cmap = 'terrain')
# plt.show()
