import taichi as ti
import numpy as np
import scabbard as scb
import pyfastflow as pf
import pyfastflow.flow.constants as cte
import matplotlib.pyplot as plt


MANNING = 0.033
S_out = 1e-2
DT = 1e-3

PREC = 50 *1e-3/3600
ALPHA = 0.


ti.init(ti.gpu, debug = False)

# dem = scb.io.load_raster('/home/bgailleton/Desktop/data/green_river_1.tif')
dem = scb.io.load_raster('/home/bgailleton/Desktop/data/NZ/archive/points_v2_6.tif')

nx,ny=dem.geo.nx,dem.geo.ny
router = pf.flow.FlowRouter(dem.geo.nx, dem.geo.ny, dem.geo.dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = False)
router.set_z(dem.Z.ravel())
router.compute_receivers()
router.reroute_flow()
router.accumulate_constant_Q(PREC, area = True)




h = ti.field(ti.f32, shape = (nx*ny))
h.fill(0.)
dh = ti.field(ti.f32, shape = (nx*ny))
dh.fill(0.)
pQ = ti.field(ti.f32, shape = (nx*ny))
pQ.copy_from(router.Q)


@ti.kernel
def diffuse_Q(zh:ti.template(), Q:ti.template(), temp:ti.template()):

	for i in Q:
		temp[i] = PREC * cte.DX * cte.DX
	
	for i in zh:

		sums = 0.
		for k in range(4):
			j = pf.flow.neighbourer_flat.neighbour(i,k)
			sums += ti.max(0., ((zh[i]-zh[j])/cte.DX) if j!=-1 else 0.)
		
		if(sums == 0.):
			continue

		for k in range(4):
			j = pf.flow.neighbourer_flat.neighbour(i,k)
			tS = ti.max(0., ((zh[i]-zh[j])/cte.DX) if j!=-1 else 0.)
			ti.atomic_add(temp[j], tS/sums * Q[i])
	for i in Q:
		Q[i] = temp[i]

@ti.kernel
def graphflood(h:ti.template(), zh:ti.template(), dh:ti.template(), rec:ti.template(), Q:ti.template()):

	# for i in Q:
	# 	Q[i] = (1-ALPHA)*Q[i] + ALPHA * nQ[i]

	for i in h:

		tS = S_out
		if(rec[i] != i):
			tS = ti.max((zh[i]-zh[rec[i]])/cte.DX, 1e-4)

		Qo = cte.DX*h[i]**(5./3.)/MANNING * ti.math.sqrt(tS)
		# Q[i] = (1-ALPHA)*nQ[i] + ALPHA * Qo
		tdh = (Q[i] - Qo)/(cte.DX**2) * DT

		h[i] += tdh
		if(h[i] < 0):
			tdh += h[i]
		dh[i] = tdh


	for i in h:
		h[i] += dh[i]
		zh[i] += dh[i]


fig,ax = plt.subplots()

im=ax.imshow(h.to_numpy().reshape(ny,nx), cmap = 'Blues', vmin = 0., vmax = 3.)
plt.colorbar(im, label='flow depth')
fig.show()

it = 0
while True:
	it+=1
	# if it==1:
	router.compute_receivers()
	router.reroute_flow()
	pf.flow.fill_z_add_delta(router.z,h,router.z_,router.receivers,router.receivers_,router.receivers__, epsilon=1e-3)

	for i in range(5):
		diffuse_Q(router.z, router.Q, router.Q_)

	graphflood(h,router.z,dh, router.receivers, router.Q)

	if(it % 10 == 0):
		im.set_data(h.to_numpy().reshape(ny,nx))
		# im.set_data(router.Q.to_numpy().reshape(ny,nx))
		fig.canvas.draw_idle()
		fig.canvas.start_event_loop(0.01)