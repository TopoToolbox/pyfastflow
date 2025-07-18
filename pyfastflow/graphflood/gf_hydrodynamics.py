"""
Hydrodynamic computation kernels for GraphFlood shallow water flow.

This module implements the core hydrodynamic algorithms for 2D shallow water
flow simulation using GPU-accelerated Taichi kernels. It provides methods for
discharge diffusion and Manning's equation-based flow depth updates.

Key algorithms:
- Discharge diffusion for multiple flow path simulation
- Manning's equation for flow resistance and depth updates
- Integration with FastFlow's flow routing system

Based on methods from Gailleton et al. 2024 for efficient GPU-based
shallow water flow modeling.

Author: B.G.
"""

import taichi as ti
from .. import constants as cte
import pyfastflow.flow as flow 

@ti.kernel
def diffuse_Q_constant_prec(zh:ti.template(), Q:ti.template(), temp:ti.template()):
	"""
	Diffuse discharge field to simulate multiple flow paths.
	
	Redistributes discharge from each cell to its neighbors based on slope
	gradients, creating a more realistic multiple flow direction pattern
	from the original single flow direction (SFD) routing.
	
	The method:
	1. Initializes precipitation input for each cell
	2. Computes slope-weighted diffusion to neighbors
	3. Redistributes discharge proportionally to slope gradients
	
	Args:
		zh (ti.template): Combined topography + water depth field
		Q (ti.template): Discharge field to diffuse
		temp (ti.template): Temporary field for intermediate calculations
	
	Author: B.G.
	"""

	# Initialize precipitation input and handle boundary conditions
	for i in Q:
		temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
		if flow.neighbourer_flat.can_leave_domain(i):
			Q[i] = 0.  # Set boundary cells to zero discharge
	
	# Diffuse discharge based on slope gradients
	for i in zh:
		# Skip boundary cells
		if flow.neighbourer_flat.can_leave_domain(i):
			continue

		# Calculate total slope gradient sum for normalization
		sums = 0.
		for k in range(4):  # Check all 4 neighbors
			j = flow.neighbourer_flat.neighbour(i,k)
			sums += ti.max(0., ((zh[i]-zh[j])/cte.DX) if j!=-1 else 0.)
		
		# Skip cells with no downslope neighbors
		if(sums == 0.):
			continue

		# Distribute discharge proportionally to slope gradients
		for k in range(4):
			j = flow.neighbourer_flat.neighbour(i,k)
			tS = ti.max(0., ((zh[i]-zh[j])/cte.DX) if j!=-1 else 0.)
			ti.atomic_add(temp[j], tS/sums * Q[i])  # Add proportional discharge

	# Update discharge field with diffused values
	for i in Q:
		Q[i] = temp[i]
		

@ti.kernel
def graphflood_core_cte_mannings(h:ti.template(), zh:ti.template(), dh:ti.template(), rec:ti.template(), Q:ti.template()):
	"""
	Core shallow water flow computation using Manning's equation.
	
	Implements the main hydrodynamic computation for 2D shallow water flow
	using Manning's equation for flow resistance. Updates water depth based
	on discharge input and outflow capacity.
	
	The method:
	1. Computes local slope from flow receivers
	2. Calculates outflow capacity using Manning's equation
	3. Updates water depth based on discharge balance
	4. Ensures non-negative depth values
	5. Updates combined topography + water surface
	
	Based on core methods from Gailleton et al. 2024.
	
	Args:
		h (ti.template): Flow depth field
		zh (ti.template): Combined topography + water depth field
		dh (ti.template): Depth change field for intermediate calculations
		rec (ti.template): Flow receiver field from flow routing
		Q (ti.template): Discharge field
	
	Author: B.G.
	"""

	# Compute depth changes using Manning's equation
	for i in h:
		# Determine local slope
		tS = cte.EDGESW  # Use edge slope for boundary/sink cells
		if(rec[i] != i):  # Interior cells with valid receivers
			tS = ti.max((zh[i]-zh[rec[i]])/cte.DX, 1e-4)  # Slope to receiver (minimum 1e-4)

		# Calculate outflow capacity using Manning's equation
		# Q = (1/n) * A * R^(2/3) * S^(1/2), where R ≈ h for wide channels
		Qo = cte.DX*h[i]**(5./3.)/cte.MANNING * ti.math.sqrt(tS)

		# Update depth based on discharge balance (inflow - outflow)
		tdh = (Q[i] - Qo)/(cte.DX**2) * cte.DT_HYDRO  # Volume change per unit area

		# Apply depth change and ensure non-negative depths
		h[i] += tdh
		if(h[i] < 0):  # Prevent negative depths
			tdh += h[i]  # Adjust change to reach zero depth
		dh[i] = tdh

	# Update water surface elevation (topography + depth)
	for i in h:
		h[i] += dh[i]  # Apply final depth change
		zh[i] += dh[i]  # Update water surface elevation
