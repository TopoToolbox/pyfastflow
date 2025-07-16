"""
Vectorized 2D grid neighbouring operations for flow calculations.

Provides efficient grid navigation with different boundary conditions:
- Normal: flow stops at boundaries
- Periodic EW: wraps around East-West borders
- Periodic NS: wraps around North-South borders

Author: B.G.
"""

import taichi as ti
from . import constants as cte
from .constants import BOUND_MODE



#########################################
###### GENERAL UTILITIES ################
#########################################

@ti.func
def rc_from_i(i:ti.i32):
	"""
	Convert vectorized index to row,col coordinates.
	
	Args:
		i: Vectorized grid index
		
	Returns:
		tuple: (row, col) coordinates
		
	Author: B.G.
	"""
	return i // cte.NX, i % cte.NX

@ti.func
def i_from_rc(row:ti.i32, col:ti.i32):
	"""
	Convert row,col coordinates to vectorized index.
	
	Args:
		row: Row coordinate
		col: Column coordinate
		
	Returns:
		int: Vectorized grid index
		
	Author: B.G.
	"""
	return row * cte.NX + col


@ti.func
def is_on_edge(i:ti.i32):
	"""
	Check if node is on grid boundary.
	
	Args:
		i: Vectorized grid index
		
	Returns:
		bool: True if node is on any edge
		
	Author: B.G.
	"""
	res = False
	tx,ty = rc_from_i(i)
	
	if(tx == 0 or tx == cte.NX-1 or ty ==0 or ty == cte.NY -1):
		res = True

	return res

@ti.func
def which_edge(i:ti.i32) -> ti.u8:
	"""
	Classify edge position: 0=interior, 1-8=specific edge/corner.
	
	Layout: |1|2|2|2|3|
	        |4|0|0|0|5|
	        |4|0|0|0|5|
	        |4|0|0|0|5|
	        |6|7|7|7|8|
	"""
	
	res:ti.u8 = 0

	tx,ty = rc_from_i(i)

	
	if (i==0):
		res = 1
	elif (i<cte.NX-1):
		res = 2
	elif (i==cte.NX-1):
		res = 3
	elif (i < cte.NX*cte.NY - cte.NX):
		if (tx == 0):
			res = 4
		elif (tx == cte.NX-1):
			res = 5
	elif ty == cte.NY-1:
		if (tx == 0):
			res = 6
		elif (tx == cte.NX-1):
			res = 8
		else:
			res = 7

	return res


#########################################
###### NORMAL BOUNDARIES ################
#########################################

# Raw neighbouring functions - no boundary checks
@ti.func
def top_n(i:ti.i32):
	"""Get top neighbor index."""
	return i-cte.NX

@ti.func
def left_n(i:ti.i32):
	"""Get left neighbor index."""
	return i-1

@ti.func
def right_n(i:ti.i32):
	"""Get right neighbor index."""
	return i+1

@ti.func
def bottom_n(i:ti.i32):
	"""Get bottom neighbor index."""
	return i+cte.NX



@ti.func
def validate_link_n(i:ti.i32, tdir:ti.template()):
	"""Check if link is valid for normal boundaries (flow stops at edges)."""
	edge = which_edge(i)
	res = True
	if edge > 0:
		if (tdir == 0 and edge <= 3):  # Top direction blocked at top edge
			res = False
		elif (tdir == 1 and (edge == 1 or edge == 4 or edge == 6)):  # Left direction blocked at left edge
			res = False
		elif (tdir==2 and (edge == 3 or edge == 5 or edge == 8)):  # Right direction blocked at right edge
			res = False
		elif (tdir==3 and (edge >= 6)):  # Bottom direction blocked at bottom edge
			res = False
	return res


@ti.func
def neighbour_n(i:ti.i32, tdir:ti.template()):
	"""Get neighbor with normal boundary validation."""
	j:ti.i32 = top_n(i) if tdir==0 else (left_n(i) if tdir==1 else (right_n(i) if tdir==2 else bottom_n(i)))
	return j if validate_link_n(i,tdir) else -1

@ti.func
def can_leave_domain_n(i:ti.i32):
	"""Check if flow can leave domain at this node."""
	return which_edge(i) > 0


#########################################
###### PERIODIC EW BOUNDARIES ###########
#########################################

# Raw neighbouring functions - periodic East-West wrapping
@ti.func
def top_pew(i:ti.i32):
	"""Get top neighbor index."""
	return i-cte.NX

@ti.func
def left_pew(i:ti.i32):
	"""Get left neighbor with EW wrapping."""
	row, col = rc_from_i(i)
	return i-1 if col > 0 else i + cte.NX - 1

@ti.func
def right_pew(i:ti.i32):
	"""Get right neighbor with EW wrapping."""
	row, col = rc_from_i(i)
	return i+1 if col < cte.NX-1 else i - cte.NX + 1

@ti.func
def bottom_pew(i:ti.i32):
	"""Get bottom neighbor index."""
	return i+cte.NX



@ti.func
def validate_link_pew(i:ti.i32, tdir:ti.template()):
	"""Check if link is valid for periodic EW boundaries (only NS blocked)."""
	edge = which_edge(i)
	res = True
	if edge > 0:
		if (tdir == 0 and edge <= 3):  # Top direction blocked at top edge
			res = False
		elif (tdir==3 and (edge >= 6)):  # Bottom direction blocked at bottom edge
			res = False
	return res

@ti.func
def neighbour_pew(i:ti.i32, tdir:ti.template()):
	"""Get neighbor with periodic EW boundary validation."""
	j:ti.i32 = top_pew(i) if tdir==0 else (left_pew(i) if tdir==1 else (right_pew(i) if tdir==2 else bottom_pew(i)))
	return j if validate_link_pew(i,tdir) else -1

@ti.func
def can_leave_domain_pew(i:ti.i32):
	"""Check if flow can leave domain (only through top/bottom edges)."""
	edge = which_edge(i)
	return edge == 2 or edge == 7


#########################################
###### PERIODIC NS BOUNDARIES ###########
#########################################

# Raw neighbouring functions - periodic North-South wrapping
@ti.func
def top_pns(i:ti.i32):
	"""Get top neighbor with NS wrapping."""
	row, col = rc_from_i(i)
	return i-cte.NX if row > 0 else i + cte.NX * (cte.NY - 1)

@ti.func
def left_pns(i:ti.i32):
	"""Get left neighbor index."""
	return i-1

@ti.func
def right_pns(i:ti.i32):
	"""Get right neighbor index."""
	return i+1

@ti.func
def bottom_pns(i:ti.i32):
	"""Get bottom neighbor with NS wrapping."""
	row, col = rc_from_i(i)
	return i+cte.NX if row < cte.NY-1 else i - cte.NX * (cte.NY - 1)



@ti.func
def validate_link_pns(i:ti.i32, tdir:ti.template()):
	"""Check if link is valid for periodic NS boundaries (only EW blocked)."""
	edge = which_edge(i)
	res = True
	if edge > 0:
		if (tdir == 1 and (edge == 1 or edge == 4 or edge == 6)):  # Left direction blocked at left edge
			res = False
		elif (tdir==2 and (edge == 3 or edge == 5 or edge == 8)):  # Right direction blocked at right edge
			res = False
	return res

@ti.func
def neighbour_pns(i:ti.i32, tdir:ti.template()):
	"""Get neighbor with periodic NS boundary validation."""
	j:ti.i32 = top_pns(i) if tdir==0 else (left_pns(i) if tdir==1 else (right_pns(i) if tdir==2 else bottom_pns(i)))
	return j if validate_link_pns(i,tdir) else -1

@ti.func
def can_leave_domain_pns(i:ti.i32):
	"""Check if flow can leave domain (only through left/right edges)."""
	edge = which_edge(i)
	return edge == 4 or edge == 5




#########################################
###### CUSTOMS BOUNDARIES ###############
#########################################

# Custom = per node field of boundary codes for fine graine management
# NOTE: the boundary field is kept as a global variable in constant to keep consistent exposed interface 
# Boundary codes:
## 0: No Data
## 1: normal node (cannot leave the domain)
## 3: can leave the domain
## 7: can only enter (special boundary for hydro - will act as normal node for the rest)
## 9: periodic (! risky, make sure you have the opposite direction adn are on a border)


# Raw neighbouring functions - custom wrapping
@ti.func
def top_custom(i:ti.i32):
	"""Get top neighbor with NS wrapping."""
	node = -1 
	tb = cte.boundaries[i]
	if(tb==9):
		node = top_pns(i)
	elif(tb>0):
		node = top_n(i)
	if (cte.boundaries[node] == 0):
		node = -1
	return node


@ti.func
def left_custom(i:ti.i32):
	"""Get left neighbor index."""
	node = -1 
	tb = cte.boundaries[i]
	if(tb==9):
		node = left_pew(i)
	elif(tb>0):
		node = left_n(i)
	if (cte.boundaries[node] == 0):
		node = -1
	return node

@ti.func
def right_custom(i:ti.i32):
	"""Get right neighbor index."""
	node = -1 
	tb = cte.boundaries[i]
	if(tb==9):
		node = right_pew(i)
	elif(tb>0):
		node = right_n(i)
	if (cte.boundaries[node] == 0):
		node = -1
	return node

@ti.func
def bottom_custom(i:ti.i32):
	"""Get bottom neighbor index."""
	node = -1 
	tb = cte.boundaries[i]
	if(tb==9):
		node = bottom_pns(i)
	elif(tb>0):
		node = bottom_n(i)
	if (cte.boundaries[node] == 0):
		node = -1
	return node




@ti.func
def validate_link_custom(i:ti.i32, tdir:ti.template()):
	"""Check if link is valid for periodic NS boundaries (only EW blocked)."""
	tb = cte.boundaries[i]
	res = True
	if(tb == 0):
		res = False
	elif (tb != 9):
		res = validate_link_n(i,tdir)
	else:
		edge = which_edge(i)
		if(edge<=3 or edge>=6):
			res = validate_link_pns(i,tdir)
		else:
			res = validate_link_pew(i,tdir)
	return res

@ti.func
def neighbour_custom(i:ti.i32, tdir:ti.template()):
	"""Get neighbor with periodic NS boundary validation."""
	j:ti.i32 = top_custom(i) if tdir==0 else (left_custom(i) if tdir==1 else (right_custom(i) if tdir==2 else bottom_custom(i)))
	return j if validate_link_custom(i,tdir) else -1

@ti.func
def can_leave_domain_custom(i:ti.i32):
	"""Check if flow can leave domain (only through left/right edges)."""
	return tb == 3




#########################################
###### EXPOSED FUNCTIONS ################
#########################################

# Main API functions - automatically switch based on BOUND_MODE
@ti.func
def top(i:ti.i32):
	"""Get top neighbor - switches between boundary modes."""
	return top_n(i) if ti.static(BOUND_MODE == 0) else (top_pew(i) if ti.static(BOUND_MODE == 1) else (top_pns(i) if ti.static(BOUND_MODE == 2) else (top_custom(i) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def left(i:ti.i32):
	"""Get left neighbor - switches between boundary modes."""
	return left_n(i) if ti.static(BOUND_MODE == 0) else (left_pew(i) if ti.static(BOUND_MODE == 1) else (left_pns(i) if ti.static(BOUND_MODE == 2) else (left_custom(i) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def right(i:ti.i32):
	"""Get right neighbor - switches between boundary modes."""
	return right_n(i) if ti.static(BOUND_MODE == 0) else (right_pew(i) if ti.static(BOUND_MODE == 1) else (right_pns(i) if ti.static(BOUND_MODE == 2) else (right_custom(i) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def bottom(i:ti.i32):
	"""Get bottom neighbor - switches between boundary modes."""
	return bottom_n(i) if ti.static(BOUND_MODE == 0) else (bottom_pew(i) if ti.static(BOUND_MODE == 1) else (bottom_pns(i) if ti.static(BOUND_MODE == 2) else (bottom_custom(i) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def validate_link(i:ti.i32, tdir:ti.template()):
	"""Validate link direction - switches between boundary modes."""
	return validate_link_n(i,tdir) if ti.static(BOUND_MODE == 0) else (validate_link_pew(i,tdir) if ti.static(BOUND_MODE == 1) else (validate_link_pns(i,tdir) if ti.static(BOUND_MODE == 2) else (validate_link_custom(i,tdir) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def neighbour(i:ti.i32, tdir:ti.template()):
	"""
	Get validated neighbor - switches between boundary modes.
	
	Args:
		i: Vectorized grid index
		tdir: Direction template (0=top, 1=left, 2=right, 3=bottom)
		
	Returns:
		int: Neighbor index if valid, -1 if blocked by boundary
		
	Author: B.G.
	"""
	return neighbour_n(i,tdir) if ti.static(BOUND_MODE == 0) else (neighbour_pew(i,tdir) if ti.static(BOUND_MODE == 1) else (neighbour_pns(i,tdir) if ti.static(BOUND_MODE == 2) else (neighbour_custom(i,tdir) if ti.static(BOUND_MODE == 3) else -1)))

@ti.func
def can_leave_domain(i:ti.i32):
	"""
	Check if flow can leave domain - switches between boundary modes.
	
	Args:
		i: Vectorized grid index
		
	Returns:
		bool: True if flow can leave domain at this node
		
	Author: B.G.
	"""
	return can_leave_domain_n(i) if ti.static(BOUND_MODE == 0) else (can_leave_domain_pew(i) if ti.static(BOUND_MODE == 1) else (can_leave_domain_pns(i) if ti.static(BOUND_MODE == 2) else (can_leave_domain_custom(i) if ti.static(BOUND_MODE == 3) else -1)))


