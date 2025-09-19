"""
High-level FlowRouter class for GPU-accelerated flow routing.

Provides an object-oriented interface for performing hydrological flow routing
on digital elevation models. Handles field allocation, boundary conditions,
and orchestrates the flow routing pipeline including receiver computation,
lake flow processing, and flow accumulation.

Author: B.G.
"""

import math

import taichi as ti

import pyfastflow as pf
import pyfastflow.general_algorithms.util_taichi as ut

# from . import environment as env  # Module not found
from .. import constants as cte
from .. import general_algorithms as gena
from . import downstream_propag as dpr
from . import fill_topo as fl
from . import lakeflow as lf
from . import receivers as rcv
from . import propagate_upstream as pup


class FlowRouter:
    """
    High-level interface for GPU-accelerated flow routing computations.

    Handles field allocation, boundary conditions, and orchestrates the complete
    flow routing pipeline including receiver computation, lake flow processing,
    and flow accumulation using parallel algorithms.

    Author: B.G.
    """

    def __init__(self, grid, lakeflow=True):
        """
        Initialize FlowRouter with grid and configuration parameters using pool-based fields.

        Creates a FlowRouter instance with automatic GPU field management through the pool
        system. Allocates essential fields (Q for flow accumulation, receivers for flow
        routing) from the pool for efficient memory usage.

        Args:
            grid (GridField): The GridField object containing elevation data and grid parameters
            lakeflow (bool): Enable lake flow processing for depression handling (default: True)
                When True, enables reroute_flow() method for handling closed basins
                When False, reroute_flow() will raise RuntimeError

        Note:
            The FlowRouter instance maintains pool-allocated fields that are automatically
            managed. Fields are released when the FlowRouter is destroyed via __del__().

        Author: B.G.
        """

        self.grid = grid

        # Store configuration parameters
        self.lakeflow = lakeflow  # Enable depression handling

        # Flow accumulation fields (primary and ping-pong buffer)
        self.Q = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        # Receiver field - downstream flow direction for each node
        self.receivers = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny)
        )

    @property
    def nx(self):
        return self.grid.nx

    @property
    def ny(self):
        return self.grid.ny

    @property
    def dx(self):
        return self.grid.dx

    @property
    def rshp(self):
        return self.grid.rshp

    def compute_receivers(self):
        """
        Compute steepest descent receivers for all grid nodes.

        Determines the downstream flow direction for each node using the steepest
        descent algorithm. Each node is assigned a receiver (downstream neighbor)
        based on the steepest downhill gradient. Results are stored in self.receivers.

        The algorithm considers all valid neighbors (4 or 8-connectivity depending on
        configuration) and selects the one with the steepest descent. Boundary
        conditions are handled according to the grid's boundary mode.

        Note:
            Results are stored in self.receivers field (pool-allocated).
            No gradient field is computed in current implementation.

        Author: B.G.
        """
        rcv.compute_receivers(self.grid.z.field, self.receivers.field)

    def compute_stochastic_receivers(self):
        """
        Compute stochastic receivers for all grid nodes using probabilistic flow routing.

        Determines downstream flow directions using a stochastic approach where multiple
        downhill neighbors can be selected with probabilities proportional to their
        gradients. This enables uncertainty quantification in flow routing and creates
        more realistic flow patterns in flat areas.

        The algorithm assigns probabilities to all downhill neighbors based on their
        slope steepness, then randomly selects one receiver per node. Multiple calls
        will produce different flow networks for the same topography.

        Note:
            Results are stored in self.receivers field (pool-allocated).
            Requires RAND_RCV constant to be enabled for stochastic behavior.
            Each call produces a different realization of the flow network.

        Author: B.G.
        """
        rcv.compute_receivers_stochastic(self.grid.z.field, self.receivers.field)

    def reroute_flow(self, carve=True):
        """
        Process lake flow to handle depressions and closed basins.

        Implements depression handling algorithms to route flow through
        or around topographic depressions. Supports both carving and
        filling approaches.

        Args:
            carve: Use carving (True) or filling (False) for depression handling
                   Carving creates channels through saddle points
                   Filling jumps flow directly to basin outlets

        Raises:
            RuntimeError: If lakeflow was not enabled during initialization

        Author: B.G.
        
        """
        
        if not self.lakeflow:
            raise RuntimeError("Flow field not compiled for lakeflow")

        # Querying fields from the pool
        bid = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        receivers_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny)
        )
        receivers__ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny)
        )
        z_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        is_border = pf.pool.taipool.get_tpfield(dtype=ti.u1, shape=(self.nx * self.ny))
        outlet = pf.pool.taipool.get_tpfield(dtype=ti.i64, shape=(self.nx * self.ny))
        basin_saddle = pf.pool.taipool.get_tpfield(
            dtype=ti.i64, shape=(self.nx * self.ny)
        )
        basin_saddlenode = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny)
        )
        tag = pf.pool.taipool.get_tpfield(dtype=ti.u1, shape=(self.nx * self.ny))
        tag_ = pf.pool.taipool.get_tpfield(dtype=ti.u1, shape=(self.nx * self.ny))
        change = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=())
        rerouted = pf.pool.taipool.get_tpfield(dtype=ti.u1, shape=(self.nx * self.ny))

        # Call the main lake flow routing algorithm
        lf.reroute_flow(
            bid.field,
            self.receivers.field,
            receivers_.field,
            receivers__.field,
            self.grid.z.field,
            z_.field,
            is_border.field,
            outlet.field,
            basin_saddle.field,
            basin_saddlenode.field,
            tag.field,
            tag_.field,
            change.field,
            rerouted.field,
            carve=carve,
        )

        bid.release()
        receivers_.release()
        receivers__.release()
        z_.release()
        is_border.release()
        outlet.release()
        basin_saddle.release()
        basin_saddlenode.release()
        tag.release()
        tag_.release()
        change.release()
        rerouted.release()

    def accumulate_constant_Q(self, value, area=True):
        """
        Accumulate constant flow values using parallel rake-compress algorithm with pool-based fields.

        Performs flow accumulation where each cell contributes a uniform input value,
        then accumulates downstream following the receiver network. Uses the efficient
        rake-and-compress algorithm for O(log N) parallel computation complexity.

        Args:
            value (float): Constant flow value to accumulate at each node (units depend on area flag)
            area (bool, optional): If True, multiply value by cell area (dx²) to get volumetric flow.
                If False, use raw value for unit-based flow. Default: True.

        Returns:
            None: Results are stored in self.Q field (pool-allocated)

        Note:
            - Uses temporary pool fields for efficient memory management
            - All temporary fields are automatically released after computation
            - Results can be accessed via get_Q() method
            - Requires valid receivers from compute_receivers() or compute_stochastic_receivers()

        Example:
            router.accumulate_constant_Q(1.0, area=True)  # Each cell contributes dx² area
            drainage_area = router.get_Q()  # Get drainage area in m²

        Author: B.G.
        """
        # Calculate number of iterations needed (log₂ of grid size)
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Get temporary fields from pool for rake-compress algorithm
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        # Initialize arrays for rake-compress algorithm
        ndonors.field.fill(0)  # Reset donor counts
        src.field.fill(0)  # Reset ping-pong state

        # Initialize flow values (multiply by area if requested)
        self.Q.field.fill(value * (cte.DX * cte.DX if area else 1.0))

        # Build donor-receiver relationships from receiver array
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)

        # Rake-compress iterations for parallel tree accumulation
        # Each iteration doubles the effective path length being compressed
        for i in range(logn + 1):
            dpr.rake_compress_accum(
                donors.field,
                ndonors.field,
                self.Q.field,
                src.field,
                donors_.field,
                ndonors_.field,
                Q_.field,
                i,
            )

        # Final fuse step to consolidate results from ping-pong buffers
        # Merge accumulated values from working arrays back to primary array
        gena.fuse(self.Q.field, src.field, Q_.field, logn)

        # Release all temporary fields back to pool
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()

    def accumulate_custom_donwstream(self, Acc: ti.template()):
        """
        Acc needs to be accumulated to the OG value to accumulate

        Author: B.G.
        """
        # Calculate number of iterations needed (log₂ of grid size)
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Get temporary fields from pool for rake-compress algorithm
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        # Initialize arrays for rake-compress algorithm
        ndonors.field.fill(0)  # Reset donor counts
        src.field.fill(0)  # Reset ping-pong state

        # Build donor-receiver relationships from receiver array
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)

        # Rake-compress iterations for parallel tree accumulation
        # Each iteration doubles the effective path length being compressed
        for i in range(logn + 1):
            dpr.rake_compress_accum(donors, ndonors, Acc, src, donors_, ndonors_, Q_, i)

        # Final fuse step to consolidate results from ping-pong buffers
        # Merge accumulated values from working arrays back to primary array
        gena.fuse(Acc, src, Q_, logn)

        # Release all temporary fields back to pool
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()

    def accumulate_affine(self, a: float, b:ti.template()):
        """
        Accumulate downstream with affine transform q_out = a*q_in + b.
        TO BE TESTED

        Args:
            a (float): Constant transmission coefficient applied at each node (0<=a<=1 typical)
            b: Taichi field (or TPField) of local source terms per node (same shape as Q)

        Semantics:
            For each node i: q[i] = b[i] + sum_{d in donors(i)} (A_path_to_i_from_d * q[d])
            with A_path_to_i_from_d = a^path_length (constant per-node coefficient).

        Returns:
            None. Results stored in self.Q field.

        Author: B.G. - G.C.
        """
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Temporary fields
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        w = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        w_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))

        # Initialize arrays
        ndonors.field.fill(0)
        src.field.fill(0)

        # Initialize p with local sources b
        b_field = getattr(b, "field", b)
        self.Q.field.copy_from(b_field)

        # Build donors and initialize per-edge weights
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)
        dpr.init_affine_weights(donors.field, ndonors.field, w.field, a)

        # Iterations
        for i in range(logn + 1):
            dpr.rake_compress_accum_affine(
                donors.field,
                ndonors.field,
                self.Q.field,
                src.field,
                donors_.field,
                ndonors_.field,
                Q_.field,
                w.field,
                w_.field,
                i,
                a,
                b_field,
            )

        # Fuse
        gena.fuse(self.Q.field, src.field, Q_.field, logn)

        # Release temporaries
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()
        w.release()
        w_.release()

    def accumulate_affine_with_loss(self, a: float, b:ti.template()):
        """
        Constant-a affine accumulation and per-node loss in one pass.

        Stores q_out in self.Q and returns a numpy array of loss values.
        loss[i] = (1 - a) * incoming_pre_attenuation[i].

        Author: B.G. - G.C.
        """
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Temporaries
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        w = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        w_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        incoming = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        incoming_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        loss = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        # Init
        ndonors.field.fill(0)
        src.field.fill(0)
        incoming.field.fill(0.0)
        incoming_.field.fill(0.0)

        b_field = getattr(b, "field", b)
        self.Q.field.copy_from(b_field)

        # Build donors and init weights
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)
        dpr.init_affine_weights(donors.field, ndonors.field, w.field, a)

        # Iterations
        for i in range(logn + 1):
            dpr.rake_compress_accum_affine_with_incoming(
                donors.field,
                ndonors.field,
                self.Q.field,
                incoming.field,
                src.field,
                donors_.field,
                ndonors_.field,
                Q_.field,
                incoming_.field,
                w.field,
                w_.field,
                i,
                a,
                b_field,
            )

        # Fuse Q and incoming
        gena.fuse(self.Q.field, src.field, Q_.field, logn)
        gena.fuse(incoming.field, src.field, incoming_.field, logn)

        # Compute loss
        # Here a is scalar, so loss = (1-a)*incoming
        # Reuse compute_loss_from_incoming by creating a temp a_field if needed would be overkill
        # Do it directly on CPU: loss = (1-a) * incoming
        inc_np = incoming.field.to_numpy()
        loss_np = (1.0 - a) * inc_np
        out = loss_np.reshape(self.rshp)

        # Release
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()
        w.release()
        w_.release()
        incoming.release()
        incoming_.release()
        loss.release()

        return out

    def accumulate_affine_var_with_loss(self, a:ti.template(), b:ti.template()):
        """
        Variable-a affine accumulation and per-node loss in one pass.

        Stores q_out in self.Q and returns per-node loss as numpy array.

        Author: B.G. - G.C.
        """
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Temporaries
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        w = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        w_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        incoming = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        incoming_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        loss = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        a_field = getattr(a, "field", a)
        b_field = getattr(b, "field", b)

        # Init
        ndonors.field.fill(0)
        src.field.fill(0)
        incoming.field.fill(0.0)
        incoming_.field.fill(0.0)

        self.Q.field.copy_from(b_field)

        # Build donors and init weights w = a[i]
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)
        dpr.init_affine_weights_var(ndonors.field, w.field, a_field)

        # Iterations
        for i in range(logn + 1):
            dpr.rake_compress_accum_affine_var_with_incoming(
                donors.field,
                ndonors.field,
                self.Q.field,
                incoming.field,
                src.field,
                donors_.field,
                ndonors_.field,
                Q_.field,
                incoming_.field,
                w.field,
                w_.field,
                i,
                a_field,
                b_field,
            )

        # Fuse
        gena.fuse(self.Q.field, src.field, Q_.field, logn)
        gena.fuse(incoming.field, src.field, incoming_.field, logn)

        # Compute loss = (1 - a[i]) * incoming[i]
        dpr.compute_loss_from_incoming(incoming.field, a_field, loss.field)
        out = loss.field.to_numpy().reshape(self.rshp)

        # Release
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()
        w.release()
        w_.release()
        incoming.release()
        incoming_.release()
        loss.release()

        return out

    def compute_loss_affine_var(self, a:ti.template()):
        """
        Compute per-node loss/deposition given current Q and spatially varying a.

        Assumes self.Q contains the outflow q_out at each node from a prior
        accumulate_affine_var or accumulate_affine call. Returns loss[i] =
        (1 - a[i]) * incoming_pre_attenuation[i], where incoming_pre_attenuation
        is the sum of upstream q_out contributions reaching i before applying a[i].

        Args:
            a: Taichi field (or TPField) of transmission coefficients per node

        Returns:
            numpy.ndarray: 2D array of per-node loss values (ny, nx)

        Author: B.G. - G.C.
        """
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Temporaries
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        incoming = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        incoming_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        w = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        w_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        loss = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        a_field = getattr(a, "field", a)

        # Init
        ndonors.field.fill(0)
        src.field.fill(0)
        incoming.field.fill(0.0)
        incoming_.field.fill(0.0)

        # Build donors and init unit weights (exclude a[i])
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)
        dpr.init_unit_weights(ndonors.field, w.field, 1.0)

        # Accumulate incoming pre-attenuation using Q as donor values
        for i in range(logn + 1):
            dpr.rake_compress_accum_affine_var_from_external(
                donors.field,
                ndonors.field,
                incoming.field,
                src.field,
                donors_.field,
                ndonors_.field,
                incoming_.field,
                w.field,
                w_.field,
                i,
                a_field,
                self.Q.field,
            )

        # Fuse incoming
        gena.fuse(incoming.field, src.field, incoming_.field, logn)

        # Compute loss = (1 - a) * incoming
        dpr.compute_loss_from_incoming(incoming.field, a_field, loss.field)

        # Copy to numpy and release
        out = loss.field.to_numpy().reshape(self.rshp)

        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        incoming.release()
        incoming_.release()
        w.release()
        w_.release()
        loss.release()

        return out

    def accumulate_affine_var(self, a:ti.template(), b:ti.template()):
        """
        Accumulate downstream with spatially varying a: q_out = a[i]*q_in + b[i].

        Args:
            a: Taichi field (or TPField) of transmission coefficients per node (shape nx*ny)
            b: Taichi field (or TPField) of local source terms per node (shape nx*ny)

        Returns:
            None. Stores result in self.Q.

        Author: B.G. - G.C.
        """
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Temporary fields
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        w = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))
        w_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny * 4))

        # Normalize inputs
        a_field = getattr(a, "field", a)
        b_field = getattr(b, "field", b)

        # Initialize arrays
        ndonors.field.fill(0)
        src.field.fill(0)

        # Initialize p with local sources b
        self.Q.field.copy_from(b_field)

        # Build donors and initialize per-edge weights with a[i]
        dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)
        dpr.init_affine_weights_var(ndonors.field, w.field, a_field)

        # Iterations
        for i in range(logn + 1):
            dpr.rake_compress_accum_affine_var(
                donors.field,
                ndonors.field,
                self.Q.field,
                src.field,
                donors_.field,
                ndonors_.field,
                Q_.field,
                w.field,
                w_.field,
                i,
                a_field,
                b_field,
            )

        # Fuse
        gena.fuse(self.Q.field, src.field, Q_.field, logn)

        # Release temporaries
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()
        w.release()
        w_.release()

    def accumulate_constant_Q_stochastic(self, value, area=True, N=4):
        """
        Accumulate constant flow values using stochastic flow routing with multiple realizations.

        Performs flow accumulation using stochastic receivers with multiple iterations to
        create ensemble flow patterns. Each iteration uses different stochastic receivers,
        and results are averaged to provide robust flow accumulation estimates with
        uncertainty quantification.

        Args:
            value (float): Constant flow value to accumulate at each node (units depend on area flag)
            area (bool, optional): If True, multiply value by cell area (dx²) for volumetric flow.
                If False, use raw value for unit-based flow. Default: True.
            N (int, optional): Number of stochastic realizations to average. Default: 4.
                Higher N provides more robust results but increases computation time.

        Returns:
            None: Results are stored in self.Q field (pool-allocated)

        Note:
            - Requires stochastic receivers from compute_stochastic_receivers()
            - Each realization uses different random flow paths
            - Final result is ensemble average of all realizations
            - Uses pool-based memory management for temporary fields
            - More computationally expensive than deterministic accumulation

        Example:
            router.compute_stochastic_receivers()  # Generate stochastic flow network
            router.accumulate_constant_Q_stochastic(1.0, area=True, N=10)
            ensemble_area = router.get_Q()  # Get ensemble-averaged drainage area

        Author: B.G.
        """
        # Get temporary accumulation field
        fullQ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        fullQ.field.fill(0.0)

        # Calculate number of iterations needed (log₂ of grid size)
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Get temporary fields from pool for rake-compress algorithm
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))

        for __ in range(N):
            self.compute_stochastic_receivers()

            # Initialize arrays for rake-compress algorithm
            ndonors.field.fill(0)  # Reset donor counts
            src.field.fill(0)  # Reset ping-pong state

            # Initialize flow values (multiply by area if requested)
            self.Q.field.fill(value * (cte.DX * cte.DX if area else 1.0))

            # Build donor-receiver relationships from receiver array
            dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)

            # Rake-compress iterations for parallel tree accumulation
            # Each iteration doubles the effective path length being compressed
            for i in range(logn + 1):
                dpr.rake_compress_accum(
                    donors.field,
                    ndonors.field,
                    self.Q.field,
                    src.field,
                    donors_.field,
                    ndonors_.field,
                    Q_.field,
                    i,
                )

            # Final fuse step to consolidate results from ping-pong buffers
            # Merge accumulated values from working arrays back to primary array
            gena.fuse(self.Q.field, src.field, Q_.field, logn)

            ut.add_B_to_weighted_A(fullQ.field, self.Q.field, 1.0 / N)

        # Release temporary fields for this iteration
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()

        self.Q.field.copy_from(fullQ.field)
        fullQ.release()

    def accumulate_constant_Q_mfd(self, value, area=True, N = 5, temporal_filtering = 0.5):
        """
        TO BE TESTED
        Author: B.G.
        """
        # Get temporary accumulation field
        fullQ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
        fullQ.field.fill(0.0)
        done_once = False

        # Calculate number of iterations needed (log₂ of grid size)
        logn = math.ceil(math.log2(self.nx * self.ny)) + 1

        # Get temporary fields from pool for rake-compress algorithm
        ndonors = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        src = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        donors_ = pf.pool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.nx * self.ny * 4)
        )
        ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        Q_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx * self.ny))
    
        for __ in range(N):
            self.compute_stochastic_receivers()

            # Initialize arrays for rake-compress algorithm
            ndonors.field.fill(0)  # Reset donor counts
            src.field.fill(0)  # Reset ping-pong state

            # Initialize flow values (multiply by area if requested)
            self.Q.field.fill(value * (cte.DX * cte.DX if area else 1.0))

            # Build donor-receiver relationships from receiver array
            dpr.rcv2donor(self.receivers.field, donors.field, ndonors.field)

            # Rake-compress iterations for parallel tree accumulation
            # Each iteration doubles the effective path length being compressed
            for i in range(logn + 1):
                dpr.rake_compress_accum(
                    donors.field,
                    ndonors.field,
                    self.Q.field,
                    src.field,
                    donors_.field,
                    ndonors_.field,
                    Q_.field,
                    i,
                )

            # Final fuse step to consolidate results from ping-pong buffers
            # Merge accumulated values from working arrays back to primary array
            gena.fuse(self.Q.field, src.field, Q_.field, logn)

            ut.weighted_mean_B_in_A(fullQ.field, self.Q.field, 1. - temporal_filtering)

        # Release temporary fields for this iteration
        ndonors.release()
        donors.release()
        src.release()
        donors_.release()
        ndonors_.release()
        Q_.release()

        self.Q.field.copy_from(fullQ.field)
        fullQ.release()

    def fill_z(self, epsilon=1e-3):
        """
        Fill topographic depressions to ensure proper flow routing.

        Args:
            epsilon: Small elevation increment for depression filling

        Author: B.G.
        """
        fl.topofill(self, epsilon=epsilon, custom_z=None)

    def get_Q(self):
        """
        Get flow accumulation results as 2D numpy array.

        Returns:
            numpy.ndarray: 2D array of flow accumulation values (ny, nx)

        Author: B.G.
        """
        return self.Q.field.to_numpy().reshape(self.rshp)

    def propagate_upstream_affine(self, a: float, b_tpfield, x_root_tpfield, result_tpfield):
        """
        Propagate values upstream using constant affine transformation: x[i] = a * x[rcv[i]] + b[i].

        This function solves the upstream propagation problem where each node's value depends
        on its receiver's value through a linear affine transformation. The constant coefficient
        'a' is applied uniformly across all nodes, while each node has its own source term 'b[i]'.

        Mathematical Formulation:
        ========================
        For each node i in the drainage network:
            x[i] = a * x[rcv[i]] + b[i]

        Where:
        - x[i]: Final value at node i (what we're solving for)
        - a: Constant transmission/propagation coefficient (scalar, 0 ≤ a ≤ 1 typically)
        - x[rcv[i]]: Value at the receiver (downstream node) of node i
        - b[i]: Local source term at node i (can be positive or negative)
        - rcv[i]: Receiver index for node i (downstream neighbor in flow network)

        Special Cases:
        - Root nodes (rcv[i] == i): Use boundary values from x_root_tpfield
        - Outlet nodes: Typically have x_root values that define system boundaries

        Algorithm Details:
        ==================
        Uses parallel pointer jumping to solve the system in O(log N) iterations:

        1. **Initialization**: Set up affine parameters (A, B) and parent pointers (P)
           - For non-root nodes: A[i] = a, B[i] = b[i], P[i] = rcv[i]
           - For root nodes: A[i] = 1, B[i] = 0, P[i] = i (identity transform)

        2. **Pointer Jumping**: Iteratively compose transforms by path doubling
           - Each iteration doubles the effective path length
           - Composes affine transforms: (A_new, B_new) = compose(A_old, A_parent, B_old, B_parent)
           - A_new[i] = A[i] * A[P[i]]
           - B_new[i] = B[i] + A[i] * B[P[i]]
           - P_new[i] = P[P[i]]

        3. **Finalization**: Apply final transform using root boundary values
           - x[i] = A[i] * x_root[P[i]] + B[i]

        Physical Interpretation Examples:
        =================================
        - **Heat conduction**: a = thermal diffusivity, b = heat sources/sinks
        - **Pollutant transport**: a = transmission rate, b = local emissions
        - **Economic flow**: a = efficiency factor, b = local production
        - **Water chemistry**: a = reaction rate, b = local inputs

        Args:
            a (float): Constant transmission coefficient applied uniformly to all nodes.
                      Typical range: 0 ≤ a ≤ 1
                      - a = 1.0: Perfect transmission (no loss)
                      - a = 0.0: No upstream influence (local sources only)
                      - 0 < a < 1: Partial transmission with decay

            b_tpfield (TPField): Local source terms for each node. Must have shape (nx*ny,).
                               Contains b[i] values that represent:
                               - Local inputs/outputs at each node
                               - Can be positive (sources) or negative (sinks)
                               - Units depend on application context

            x_root_tpfield (TPField): Boundary values for root/outlet nodes. Must have shape (nx*ny,).
                                    Only values at root positions (where rcv[i] == i) are used.
                                    These define the boundary conditions for the system.

            result_tpfield (TPField): Output field where results will be stored. Must have shape (nx*ny,).
                                    Will contain the computed x[i] values after propagation.

        Returns:
            None: Results are stored directly in result_tpfield

        Raises:
            AttributeError: If tpfields don't have .field attribute
            ValueError: If field shapes don't match grid dimensions

        Example Usage:
        ==============
        ```python
        import numpy as np
        import taichi as ti
        import pyfastflow as pf

        # Setup grid and flow router
        nx, ny = 100, 100
        grid = pf.create_grid(nx, ny, dx=1.0)
        router = pf.FlowRouter(grid)

        # Compute flow network
        router.compute_receivers()

        # Example 1: Simple heat diffusion
        # Heat conducts upstream with 90% efficiency, local heat sources
        a = 0.9  # 90% heat transmission

        # Create local heat sources (more heat in center)
        b_data = np.zeros((ny, nx))
        b_data[ny//2-10:ny//2+10, nx//2-10:nx//2+10] = 100.0  # Central heat source
        b_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        b_tpfield.field.from_numpy(b_data.flatten())

        # Set boundary conditions (outlets at 0°C)
        x_root_data = np.zeros((ny, nx))  # All outlets at 0°C
        x_root_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        x_root_tpfield.field.from_numpy(x_root_data.flatten())

        # Create result field
        result_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))

        # Propagate heat upstream
        router.propagate_upstream_affine(a, b_tpfield, x_root_tpfield, result_tpfield)

        # Extract results
        temperature = result_tpfield.field.to_numpy().reshape((ny, nx))
        print(f"Max temperature: {temperature.max():.2f}°C")

        # Clean up
        b_tpfield.release()
        x_root_tpfield.release()
        result_tpfield.release()

        # Example 2: Pollutant transport with decay
        # 80% of pollutant survives transport, industrial sources
        a = 0.8  # 20% decay during transport

        # Industrial pollution sources
        pollution_sources = np.random.exponential(10.0, (ny, nx))  # Random industrial sources
        b_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        b_tpfield.field.from_numpy(pollution_sources.flatten())

        # Clean water at outlets
        clean_boundary = np.zeros((ny, nx))
        x_root_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        x_root_tpfield.field.from_numpy(clean_boundary.flatten())

        result_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))

        router.propagate_upstream_affine(a, b_tpfield, x_root_tpfield, result_tpfield)

        pollution_map = result_tpfield.field.to_numpy().reshape((ny, nx))
        print(f"Max pollution level: {pollution_map.max():.2f}")

        # Clean up
        b_tpfield.release()
        x_root_tpfield.release()
        result_tpfield.release()
        ```

        Performance Notes:
        ==================
        - Time complexity: O(log N) where N = nx * ny
        - Space complexity: O(N) for temporary fields
        - GPU-accelerated using Taichi parallel kernels
        - Memory efficient with automatic cleanup of temporary fields
        - Suitable for grids up to millions of nodes

        Implementation Notes:
        =====================
        - Uses pointer jumping algorithm for parallel tree traversal
        - Temporary fields (A, B, P) are automatically managed via pool system
        - All computations performed on GPU for maximum efficiency
        - Handles arbitrary drainage network topologies including multiple outlets
        - Numerically stable for typical coefficient ranges

        See Also:
        =========
        propagate_upstream_affine_var: Version with spatially varying coefficients
        accumulate_affine: Downstream accumulation with affine transforms

        Author: B.G.
        """
        import math

        # Grid dimensions and logarithmic iteration count for pointer jumping
        N = self.nx * self.ny  # Total number of grid nodes
        logn = math.ceil(math.log2(N)) + 1  # Number of pointer jumping iterations needed

        # Allocate temporary fields from pool for affine transform parameters
        # A[i]: Cumulative coefficient from node i to its final ancestor
        # B[i]: Cumulative offset from node i to its final ancestor
        # P[i]: Parent pointer - points to upstream node in current iteration
        A = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))      # Current coefficients
        B = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))      # Current offsets
        P = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))      # Current parent pointers
        A_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))     # Next iteration coefficients
        B_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))     # Next iteration offsets
        P_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))     # Next iteration pointers

        # Extract taichi fields from tpfield wrappers for kernel calls
        b_field = b_tpfield.field           # Local source terms field
        xroot_field = x_root_tpfield.field  # Boundary values field
        result_field = result_tpfield.field # Output field

        # Phase 1: Initialize affine parameters and parent pointers
        # Sets up initial transform: x[i] = a * x[rcv[i]] + b[i]
        # For root nodes (rcv[i] == i): A[i] = 1, B[i] = 0, P[i] = i (identity)
        # For regular nodes: A[i] = a, B[i] = b[i], P[i] = rcv[i]
        pup.init_upstream_AB_const(
            self.receivers.field,  # Receiver network (rcv[i] for each node i)
            A.field,               # Output: coefficient array A[i]
            B.field,               # Output: offset array B[i]
            P.field,               # Output: parent pointer array P[i]
            a,                     # Input: constant transmission coefficient
            b_field                # Input: local source terms b[i]
        )

        # Phase 2: Pointer jumping iterations to compose affine transforms
        # Each iteration doubles the effective path length by composing transforms:
        # If node i points to j, and j points to k, compose the transforms i->j and j->k
        # to get direct transform i->k, then update pointer i->k
        for iteration in range(logn):
            # Compose affine transforms by path doubling
            # New transform: x[i] = A[i] * (A[j] * x[k] + B[j]) + B[i]
            #              = (A[i] * A[j]) * x[k] + (A[i] * B[j] + B[i])
            # Where j = P[i] and k = P[j]
            pup.pointer_jump_affine(
                P.field,   # Input: current parent pointers P[i]
                A.field,   # Input: current coefficients A[i]
                B.field,   # Input: current offsets B[i]
                P_.field,  # Output: updated parent pointers P_[i] = P[P[i]]
                A_.field,  # Output: updated coefficients A_[i] = A[i] * A[P[i]]
                B_.field   # Output: updated offsets B_[i] = B[i] + A[i] * B[P[i]]
            )

            # Swap buffers: current <- next for ping-pong computation
            # This avoids race conditions during parallel updates
            P.field.copy_from(P_.field)   # Update parent pointers
            A.field.copy_from(A_.field)   # Update coefficients
            B.field.copy_from(B_.field)   # Update offsets

        # Phase 3: Finalize by applying transforms with boundary conditions
        # After pointer jumping, each node i has direct transform to its root:
        # x[i] = A[i] * x_root[P[i]] + B[i]
        # where P[i] now points directly to the root of i's subtree
        pup.finalize_upstream_x(
            P.field,        # Input: final parent pointers (point to roots)
            A.field,        # Input: final cumulative coefficients
            B.field,        # Input: final cumulative offsets
            xroot_field,    # Input: boundary values at root nodes
            result_field    # Output: computed values x[i]
        )

        # Phase 4: Release temporary fields back to pool
        # Ensures efficient memory management and prevents leaks
        A.release()      # Release coefficient arrays
        B.release()
        P.release()      # Release pointer arrays
        A_.release()
        B_.release()
        P_.release()

    def propagate_upstream_affine_var(self, a_tpfield, b_tpfield, x_root_tpfield, result_tpfield):
        """
        Propagate values upstream using spatially varying affine transformation: x[i] = a[i] * x[rcv[i]] + b[i].

        This function solves the upstream propagation problem where each node's value depends
        on its receiver's value through a spatially varying linear affine transformation. Unlike
        the constant version, each node has its own transmission coefficient 'a[i]' and source
        term 'b[i]', enabling heterogeneous material properties and spatially variable processes.

        Mathematical Formulation:
        ========================
        For each node i in the drainage network:
            x[i] = a[i] * x[rcv[i]] + b[i]

        Where:
        - x[i]: Final value at node i (what we're solving for)
        - a[i]: Spatially varying transmission/propagation coefficient at node i (0 ≤ a[i] ≤ 1 typically)
        - x[rcv[i]]: Value at the receiver (downstream node) of node i
        - b[i]: Local source term at node i (can be positive or negative)
        - rcv[i]: Receiver index for node i (downstream neighbor in flow network)

        Special Cases:
        - Root nodes (rcv[i] == i): Use boundary values from x_root_tpfield
        - Different nodes can have vastly different transmission properties
        - Enables modeling of heterogeneous landscapes with varying material properties

        Algorithm Details:
        ==================
        Uses parallel pointer jumping with spatially varying coefficients in O(log N) iterations:

        1. **Initialization**: Set up affine parameters (A, B) and parent pointers (P)
           - For non-root nodes: A[i] = a[i], B[i] = b[i], P[i] = rcv[i]
           - For root nodes: A[i] = 1, B[i] = 0, P[i] = i (identity transform)

        2. **Pointer Jumping**: Iteratively compose variable transforms by path doubling
           - Each iteration doubles the effective path length
           - Composes variable affine transforms maintaining spatial heterogeneity
           - A_new[i] = A[i] * A[P[i]]  (product of transmission along path)
           - B_new[i] = B[i] + A[i] * B[P[i]]  (accumulated sources with transmission)
           - P_new[i] = P[P[i]]  (skip intermediate nodes)

        3. **Finalization**: Apply final transform using root boundary values
           - x[i] = A[i] * x_root[P[i]] + B[i]

        Physical Interpretation Examples:
        =================================
        - **Heterogeneous heat conduction**: a[i] = thermal conductivity map, b[i] = heat sources/sinks
        - **Variable pollutant transport**: a[i] = local transmission/decay rates, b[i] = pollution sources
        - **Landscape erosion**: a[i] = rock resistance, b[i] = uplift rates
        - **Groundwater flow**: a[i] = hydraulic conductivity, b[i] = recharge/discharge
        - **Economic networks**: a[i] = local efficiency, b[i] = local production
        - **Ecosystem services**: a[i] = habitat connectivity, b[i] = local service provision

        Comparison with Constant Version:
        ================================
        - **Constant a**: All nodes have same transmission properties (homogeneous landscape)
        - **Variable a[i]**: Each node can have different properties (heterogeneous landscape)
        - **Computational cost**: Variable version is slightly more expensive due to field access
        - **Memory usage**: Variable version requires additional storage for a[i] field
        - **Modeling power**: Variable version can represent much more complex systems

        Args:
            a_tpfield (TPField): Spatially varying transmission coefficients for each node.
                               Must have shape (nx*ny,). Contains a[i] values that represent:
                               - Local transmission efficiency at each node
                               - Typical range: 0 ≤ a[i] ≤ 1, but can be outside for special cases
                               - a[i] = 1.0: Perfect transmission (no local loss)
                               - a[i] = 0.0: Complete absorption (no upstream influence)
                               - 0 < a[i] < 1: Partial transmission with local decay
                               - a[i] > 1.0: Local amplification (rare, can cause instability)

            b_tpfield (TPField): Local source terms for each node. Must have shape (nx*ny,).
                               Contains b[i] values that represent:
                               - Local inputs/outputs at each node
                               - Can be positive (sources) or negative (sinks)
                               - Units depend on application context
                               - Can vary dramatically across the domain

            x_root_tpfield (TPField): Boundary values for root/outlet nodes. Must have shape (nx*ny,).
                                    Only values at root positions (where rcv[i] == i) are used.
                                    These define the boundary conditions for the system.

            result_tpfield (TPField): Output field where results will be stored. Must have shape (nx*ny,).
                                    Will contain the computed x[i] values after propagation.

        Returns:
            None: Results are stored directly in result_tpfield

        Raises:
            AttributeError: If tpfields don't have .field attribute
            ValueError: If field shapes don't match grid dimensions
            RuntimeError: If transmission coefficients cause numerical instability

        Example Usage:
        ==============
        ```python
        import numpy as np
        import taichi as ti
        import pyfastflow as pf

        # Setup grid and flow router
        nx, ny = 100, 100
        grid = pf.create_grid(nx, ny, dx=1.0)
        router = pf.FlowRouter(grid)

        # Compute flow network
        router.compute_receivers()

        # Example 1: Heterogeneous landscape erosion
        # Variable rock resistance with spatially varying uplift

        # Create spatially varying rock resistance (transmission coefficient)
        # Higher values = less resistant rock (more transmission)
        # Lower values = more resistant rock (less transmission)
        elevation = grid.get_Z()
        a_data = np.ones((ny, nx)) * 0.8  # Base resistance
        a_data[elevation > np.percentile(elevation, 75)] = 0.3  # High elevation = resistant
        a_data[elevation < np.percentile(elevation, 25)] = 0.95  # Low elevation = weak
        a_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        a_tpfield.field.from_numpy(a_data.flatten())

        # Create spatially varying uplift pattern (source terms)
        x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny))
        center_x, center_y = nx//2, ny//2
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        b_data = 10.0 * np.exp(-distance**2 / (2 * (nx/6)**2))  # Gaussian uplift
        b_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        b_tpfield.field.from_numpy(b_data.flatten())

        # Boundary conditions (base level at outlets)
        x_root_data = np.zeros((ny, nx))
        x_root_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        x_root_tpfield.field.from_numpy(x_root_data.flatten())

        # Create result field
        result_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))

        # Propagate erosion upstream with variable rock resistance
        router.propagate_upstream_affine_var(a_tpfield, b_tpfield, x_root_tpfield, result_tpfield)

        # Extract and analyze results
        erosion_potential = result_tpfield.field.to_numpy().reshape((ny, nx))
        print(f"Max erosion potential: {erosion_potential.max():.2f}")
        print(f"Mean erosion potential: {erosion_potential.mean():.2f}")

        # Example 2: Pollutant transport with variable decay rates
        # Different land use types have different pollutant retention

        # Create land use based transmission (simplified)
        land_use = np.random.choice([0, 1, 2], size=(ny, nx), p=[0.4, 0.4, 0.2])
        a_land = np.zeros((ny, nx))
        a_land[land_use == 0] = 0.9   # Urban: high transmission
        a_land[land_use == 1] = 0.6   # Forest: medium retention
        a_land[land_use == 2] = 0.3   # Wetland: high retention
        a_tpfield.field.from_numpy(a_land.flatten())

        # Point source pollution pattern
        pollution_sources = np.zeros((ny, nx))
        n_sources = 5
        for _ in range(n_sources):
            sx, sy = np.random.randint(10, nx-10), np.random.randint(10, ny-10)
            pollution_sources[sy-2:sy+3, sx-2:sx+3] = np.random.exponential(50.0)
        b_tpfield.field.from_numpy(pollution_sources.flatten())

        # Clean boundaries
        x_root_data.fill(0.0)
        x_root_tpfield.field.from_numpy(x_root_data.flatten())

        # Propagate pollution with land use dependent retention
        router.propagate_upstream_affine_var(a_tpfield, b_tpfield, x_root_tpfield, result_tpfield)

        pollution_map = result_tpfield.field.to_numpy().reshape((ny, nx))

        # Analyze pollution by land use
        for lu, name in enumerate(['Urban', 'Forest', 'Wetland']):
            mask = land_use == lu
            if mask.any():
                mean_pollution = pollution_map[mask].mean()
                print(f"{name} average pollution: {mean_pollution:.2f}")

        # Clean up all fields
        a_tpfield.release()
        b_tpfield.release()
        x_root_tpfield.release()
        result_tpfield.release()

        # Example 3: Advanced ecosystem connectivity modeling
        # Habitat quality affects species dispersal efficiency

        # Create habitat quality map (affects transmission)
        np.random.seed(42)
        habitat_quality = np.random.beta(2, 2, (ny, nx))  # Values between 0 and 1
        # Add some large protected areas
        protected_areas = np.zeros((ny, nx))
        for _ in range(3):
            cx, cy = np.random.randint(20, nx-20), np.random.randint(20, ny-20)
            radius = np.random.randint(8, 15)
            for i in range(ny):
                for j in range(nx):
                    if (i-cy)**2 + (j-cx)**2 <= radius**2:
                        protected_areas[i, j] = 1

        # High quality in protected areas
        habitat_quality = np.where(protected_areas,
                                 np.clip(habitat_quality + 0.4, 0, 1),
                                 habitat_quality)

        a_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        a_tpfield.field.from_numpy(habitat_quality.flatten())

        # Species source populations (breeding sites)
        breeding_sites = protected_areas * np.random.exponential(20.0, (ny, nx))
        b_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        b_tpfield.field.from_numpy(breeding_sites.flatten())

        # Population sinks at boundaries
        x_root_data.fill(0.0)
        x_root_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))
        x_root_tpfield.field.from_numpy(x_root_data.flatten())

        result_tpfield = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx*ny,))

        # Model species connectivity
        router.propagate_upstream_affine_var(a_tpfield, b_tpfield, x_root_tpfield, result_tpfield)

        species_density = result_tpfield.field.to_numpy().reshape((ny, nx))
        connectivity_index = (species_density * habitat_quality).sum()
        print(f"Landscape connectivity index: {connectivity_index:.1f}")

        # Clean up
        a_tpfield.release()
        b_tpfield.release()
        x_root_tpfield.release()
        result_tpfield.release()
        ```

        Performance Notes:
        ==================
        - Time complexity: O(log N) where N = nx * ny (same as constant version)
        - Space complexity: O(N) for temporary fields plus input field storage
        - GPU-accelerated using Taichi parallel kernels with efficient memory access
        - Slightly slower than constant version due to variable coefficient access
        - Memory bandwidth limited for large grids (optimize data layout)
        - Suitable for grids up to millions of nodes with careful memory management

        Implementation Notes:
        =====================
        - Uses same pointer jumping algorithm as constant version
        - Variable coefficients maintained throughout pointer jumping process
        - Temporary fields (A, B, P) automatically managed via pool system
        - All computations performed on GPU with coalesced memory access
        - Handles arbitrary drainage network topologies including multiple outlets
        - Numerically stable for reasonable coefficient ranges (avoid a[i] >> 1)
        - Supports both positive and negative coefficients for advanced modeling

        Numerical Considerations:
        ========================
        - Avoid a[i] significantly greater than 1.0 to prevent instability
        - Very small a[i] values (< 1e-6) may cause precision issues
        - Large dynamic ranges in a[i] or b[i] may require careful scaling
        - Monitor for overflow/underflow in accumulated products along long paths

        See Also:
        =========
        propagate_upstream_affine: Version with constant transmission coefficient
        accumulate_affine_var: Downstream accumulation with variable coefficients
        compute_loss_affine_var: Compute loss/deposition patterns

        Author: B.G.
        """
        import math

        # Grid dimensions and logarithmic iteration count for pointer jumping
        N = self.nx * self.ny  # Total number of grid nodes
        logn = math.ceil(math.log2(N)) + 1  # Number of pointer jumping iterations needed

        # Allocate temporary fields from pool for affine transform parameters
        # A[i]: Cumulative coefficient product from node i to its final ancestor
        # B[i]: Cumulative offset sum from node i to its final ancestor
        # P[i]: Parent pointer - points to upstream node in current iteration
        A = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))      # Current coefficients
        B = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))      # Current offsets
        P = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))      # Current parent pointers
        A_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))     # Next iteration coefficients
        B_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))     # Next iteration offsets
        P_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))     # Next iteration pointers

        # Extract taichi fields from tpfield wrappers for kernel calls
        a_field = a_tpfield.field           # Spatially varying transmission coefficients
        b_field = b_tpfield.field           # Local source terms field
        xroot_field = x_root_tpfield.field  # Boundary values field
        result_field = result_tpfield.field # Output field

        # Phase 1: Initialize affine parameters and parent pointers with variable coefficients
        # Sets up initial transform: x[i] = a[i] * x[rcv[i]] + b[i]
        # For root nodes (rcv[i] == i): A[i] = 1, B[i] = 0, P[i] = i (identity)
        # For regular nodes: A[i] = a[i], B[i] = b[i], P[i] = rcv[i]
        # This initialization preserves spatial heterogeneity in transmission properties
        pup.init_upstream_AB_var(
            self.receivers.field,  # Receiver network (rcv[i] for each node i)
            A.field,               # Output: coefficient array A[i] = a[i]
            B.field,               # Output: offset array B[i] = b[i]
            P.field,               # Output: parent pointer array P[i] = rcv[i]
            a_field,               # Input: spatially varying transmission coefficients a[i]
            b_field                # Input: local source terms b[i]
        )

        # Phase 2: Pointer jumping iterations to compose variable affine transforms
        # Each iteration doubles the effective path length by composing heterogeneous transforms:
        # If node i points to j with coefficient a[i], and j points to k with coefficient a[j],
        # compose to get direct transform i->k with coefficient a[i]*a[j]
        # This maintains the product of all transmission coefficients along the path
        for iteration in range(logn):
            # Compose spatially varying affine transforms by path doubling
            # For variable coefficients, the composition becomes:
            # New transform: x[i] = A[i] * (A[j] * x[k] + B[j]) + B[i]
            #              = (A[i] * A[j]) * x[k] + (A[i] * B[j] + B[i])
            # Where A[i] and A[j] contain products of variable a[k] values along paths
            # This preserves the heterogeneous nature of transmission through the network
            pup.pointer_jump_affine(
                P.field,   # Input: current parent pointers P[i]
                A.field,   # Input: current cumulative coefficients A[i]
                B.field,   # Input: current cumulative offsets B[i]
                P_.field,  # Output: updated parent pointers P_[i] = P[P[i]]
                A_.field,  # Output: updated coefficients A_[i] = A[i] * A[P[i]]
                B_.field   # Output: updated offsets B_[i] = B[i] + A[i] * B[P[i]]
            )

            # Swap buffers: current <- next for ping-pong computation
            # This avoids race conditions during parallel updates while maintaining
            # the accumulated products of variable coefficients
            P.field.copy_from(P_.field)   # Update parent pointers
            A.field.copy_from(A_.field)   # Update cumulative coefficients
            B.field.copy_from(B_.field)   # Update cumulative offsets

        # Phase 3: Finalize by applying variable transforms with boundary conditions
        # After pointer jumping, each node i has direct transform to its root incorporating
        # all variable coefficients along the path: x[i] = A[i] * x_root[P[i]] + B[i]
        # where A[i] contains the product of all a[k] values from i to its root
        # and B[i] contains the sum of all weighted source terms along the path
        pup.finalize_upstream_x(
            P.field,        # Input: final parent pointers (point to roots)
            A.field,        # Input: final cumulative coefficients (products of a[k])
            B.field,        # Input: final cumulative offsets (weighted sums of b[k])
            xroot_field,    # Input: boundary values at root nodes
            result_field    # Output: computed values x[i]
        )

        # Phase 4: Release temporary fields back to pool
        # Ensures efficient memory management and prevents GPU memory leaks
        A.release()      # Release coefficient arrays
        B.release()
        P.release()      # Release pointer arrays
        A_.release()
        B_.release()
        P_.release()

    def get_Z(self):
        """
        Get elevation data as 2D numpy array.

        Returns:
            numpy.ndarray: 2D array of elevation values (ny, nx)

        Author: B.G.
        """
        return self.grid.z.field.to_numpy().reshape(self.rshp)

    def get_receivers(self):
        """
        Get receiver data as 2D numpy array.

        Returns:
            numpy.ndarray: 2D array of receiver indices (ny, nx)

        Author: B.G.
        """
        return self.receivers.field.to_numpy().reshape(self.rshp)

    def destroy(self):
        """
        Release all pooled fields and free GPU memory.

        Should be called when finished with the FlowRouter to ensure
        proper cleanup of GPU resources. After calling this method,
        the FlowRouter should not be used.

        Author: B.G.
        """
        # Release core fields back to pool
        if hasattr(self, "Q") and self.Q is not None:
            self.Q.release()
            self.Q = None

        if hasattr(self, "receivers") and self.receivers is not None:
            self.receivers.release()
            self.receivers = None

    def __del__(self):
        """
        Destructor - automatically release fields when object is deleted.

        Author: B.G.
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass  # Ignore errors during destruction
