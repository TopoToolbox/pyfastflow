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

    def propagate_upstream_affine(self, a: float, b: ti.template(), x_root: ti.template()):
        """
        Upstream affine propagation with constant a: x[i] = a * x[rcv[i]] + b[i].

        Args:
            a: scalar coefficient
            b: per-node source field
            x_root: per-node boundary values for root/self-receiver nodes

        Stores result in self.Q.
        """
        import math

        N = self.nx * self.ny
        logn = math.ceil(math.log2(N)) + 1

        # Temp fields
        A = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        B = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        P = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))
        A_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        B_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        P_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))

        b_field = getattr(b, "field", b)
        xroot_field = getattr(x_root, "field", x_root)

        pup.init_upstream_AB_const(
            self.receivers.field, A.field, B.field, P.field, a, b_field
        )

        # Pointer jumping
        for _ in range(logn):
            pup.pointer_jump_affine(P.field, A.field, B.field, P_.field, A_.field, B_.field)
            # swap
            P.field.copy_from(P_.field)
            A.field.copy_from(A_.field)
            B.field.copy_from(B_.field)

        # Finalize
        pup.finalize_upstream_x(P.field, A.field, B.field, xroot_field, self.Q.field)

        # Release temps
        A.release(); B.release(); P.release(); A_.release(); B_.release(); P_.release()

    def propagate_upstream_affine_var(self, a: ti.template(), b: ti.template(), x_root: ti.template()):
        """
        Upstream affine propagation with spatially varying a: x[i] = a[i] * x[rcv[i]] + b[i].

        Args:
            a: per-node coefficient field
            b: per-node source field
            x_root: per-node boundary values for root/self-receiver nodes

        Stores result in self.Q.
        """
        import math

        N = self.nx * self.ny
        logn = math.ceil(math.log2(N)) + 1

        # Temp fields
        A = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        B = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        P = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))
        A_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        B_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(N,))
        P_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(N,))

        a_field = getattr(a, "field", a)
        b_field = getattr(b, "field", b)
        xroot_field = getattr(x_root, "field", x_root)

        pup.init_upstream_AB_var(
            self.receivers.field, A.field, B.field, P.field, a_field, b_field
        )

        for _ in range(logn):
            pup.pointer_jump_affine(P.field, A.field, B.field, P_.field, A_.field, B_.field)
            P.field.copy_from(P_.field)
            A.field.copy_from(A_.field)
            B.field.copy_from(B_.field)

        pup.finalize_upstream_x(P.field, A.field, B.field, xroot_field, self.Q.field)

        A.release(); B.release(); P.release(); A_.release(); B_.release(); P_.release()

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
