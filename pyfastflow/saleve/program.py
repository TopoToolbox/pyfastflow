"""Salève: one capacity-sized analytical terrain program with active grid levels."""

import math
from numbers import Integral

import cupy as cp
import numpy as np

from pyfastflow.flow import SFDFlowProgram

from ._cliffs import SaleveCliffProgram
from ._finite import SaleveFiniteProgram
from ._levels import SaleveLevelProgram
from ._steady import SaleveSteadyProgram
from ._thermal import SaleveThermalProgram

#: Link travel-time corrections.
SLOPE_CORRECTIONS = ("none", "gradient")


class _ActiveField:
    """A public view of the compact, active prefix of a capacity-sized field."""

    def __init__(self, owner, base):
        self._owner, self._base = owner, base

    @property
    def shape(self):
        return self._owner.ny, self._owner.nx

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def array(self):
        return self._base.array[:self._owner.nx * self._owner.ny].reshape(self.shape)

    def from_numpy(self, array):
        arr = np.asarray(array)
        if arr.shape != self.shape or arr.dtype != self.dtype:
            raise ValueError(f"expected {self.shape} {self.dtype}, got {arr.shape} {arr.dtype}")
        self.array[...] = cp.asarray(arr)

    def to_numpy(self):
        return cp.asnumpy(self.array)


class SaleveProgram:
    """CuPy stream-power program whose active resolution changes in place.

    Constructor ``nx, ny, dx`` describe the *finest* level and its capacity.
    Call ``initialize()`` after supplying ``z``. ``restrict()`` halves the
    active shape and doubles ``dx``; ``prolong()`` reverses that hop. The
    original ``z0`` and grid masks are restored per level, while the evolving
    surface is transferred between levels. ``slope_correction="gradient"``
    adjusts link travel time using the local downhill gradient; the default
    ``"none"`` preserves the original solver. ``thermal_erosion`` is the
    paper's talus coefficient (zero disables it); ``critical_slope`` is its
    dimensionless slope threshold. ``hillslope_erosion`` adds the paper's
    Hack-law ridge term (zero disables it; see ``_speed.py``), with
    ``hack_constant`` and ``hack_exponent`` the law's C and h.
    ``cliff_optimization`` applies fixed-network post-correction after each
    high-level solve.
    """

    def __init__(self, backend, *, nx, ny, dx=1.0, m=0.4,
                 uplift=1.0, erodibility=1.0,
                 local_minima="cordonnier_carve", epsilon=1.0e-3,
                 topology="D8", boundary="normal", outlet="edge", nodata=False,
                 jitter=0.25, seed=0,
                 receiver_mode="steepest", receiver_seed=0,
                 slope_correction="none", min_link_slope=1.0e-6,
                 max_slope_correction=100.0,
                 thermal_erosion=0.0, critical_slope=0.57,
                 hillslope_erosion=0.0, hack_constant=1.5,
                 hack_exponent=0.6, cliff_optimization=False,
                 cliff_iterations=50, cliff_learning_rate=0.01,
                 cliff_river_weight=1.0 / 3.0):
        if slope_correction not in SLOPE_CORRECTIONS:
            raise ValueError(f"slope_correction must be one of {SLOPE_CORRECTIONS}")
        if not math.isfinite(min_link_slope) or min_link_slope <= 0:
            raise ValueError("min_link_slope must be finite and positive")
        if not math.isfinite(max_slope_correction) or max_slope_correction < 1:
            raise ValueError("max_slope_correction must be finite and >= 1")
        if not math.isfinite(critical_slope) or critical_slope <= 0:
            raise ValueError("critical_slope must be finite and positive")
        if np.ndim(thermal_erosion) == 0:
            if not math.isfinite(float(thermal_erosion)) or thermal_erosion < 0:
                raise ValueError("thermal_erosion must be finite and non-negative")
        else:
            thermal_array = np.asarray(thermal_erosion)
            if not np.isfinite(thermal_array).all() or (thermal_array < 0).any():
                raise ValueError("thermal_erosion must be finite and non-negative")
        hill_array = np.asarray(hillslope_erosion)
        if not np.isfinite(hill_array).all() or (hill_array < 0).any():
            raise ValueError("hillslope_erosion must be finite and non-negative")
        if not math.isfinite(hack_constant) or hack_constant <= 0:
            raise ValueError("hack_constant must be finite and positive")
        if not math.isfinite(hack_exponent) or hack_exponent < 0:
            raise ValueError("hack_exponent must be finite and non-negative")
        if not isinstance(cliff_iterations, Integral) or cliff_iterations < 0:
            raise ValueError("cliff_iterations must be a non-negative integer")
        if not math.isfinite(cliff_learning_rate) or cliff_learning_rate <= 0:
            raise ValueError("cliff_learning_rate must be finite and positive")
        if not math.isfinite(cliff_river_weight) or not 0 <= cliff_river_weight <= 1:
            raise ValueError("cliff_river_weight must lie in [0, 1]")
        self.nx, self.ny, self.dx = int(nx), int(ny), float(dx)
        self._finest_nx, self._finest_ny = self.nx, self.ny
        self._thermal_option = thermal_erosion
        self._hillslope_option = hillslope_erosion
        self._critical_slope = critical_slope
        self._hack_constant, self._hack_exponent = hack_constant, hack_exponent
        self._cliff_enabled = bool(cliff_optimization)
        self._cliff_iterations = cliff_iterations
        self._cliff_learning_rate = cliff_learning_rate
        self._cliff_river_weight = cliff_river_weight
        self._capacity = self.nx * self.ny
        self._rounds = math.ceil(math.log2(self._capacity))
        self._initialized = False
        self._history = []
        self._outlet_mode = outlet
        self._slope_correction_mode = slope_correction
        self._uplift_field = np.ndim(uplift) != 0
        self._erodibility_field = np.ndim(erodibility) != 0
        self._thermal_field = np.ndim(thermal_erosion) != 0
        self._hillslope_field = np.ndim(hillslope_erosion) != 0
        self._thermal_enabled = self._thermal_field or float(thermal_erosion) > 0
        self.flow = SFDFlowProgram(
            backend, nx=nx, ny=ny, dx=dx, topology=topology,
            boundary=boundary, outlet=outlet, nodata=True,
            dynamic_grid=True, accumulation="pointer_jump_push",
            local_minima=local_minima,
            receiver_mode=receiver_mode, receiver_seed=receiver_seed,
            **({"source": np.ones((ny, nx), dtype=np.float32)}
               if cliff_optimization else {}),
        )
        self.steady = SaleveSteadyProgram(
            backend, nx=nx, ny=ny, dx=dx, m=m,
            uplift=uplift, erodibility=erodibility,
            thermal_erosion=thermal_erosion, critical_slope=critical_slope,
            hillslope_erosion=hillslope_erosion,
            hack_constant=hack_constant, hack_exponent=hack_exponent,
        )
        self.steady.rec.adopt(self.flow.rec.array)
        self.steady.drainage.adopt(self.flow.drainage.array)
        self.steady.slope_correction.adopt(self.flow.slope_correction.array)
        self.flow.slope_correction.array.fill(1.0)
        self.flow.min_link_slope.set(min_link_slope)
        self.flow.max_slope_correction.set(max_slope_correction)
        self.steady.z.adopt(self.flow.z.array)
        self.flow.boundary_z.adopt(self.steady.outlet_z.array)
        self._levels = SaleveLevelProgram(
            backend, nx=nx, ny=ny, boundary=boundary,
            jitter=jitter, seed=seed,
        )
        self._levels.z.adopt(self.flow.z.array)
        self._levels.z0.adopt(self.steady.outlet_z.array)
        self._levels.nodata.adopt(self.flow.nodata_mask.array)
        if outlet == "mask":
            self._levels.outlet.adopt(self.flow.outlet_mask.array)
        if self._uplift_field:
            self._levels.uplift.adopt(self.steady._params["uplift"].handle().array)
        if self._erodibility_field:
            self._levels.erodibility.adopt(
                self.steady._params["erodibility"].handle().array)
        if self._thermal_field:
            self._levels.thermal_erosion.adopt(
                self.steady._params["thermal_erosion"].handle().array)
        if self._hillslope_field:
            self._levels.hillslope_erosion.adopt(
                self.steady._params["hillslope_erosion"].handle().array)
        self.z = _ActiveField(self, self.flow.z)
        self.rec = _ActiveField(self, self.flow.rec)
        self.drainage = _ActiveField(self, self.flow.drainage)
        self.outlet_z = _ActiveField(self, self.steady.outlet_z)
        self.nodata_mask = _ActiveField(self, self.flow.nodata_mask)
        self.outlet_mask = (_ActiveField(self, self.flow.outlet_mask)
                            if outlet == "mask" else None)
        self.uplift = self.steady.uplift
        self.erodibility = self.steady.erodibility
        self.thermal_erosion = self.steady.thermal_erosion
        self.hillslope_erosion = self.steady.hillslope_erosion
        self.critical_slope = self.steady.critical_slope
        self._finite_options = dict(nx=nx, ny=ny, dx=dx, m=m,
                                    epsilon=epsilon, uplift=uplift,
                                    erodibility=erodibility,
                                    hillslope_erosion=hillslope_erosion,
                                    hack_constant=hack_constant,
                                    hack_exponent=hack_exponent)
        self._finite = None
        self._thermal = None
        self._cliff = None
        self._set_active(self.nx, self.ny, self.dx)
        if nodata:
            # The mask is ready for user input; all cells start valid.
            self.nodata_mask.array[...] = 0

    def _set_active(self, nx, ny, dx):
        self.nx, self.ny, self.dx = int(nx), int(ny), float(dx)
        grid = self.flow._bundle_params["grid"]
        grid["NX"].set(self.nx)
        grid["NY"].set(self.ny)
        grid["DX"].set(self.dx)
        n = self.nx * self.ny
        self.flow.active_n.set(n)
        for solver in (self.steady, self._finite, self._thermal):
            if solver is not None:
                solver.active_nx.set(self.nx)
                solver.active_n.set(n)
                solver.active_dx.set(self.dx)
        if self._cliff is not None:
            self._cliff.active_n.set(n)
            cliff_grid = self._cliff._bundle_params["grid"]
            cliff_grid["NX"].set(self.nx)
            cliff_grid["NY"].set(self.ny)
        self._levels.active_nx.set(self.nx)
        self._levels.active_ny.set(self.ny)

    def initialize(self):
        """Freeze the current active terrain as the original elevation z0."""
        if self._history:
            raise RuntimeError("initialize() requires the finest active level")
        n = self.nx * self.ny
        self.steady.outlet_z.array[:n] = self.flow.z.array[:n]
        self._initialized = True

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("call initialize() after setting z")

    def _finite_program(self):
        if self._finite is None:
            finite = SaleveFiniteProgram(
                self.flow._be, levels=self._rounds + 1,
                **self._finite_options,
            )
            finite.rec.adopt(self.flow.rec.array)
            finite.drainage.adopt(self.flow.drainage.array)
            finite.slope_correction.adopt(self.flow.slope_correction.array)
            finite.z0.adopt(self.steady.outlet_z.array)
            finite.z.adopt(self.flow.z.array)
            for name, field in (("uplift", self._uplift_field),
                                ("erodibility", self._erodibility_field),
                                ("hillslope_erosion", self._hillslope_field)):
                if field:
                    finite._params[name].handle().array[...] = (
                        self.steady._params[name].handle().array)
            self._finite = finite
            self._set_active(self.nx, self.ny, self.dx)
        return self._finite

    def _thermal_program(self, finite):
        if self._thermal is None:
            thermal = SaleveThermalProgram(
                self.flow._be, nx=self._finest_nx, ny=self._finest_ny,
                levels=self._rounds + 1,
                m=self._finite_options["m"],
                uplift=self._finite_options["uplift"],
                erodibility=self._finite_options["erodibility"],
                thermal_erosion=self._thermal_option,
                hillslope_erosion=self._hillslope_option,
                critical_slope=self._critical_slope,
                hack_constant=self._hack_constant,
                hack_exponent=self._hack_exponent,
            )
            for name, source in (("rec", self.flow.rec),
                                 ("drainage", self.flow.drainage),
                                 ("slope_correction", self.flow.slope_correction),
                                 ("ancestors", finite._data["ancestors"]),
                                 ("z0", finite._data["conditioned_z0"]),
                                 ("tau", finite._data["tau_a"]),
                                 ("phi", finite._data["phi_a"]),
                                 ("z", self.flow.z)):
                getattr(thermal, name).adopt(source.array)
            self._thermal = thermal
            self._set_active(self.nx, self.ny, self.dx)
            self._sync_fields(self.nx * self.ny)
        return self._thermal

    def _route_and_accumulate(self):
        if self._cliff_enabled:
            self.flow._params["source"].handle().array.fill(1.0)
        self.flow.route()
        self.flow.clear_inactive_receivers()
        self.flow.resolve_minima()
        self.flow.accumulate()

    def _cliff_program(self):
        if self._cliff is None:
            cliff = SaleveCliffProgram(
                self.flow._be, nx=self._finest_nx, ny=self._finest_ny,
                topology=self.flow._config["topology"],
                boundary=self.flow._config["boundary"],
            )
            cliff.rec.adopt(self.flow.rec.array)
            cliff.physical_z.adopt(self.flow.z.array)
            cliff.grad_sum.adopt(self.flow.drainage.array)
            cliff._adopt("source", self.flow._params["source"].handle().array)
            self._cliff = cliff
            self._set_active(self.nx, self.ny, self.dx)
        return self._cliff

    def optimize_cliffs(self, iterations=None):
        """Paper-style fixed-network cliff correction on the current terrain."""
        self._require_initialized()
        if not self._cliff_enabled:
            raise RuntimeError("cliff_optimization=True is required")
        count = self._cliff_iterations if iterations is None else iterations
        if not isinstance(count, Integral) or count < 0:
            raise ValueError("iterations must be a non-negative integer")
        if count == 0:
            return
        n = self.nx * self.ny
        valid = self.flow.nodata_mask.array.ravel()[:n] == 0
        if not bool(cp.any(valid)):
            return
        physical = self.flow.z.array.ravel()[:n]
        minimum = float(cp.min(cp.where(valid, physical, cp.inf)).item())
        maximum = float(cp.max(cp.where(valid, physical, -cp.inf)).item())
        if maximum == minimum:
            return
        cliff = self._cliff_program()
        cliff.minimum.set(minimum)
        cliff.scale.set(maximum - minimum)
        cliff.learning_rate.set(self._cliff_learning_rate)
        cliff.river_weight.set(self._cliff_river_weight)
        cliff._bundle_params["grid"]["NODATA_MASK"].handle().array.ravel()[:n] = (
            self.flow.nodata_mask.array.ravel()[:n])
        cliff._data["area"].array.ravel()[:n] = self.flow.drainage.array.ravel()[:n]
        cliff.initialize()
        try:
            for _ in range(count):
                cliff.local_gradient()
                self.flow.accumulate()  # source is the local cliff gradient
                cliff.update()
                cliff.path_init()
                for k in range(self._rounds):
                    (cliff.jump_a_to_b if k % 2 == 0
                     else cliff.jump_b_to_a)()
                (cliff.finish_a if self._rounds % 2 == 0
                 else cliff.finish_b)()
            cliff.export()
        finally:
            self.flow.drainage.array.ravel()[:n] = cliff._data["area"].array.ravel()[:n]
            self.flow._params["source"].handle().array.fill(1.0)

    def solve_fixed_network(self):
        """Solve steady state on the current receiver forest and drainage."""
        self._require_initialized()
        if self._slope_correction_mode == "gradient":
            self.flow.compute_slope_correction()
        self.steady.initialize()
        for k in range(self._rounds):
            (self.steady.jump_a_to_b if k % 2 == 0
             else self.steady.jump_b_to_a)()
        (self.steady.finish_a if self._rounds % 2 == 0
         else self.steady.finish_b)()

    def run_steady_state(self, n=1):
        """Rebuild the network and solve steady state ``n`` times."""
        self._require_initialized()
        for _ in range(n):
            self._route_and_accumulate()
            self.solve_fixed_network()
        if n and self._cliff_enabled:
            self.optimize_cliffs()

    def solve_finite_time_fixed_network(self, time):
        """Solve at target time on the current receiver forest and drainage."""
        self._require_initialized()
        if not math.isfinite(time) or time < 0:
            raise ValueError("time must be finite and non-negative")
        if time == 0:
            n = self.nx * self.ny
            self.flow.z.array[:n] = self.steady.outlet_z.array[:n]
            return
        finite = self._finite_program()
        if self._slope_correction_mode == "gradient":
            self.flow.compute_slope_correction()
        finite.time.set(time)
        finite.initialize()
        for k in range(self._rounds):
            finite.level.set(k)
            (finite.jump_a_to_b if k % 2 == 0 else finite.jump_b_to_a)()
        depth = "a" if self._rounds % 2 == 0 else "b"

        def path_max():
            for k in range(self._rounds):
                finite.level.set(k)
                (finite.max_jump_a_to_b if k % 2 == 0
                 else finite.max_jump_b_to_a)()

        getattr(finite, f"seed_initial_{depth}")()
        path_max()
        getattr(finite, f"condition_initial_{depth}_{depth}")()
        if self._thermal_enabled:
            thermal = self._thermal_program(finite)
            active_n = self.nx * self.ny
            depths = finite._data[f"depth_{depth}"].array[:active_n]
            max_depth = int(cp.max(depths).item())
            counts = cp.bincount(depths, minlength=max_depth + 1).astype(cp.int32)
            ends = cp.cumsum(counts, dtype=cp.int32)
            thermal._data["order"].array[:active_n] = cp.argsort(depths).astype(cp.int32)
            thermal._data["starts"].array[0] = 0
            thermal._data["starts"].array[1:max_depth + 1] = ends[:-1]
            thermal._data["ends"].array[:max_depth + 1] = ends
            thermal.max_depth.set(max_depth)
            thermal.time.set(time)
            thermal._data["barrier"].array.fill(0)
            thermal.solve()
            return
        getattr(finite, f"finish_{depth}")()
        getattr(finite, f"seed_candidate_{depth}")()
        path_max()
        getattr(finite, f"condition_candidate_{depth}_{depth}")()

    def run_finite_time(self, time, n=1):
        """Refine the network ``n`` times for the same target time."""
        self._require_initialized()
        for _ in range(n):
            self._route_and_accumulate()
            self.solve_finite_time_fixed_network(time)
        if n and time > 0 and self._cliff_enabled:
            self.optimize_cliffs()

    def _run_multigrid(self, *, time, levels, iterations, relaxation):
        self._require_initialized()
        if not isinstance(levels, Integral) or isinstance(levels, bool) or levels < 1:
            raise ValueError("levels must be a positive integer")
        if isinstance(iterations, Integral) and not isinstance(iterations, bool):
            schedule = (int(iterations),) * levels
        else:
            try:
                schedule = tuple(iterations)
            except TypeError as exc:
                raise ValueError("iterations must be an integer or a sequence") from exc
        if len(schedule) != levels or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 0
            for n in schedule
        ):
            raise ValueError("iterations must give a non-negative count per level")
        if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
            raise ValueError("relaxation must be finite and in (0, 1]")
        if time is not None and (not math.isfinite(time) or time < 0):
            raise ValueError("time must be finite and non-negative")
        nx, ny = self.nx, self.ny
        for _ in range(levels - 1):
            if nx < 4 or ny < 4 or nx % 2 or ny % 2:
                raise ValueError("requested levels cannot be reached by 2x restriction")
            nx //= 2
            ny //= 2
        if time == 0:
            self.solve_finite_time_fixed_network(0)
            return

        entered = 0
        self._levels.relaxation.set(relaxation)
        try:
            for _ in range(levels - 1):
                self.restrict()
                entered += 1
            for level, count in enumerate(schedule):
                self._levels.copy_n.set(self.nx * self.ny)
                for _ in range(count):
                    self._levels.snapshot_z()
                    self._route_and_accumulate()
                    if time is None:
                        self.solve_fixed_network()
                    else:
                        self.solve_finite_time_fixed_network(time)
                    self._levels.blend_z()
                if entered:
                    self.prolong()
                    entered -= 1
        finally:
            while entered:
                self.prolong()
                entered -= 1
        if self._cliff_enabled and any(schedule):
            self.optimize_cliffs()

    def run_multigrid_steady(self, *, levels=5, iterations=6, relaxation=0.25):
        """Paper-style coarse-to-fine steady solve; counts run coarse to fine."""
        self._run_multigrid(time=None, levels=levels,
                            iterations=iterations, relaxation=relaxation)

    def run_multigrid_finite_time(self, time, *, levels=5, iterations=6,
                                  relaxation=0.25):
        """Paper-style coarse-to-fine finite-time solve at one target time."""
        self._run_multigrid(time=time, levels=levels,
                            iterations=iterations, relaxation=relaxation)

    def restrict(self):
        """Switch to a 2× coarser active grid, preserving fine-level z0."""
        self._require_initialized()
        nx, ny = self.nx, self.ny
        if nx < 4 or ny < 4 or nx % 2 or ny % 2:
            raise ValueError("restrict requires even nx, ny >= 4")
        n = nx * ny
        masks = {"nodata": cp.copy(self.flow.nodata_mask.array[:n])}
        if self._outlet_mode == "mask":
            masks["outlet"] = cp.copy(self.flow.outlet_mask.array[:n])
        fields = {}
        for name, enabled in (("uplift", self._uplift_field),
                              ("erodibility", self._erodibility_field),
                              ("thermal_erosion", self._thermal_field),
                              ("hillslope_erosion", self._hillslope_field)):
            if enabled:
                fields[name] = cp.copy(self.steady._params[name].handle().array[:n])
        self._history.append((nx, ny, self.dx,
                              cp.copy(self.steady.outlet_z.array[:n]), masks,
                              fields))
        coarse_n = n // 4
        self._levels.copy_n.set(coarse_n)
        for name in ("z", "z0", *fields):
            getattr(self._levels, f"restrict_{name}")()
            getattr(self._levels, f"copy_{name}")()
        for name in masks:
            getattr(self._levels, f"restrict_{name}")()
            getattr(self._levels, f"copy_{name}")()
        self._sync_fields(coarse_n)
        self._set_active(nx // 2, ny // 2, self.dx * 2)
        self.flow.restore_outlets()

    def _sync_fields(self, n):
        if self._finite is not None:
            for name, enabled in (("uplift", self._uplift_field),
                                  ("erodibility", self._erodibility_field),
                                  ("hillslope_erosion", self._hillslope_field)):
                if enabled:
                    self._finite._params[name].handle().array[:n] = (
                        self.steady._params[name].handle().array[:n])
        if self._thermal is not None:
            for name, enabled in (("uplift", self._uplift_field),
                                  ("erodibility", self._erodibility_field),
                                  ("thermal_erosion", self._thermal_field),
                                  ("hillslope_erosion", self._hillslope_field)):
                if enabled:
                    self._thermal._params[name].handle().array[:n] = (
                        self.steady._params[name].handle().array[:n])

    def prolong(self):
        """Return to the previous finer grid, prolonging the working z."""
        self._require_initialized()
        if not self._history:
            raise ValueError("no coarser-level hop to undo")
        nx, ny, dx, z0, masks, fields = self._history[-1]
        self._levels.copy_n.set(nx * ny)
        self._levels.prolong_z()
        self._levels.copy_z()
        n = nx * ny
        self.steady.outlet_z.array[:n] = z0
        for name, snapshot in masks.items():
            mask = getattr(self.flow, f"{name}_mask").array
            mask[:n] = snapshot
            mask[n:] = 1 if name == "nodata" else 0
        for name, snapshot in fields.items():
            self.steady._params[name].handle().array[:n] = snapshot
        self._sync_fields(n)
        self._history.pop()
        self._set_active(nx, ny, dx)
        self.flow.restore_outlets()

    def close(self):
        if self._cliff is not None:
            self._cliff.close()
        if self._thermal is not None:
            self._thermal.close()
        if self._finite is not None:
            self._finite.close()
        self._levels.close()
        self.steady.close()
        self.flow.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["SaleveProgram"]
