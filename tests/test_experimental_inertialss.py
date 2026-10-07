"""GPU checks for the standalone stationary D4 flood experiment."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CuPy device", allow_module_level=True)
except Exception:
    pytest.skip("no CuPy device", allow_module_level=True)

from pyfastflow.core import Backend
from pyfastflow.flood import InertialFloodProgram
from experimental.inertialss import SteadyInertialProgram, solve_steady


def test_periodic_plane_preserves_inlet_flux_to_masked_outlet():
    nx, ny, dx = 12, 20, 10.0
    top_inflow = 0.1
    z = np.broadcast_to(
        -0.01 * dx * np.arange(ny, dtype=np.float32)[:, None],
        (ny, nx),
    ).copy()
    outlet = np.zeros((ny, nx), dtype=np.uint8)
    outlet[-1] = 1
    model = SteadyInertialProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny, dx=dx,
        boundary="periodic_EW", outlet="mask", outlet_depth=0.0,
    )
    try:
        model.z.from_numpy(z)
        model.h.from_numpy(np.full((ny, nx), 0.04, dtype=np.float32))
        model.outlet_mask.from_numpy(outlet)
        model.top_inflow.set(top_inflow)
        info = solve_steady(model, mass_tolerance=1e-5)
        assert info.converged, info
        h = model.h.to_numpy()
        qx = model.qx.to_numpy()
        qy = model.qy.to_numpy()
        assert np.min(h) >= 0.0
        np.testing.assert_array_equal(h[-1], 0.0)
        np.testing.assert_allclose(qx, 0.0, atol=1e-5)
        np.testing.assert_allclose(qy[1:ny] * dx, top_inflow,
                                   rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(qy[0], 0.0)
        np.testing.assert_allclose(qy[-1], 0.0)

        # With the numerical caps inactive, this should also be a stationary
        # state of the existing unfiltered local-inertia program.
        transient = InertialFloodProgram(
            Backend.from_name("cupy"), nx=nx, ny=ny, dx=dx,
            topology="D4", boundary="periodic_EW", outlet="mask",
            outlet_depth=0.0, regulator="bates",
        )
        try:
            transient.z.from_numpy(z)
            transient.h.from_numpy(h.astype(np.float32))
            transient.qx.from_numpy(qx.astype(np.float32))
            transient.qy.from_numpy(qy.astype(np.float32))
            transient.outlet_mask.from_numpy(outlet)
            transient.top_inflow.set(top_inflow)
            transient.manning.set(model.manning.read())
            transient.min_flow_depth.set(model.min_flow_depth.read())
            transient.froude_limit.set(100.0)
            transient.transfer_fraction.set(1.0)
            transient.dt.set(0.01)
            transient.step(5)
            np.testing.assert_allclose(transient.h.to_numpy(), h,
                                       atol=3e-6, rtol=0)
            np.testing.assert_allclose(transient.qy.to_numpy(), qy,
                                       atol=3e-5, rtol=0)
        finally:
            transient.close()
    finally:
        model.close()


def test_closed_domain_reports_no_steady_state():
    nx = ny = 8
    model = SteadyInertialProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny,
        outlet="mask", outlet_depth=0.0,
    )
    try:
        model.outlet_mask.from_numpy(np.zeros((ny, nx), dtype=np.uint8))
        model.z.from_numpy(np.zeros((ny, nx), dtype=np.float32))
        model.h.from_numpy(np.zeros((ny, nx), dtype=np.float32))
        model.top_inflow.set(0.1)
        info = solve_steady(model)
        assert not info.converged
        assert "no outlet" in info.message
    finally:
        model.close()


@pytest.mark.parametrize("linear_solver", ["gmres", "direct"])
def test_flat_bed_produces_a_backwater_gradient(linear_solver):
    nx, ny, dx = 12, 20, 10.0
    outlet = np.zeros((ny, nx), dtype=np.uint8)
    outlet[-1] = 1
    model = SteadyInertialProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny, dx=dx,
        boundary="periodic_EW", outlet="mask", outlet_depth=0.0,
    )
    try:
        model.z.from_numpy(np.zeros((ny, nx), dtype=np.float32))
        model.h.from_numpy(np.full((ny, nx), 0.1, dtype=np.float32))
        model.outlet_mask.from_numpy(outlet)
        model.top_inflow.set(0.1)
        info = solve_steady(model, mass_tolerance=1e-5,
                            max_outer=80, max_linear=300,
                            linear_solver=linear_solver)
        assert info.converged, info
        h = model.h.to_numpy()
        assert h[0, 0] > h[ny // 2, 0] > h[-1, 0]
        np.testing.assert_allclose(model.qy.to_numpy()[1:ny] * dx,
                                   0.1, rtol=2e-3, atol=2e-4)
    finally:
        model.close()


def test_perturbed_terrain_routes_around_nodata():
    nx, ny, dx = 24, 24, 10.0
    yy, xx = np.mgrid[:ny, :nx].astype(np.float32)
    z = (-0.01 * dx * yy + 0.025 * np.sin(xx * 0.5)
         * np.cos(yy * 0.4)).astype(np.float32)
    outlet = np.zeros((ny, nx), dtype=np.uint8)
    outlet[-1] = 1
    nodata = np.zeros((ny, nx), dtype=np.uint8)
    nodata[8:11, 11:14] = 1
    model = SteadyInertialProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny, dx=dx,
        boundary="periodic_EW", outlet="mask", nodata=True,
        outlet_depth=0.0,
    )
    try:
        model.z.from_numpy(z)
        model.h.from_numpy(np.full((ny, nx), 0.04, dtype=np.float32))
        model.outlet_mask.from_numpy(outlet)
        model.nodata_mask.from_numpy(nodata)
        model.top_inflow.set(0.1)
        model.linear_shift.set(0.1)
        info = solve_steady(model, mass_tolerance=1e-4,
                            max_outer=80, max_linear=300)
        assert info.converged, info
        h = model.h.to_numpy()
        assert np.all(h[nodata != 0] == 0.0)
        assert np.all(h[nodata == 0] >= 0.0)
        assert np.max(np.abs(model.qx.to_numpy())) > 0.0
        qy = model.qy.to_numpy()
        assert np.sum(qy[-2] * dx) == pytest.approx(nx * 0.1, rel=2e-3)
    finally:
        model.close()
