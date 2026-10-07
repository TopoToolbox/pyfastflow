"""CuPy checks for D4/D8 local-inertia flood programs."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no cupy device", allow_module_level=True)
except Exception:
    pytest.skip("no cupy device", allow_module_level=True)

from pyfastflow.core import Backend
from pyfastflow.flood import InertialFloodProgram


def test_d4_keeps_diagonal_storage_empty_and_rainfall_is_exact():
    ny, nx = 12, 16
    flood = InertialFloodProgram(Backend.from_name("cupy"), nx=nx, ny=ny, dx=2.0)
    try:
        assert flood.topology == "D4"
        assert flood.qxy.shape == (0, 0)
        assert flood.qyx.shape == (0, 0)
        flood.z.from_numpy(np.zeros((ny, nx), dtype=np.float32))
        flood.dt.set(0.25)
        flood.rainfall.set(2.0e-3)
        flood.reset()
        flood.step(4)
        np.testing.assert_allclose(flood.h.to_numpy(), 2.0e-3)
        np.testing.assert_allclose(flood.qx.to_numpy(), 0.0)
        np.testing.assert_allclose(flood.qy.to_numpy(), 0.0)
    finally:
        flood.close()


@pytest.mark.parametrize("topology", ["D4", "D8"])
@pytest.mark.parametrize(
    "regulator",
    ["bates", "q_upwind", "q_centered", "s_upwind", "s_centered"],
)
def test_internal_links_conserve_periodic_water_volume(topology, regulator):
    ny = nx = 20
    yy, xx = np.mgrid[:ny, :nx].astype(np.float32)
    z = (0.3 * np.sin(xx * 0.4) + 0.2 * np.cos(yy * 0.3)).astype(np.float32)
    h = (0.1 + 0.03 * np.sin(xx * 0.2 + yy * 0.1)).astype(np.float32)
    flood = InertialFloodProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny, dx=2.0,
        topology=topology, boundary="periodic_EW", regulator=regulator,
    )
    try:
        flood.z.from_numpy(z)
        flood.h.from_numpy(h)
        flood.dt.set(0.01)
        flood.regulator_theta.set(0.9)
        before = flood.h.to_numpy().sum(dtype=np.float64)
        flood.step(2)
        after = flood.h.to_numpy().sum(dtype=np.float64)
        assert after == pytest.approx(before, rel=2e-6, abs=2e-6)
        if topology == "D8":
            assert flood.qxy.shape == (ny, nx)
            assert flood.qyx.shape == (ny, nx)
    finally:
        flood.close()


@pytest.mark.parametrize("topology", ["D4", "D8"])
@pytest.mark.parametrize("regulator", ["q_upwind", "q_centered"])
def test_regulator_theta_one_is_exactly_bates(topology, regulator):
    """Filtering must be an identity operation at theta=1."""
    ny, nx = 11, 13
    yy, xx = np.mgrid[:ny, :nx].astype(np.float32)
    z = (0.1 * xx - 0.06 * yy + 0.02 * np.sin(xx + yy)).astype(np.float32)
    h = (0.08 + 0.02 * np.cos(xx * 0.3 - yy * 0.2)).astype(np.float32)
    common = dict(nx=nx, ny=ny, dx=2.0, topology=topology,
                  boundary="periodic_EW")
    bates = InertialFloodProgram(Backend.from_name("cupy"), **common)
    filtered = InertialFloodProgram(
        Backend.from_name("cupy"), regulator=regulator, **common,
    )
    try:
        for flood in (bates, filtered):
            flood.reset()
            flood.z.from_numpy(z)
            flood.h.from_numpy(h)
            flood.dt.set(0.01)
        filtered.regulator_theta.set(1.0)
        bates.step(4)
        filtered.step(4)
        for name in ("h", "qx", "qy", "qxy", "qyx"):
            np.testing.assert_array_equal(
                getattr(filtered, name).to_numpy(), getattr(bates, name).to_numpy(),
            )
    finally:
        bates.close()
        filtered.close()


def test_nodata_and_masked_outlet_are_grid_owned():
    ny = nx = 10
    flood = InertialFloodProgram(
        Backend.from_name("cupy"), nx=nx, ny=ny, dx=1.0,
        topology="D8", outlet="mask", nodata=True, outlet_depth=0.0,
    )
    try:
        nodata = np.zeros((ny, nx), dtype=np.uint8)
        nodata[2, 3] = 1
        outlet = np.zeros((ny, nx), dtype=np.uint8)
        outlet[5, 5] = 1
        flood.nodata_mask.from_numpy(nodata)
        flood.outlet_mask.from_numpy(outlet)
        flood.z.from_numpy(np.zeros((ny, nx), dtype=np.float32))
        flood.h.from_numpy(np.ones((ny, nx), dtype=np.float32))
        flood.dt.set(0.25)
        flood.min_depth.set(0.1)
        flood.step()
        h = flood.h.to_numpy()
        assert h[2, 3] == 0.0
        assert h[5, 5] == 0.0
        np.testing.assert_array_equal(flood.nodata_mask.to_numpy(), nodata)
        np.testing.assert_array_equal(flood.outlet_mask.to_numpy(), outlet)
    finally:
        flood.close()
