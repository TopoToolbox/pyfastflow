"""Small GPU execution checks for the experimental CuPy Programs."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no cupy device", allow_module_level=True)
except Exception:
    pytest.skip("no cupy device", allow_module_level=True)

from pyfastflow.core import Backend
from pyfastflow.graphflood import (
    GraphFloodParticles, GraphFloodRelax, GraphFloodVanilla,
)
from pyfastflow.flow import MFDFlowProgram, SFDFlowProgram
from pyfastflow.visu import HillshadeProgram
from pyfastflow.experimental.golem import GolemNoSedProgram, GolemSedProgram


def _terrain(side):
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    return (
        0.03 * (xx - side / 2) ** 2
        + 0.02 * (yy - side / 2) ** 2
        + 2.0 * np.sin(xx * 0.31) * np.cos(yy * 0.27)
    ).astype(np.float32)


def test_sfd_none_is_executable():
    side = 32
    flow = SFDFlowProgram(
        Backend.from_name("cupy"), nx=side, ny=side, local_minima="none"
    )
    try:
        flow.z.from_numpy(_terrain(side))
        flow.route()
        flow.resolve_minima()
        flow.accumulate()
        result = flow.drainage.to_numpy()
        assert np.isfinite(result).all()
        assert result.min() >= 1.0
    finally:
        flow.close()


def test_golem_nosed_fluvial_step_keeps_outlets_fixed():
    side = 32
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    initial = (100.0 - yy - 0.1 * xx + 0.1 * np.sin(xx)).astype(np.float32)
    golem = GolemNoSedProgram(
        Backend.from_name("cupy"), nx=side, ny=side, topology="D8",
        local_minima="none", erodibility=np.full_like(initial, 1.0e-3),
        hillslope_diffusivity=np.full_like(initial, 1.0e-2), dt=np.float32(10.0),
        diffusion_iterations=12,
    )
    try:
        golem.z.from_numpy(initial)
        golem.initialize()
        golem.run_n_step()
        final = golem.z.to_numpy()
        erosion = golem.erosion_rate.to_numpy()
        assert np.isfinite(final).all() and np.isfinite(erosion).all()
        assert erosion[1:-1, 1:-1].max() > 0.0
        np.testing.assert_array_equal(final[[0, -1], :], initial[[0, -1], :])
        np.testing.assert_array_equal(final[:, [0, -1]], initial[:, [0, -1]])
    finally:
        golem.close()


def test_golem_nosed_d4_periodic_mask_and_nodata():
    side = 16
    initial = -np.arange(side, dtype=np.float32)[:, None] + np.zeros(
        (side, side), dtype=np.float32,
    )
    outlet = np.zeros_like(initial, dtype=np.uint8)
    outlet[-1, 0] = 1
    nodata = np.zeros_like(initial, dtype=np.uint8)
    nodata[4, 4] = 1
    golem = GolemNoSedProgram(
        Backend.from_name("cupy"), nx=side, ny=side, topology="D4",
        boundary="periodic_EW", outlet="mask", nodata=True,
        local_minima="none", diffusion_iterations=3,
    )
    try:
        golem.z.from_numpy(initial)
        golem.outlet_mask.from_numpy(outlet)
        golem.nodata_mask.from_numpy(nodata)
        golem.initialize()
        golem.run_n_step()
        assert np.isfinite(golem.z.to_numpy()).all()
        assert golem.z.to_numpy()[-1, 0] == initial[-1, 0]
        assert golem.drainage_area.to_numpy()[4, 4] == 0.0
    finally:
        golem.close()


def test_golem_sed_sfd_step_conserves_sediment_flux():
    side = 32
    dx = 3.0
    dt = np.float32(2.0)
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    initial = (100.0 - yy - 0.01 * xx).astype(np.float32)
    thickness = np.full_like(initial, 0.25)
    golem = GolemSedProgram(
        Backend.from_name("cupy"), nx=side, ny=side, dx=dx,
        local_minima="none", dt=dt,
        rock_erodibility=np.full_like(initial, 2.0e-3),
        sediment_erodibility=np.full_like(initial, 3.0e-3),
        transport_length=np.full_like(initial, 12.0),
    )
    try:
        golem.z.from_numpy(initial)
        golem.sediment_thickness.from_numpy(thickness)
        golem.initialize()
        golem.run_n_step()
        final = golem.z.to_numpy()
        hsed = golem.sediment_thickness.to_numpy()
        source = golem.sediment_source_rate.to_numpy().sum(dtype=np.float64) * dt
        deposited = (
            golem.deposition_rate.to_numpy().sum(dtype=np.float64) * dx * dx * dt
        )
        exported = float(golem.sediment_export_rate.to_numpy()[0]) * dt
        assert np.isfinite(final).all() and np.isfinite(hsed).all()
        assert hsed.min() >= 0.0
        assert golem.rock_erosion_rate.to_numpy()[1:-1, 1:-1].max() > 0.0
        np.testing.assert_allclose(source, deposited + exported, rtol=5e-5, atol=2e-5)
        np.testing.assert_array_equal(final[[0, -1], :], initial[[0, -1], :])
        np.testing.assert_array_equal(final[:, [0, -1]], initial[:, [0, -1]])
    finally:
        golem.close()


def test_golem_sed_d4_masked_diffusion_preserves_outlet():
    side = 16
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    initial = (50.0 - yy + 2.0 * np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 4.0))
    initial = initial.astype(np.float32)
    outlet = np.zeros_like(initial, dtype=np.uint8)
    outlet[-1, :] = 1
    nodata = np.zeros_like(initial, dtype=np.uint8)
    nodata[5, 5] = 1
    golem = GolemSedProgram(
        Backend.from_name("cupy"), nx=side, ny=side, topology="D4",
        boundary="periodic_EW", outlet="mask", nodata=True,
        local_minima="none", hillslope_model="linear_implicit",
        hillslope_diffusivity=np.full_like(initial, 0.2), dt=np.float32(1.0),
        diffusion_iterations=5,
    )
    try:
        golem.z.from_numpy(initial)
        golem.sediment_thickness.from_numpy(np.full_like(initial, 0.2))
        golem.outlet_mask.from_numpy(outlet)
        golem.nodata_mask.from_numpy(nodata)
        golem.initialize()
        golem.run_n_step()
        final = golem.z.to_numpy()
        hsed = golem.sediment_thickness.to_numpy()
        assert np.isfinite(final).all() and np.isfinite(hsed).all()
        assert hsed.min() >= 0.0
        np.testing.assert_array_equal(final[-1, :], initial[-1, :])
        assert final[5, 5] == initial[5, 5]
        assert hsed[5, 5] == np.float32(0.2)
    finally:
        golem.close()


@pytest.mark.parametrize("topology", ["D4", "D8"])
def test_golem_shared_stream_power_matches_direct_linear_system(topology):
    side = 8
    dx = 2.0
    dt = 3.0
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    initial = (50.0 - 0.7 * yy + 0.08 * xx + 0.2 * np.sin(xx)).astype(np.float32)
    cover = np.full_like(initial, 0.25)
    kd = (0.004 + 0.0002 * xx).astype(np.float32)
    kt = (0.003 + 0.0003 * yy).astype(np.float32)
    model = GolemSedProgram(
        Backend.from_name("cupy"), nx=side, ny=side, dx=dx,
        topology=topology, boundary="periodic_EW", outlet="mask",
        local_minima="none", fluvial_model="shared_stream_power",
        detachment_erodibility=kd, transport_erodibility=kt,
        dt=np.float32(dt), m=np.float32(0.5),
    )
    outlet = np.zeros_like(initial, dtype=np.uint8)
    outlet[-1, :] = 1
    try:
        model.z.from_numpy(initial)
        model.sediment_thickness.from_numpy(cover)
        model.outlet_mask.from_numpy(outlet)
        model.initialize()
        model.run_n_step()
        rec = model.rec.to_numpy().ravel()
        area = model.drainage_area.to_numpy().ravel()
        z0 = initial.ravel().astype(np.float64)
        n = side * side
        lhs = np.zeros((2 * n, 2 * n), dtype=np.float64)
        rhs = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            r = int(rec[i])
            is_outlet = bool(outlet.ravel()[i])
            if is_outlet:
                lhs[2 * i, 2 * i] = 1.0
                rhs[2 * i] = z0[i]
            else:
                lhs[2 * i, 2 * i] = dx * dx / dt
                lhs[2 * i, 2 * i + 1] = 1.0
                rhs[2 * i] = dx * dx * z0[i] / dt
            for j in np.flatnonzero(rec == i):
                if j != i:
                    lhs[2 * i + int(is_outlet), 2 * j + 1] -= 1.0
            if is_outlet:
                lhs[2 * i + 1, 2 * i + 1] = 1.0
            elif r == i:
                lhs[2 * i + 1, 2 * i + 1] = 1.0
            else:
                iy, ix = divmod(i, side)
                ry, rx = divmod(r, side)
                span_x = min(abs(ix - rx), side - abs(ix - rx))
                distance = dx * np.hypot(span_x, abs(iy - ry))
                phi = float(kd.ravel()[i]) * area[i] ** 0.5
                psi = float(kd.ravel()[i]) / (float(kt.ravel()[i]) * area[i])
                lhs[2 * i + 1, 2 * i] = 1.0 / dt + phi / distance
                lhs[2 * i + 1, 2 * r] = -phi / distance
                lhs[2 * i + 1, 2 * i + 1] = -psi
                rhs[2 * i + 1] = z0[i] / dt
        expected = np.linalg.solve(lhs, rhs)
        np.testing.assert_allclose(
            model.z.to_numpy().ravel(), expected[::2], rtol=2e-5, atol=2e-4,
        )
        np.testing.assert_allclose(
            model.sediment_flux.to_numpy().ravel(), expected[1::2],
            rtol=2e-4, atol=2e-4,
        )
        source = model.sediment_source_rate.to_numpy().sum() * dt
        deposit = model.deposition_rate.to_numpy().sum() * dx * dx * dt
        exported = model.sediment_export_rate.to_numpy()[0] * dt
        np.testing.assert_allclose(source, deposit + exported,
                                   rtol=3e-4, atol=3e-4)
        assert model.sediment_thickness.to_numpy().min() >= 0.0
    finally:
        model.close()


@pytest.mark.parametrize(
    "method",
    ["none", "reconstruct_epsilon", "cordonnier_carve", "fill_cordonnier"],
)
def test_mfd_program_pipeline(method):
    side = 32
    flow = MFDFlowProgram(
        Backend.from_name("cupy"), nx=side, ny=side, local_minima=method
    )
    try:
        flow.z.from_numpy(_terrain(side))
        if method in ("cordonnier_carve", "fill_cordonnier"):
            flow.route()
        if method == "cordonnier_carve":
            flow.snapshot_receivers()
        flow.resolve_minima()
        flow.prepare_mfd_surface()
        flow.build_topology()
        flow.accumulate()
        result = flow.drainage.to_numpy()
        assert np.isfinite(result).all()
        assert result.min() >= 1.0
        if method != "none":
            edge = np.zeros((side, side), dtype=bool)
            edge[[0, -1], :] = True
            edge[:, [0, -1]] = True
            assert result[edge].sum() == pytest.approx(side * side, rel=3e-4)
        if method == "fill_cordonnier":
            assert np.all(flow.filled.to_numpy() >= flow.z.to_numpy())
    finally:
        flow.close()


def test_mfd_program_float32_weight_option():
    side = 16
    flow = MFDFlowProgram(
        Backend.from_name("cupy"), nx=side, ny=side,
        local_minima="none", quantized_weight=False,
    )
    try:
        flow.z.from_numpy(_terrain(side))
        flow.resolve_minima()
        flow.build_topology()
        flow.accumulate()
        assert np.isfinite(flow.drainage.to_numpy()).all()
    finally:
        flow.close()


@pytest.mark.parametrize("method", ["hillshade", "multishade"])
def test_hillshade_program_flat_surface(method):
    side = 16
    shading = HillshadeProgram(
        Backend.from_name("cupy"), nx=side, ny=side, method=method, altitude=45.0
    )
    try:
        shading.z.from_numpy(np.zeros((side, side), dtype=np.float32))
        shading.render()
        image = shading.image.to_numpy()
        np.testing.assert_allclose(image, np.sqrt(0.5), rtol=2e-6, atol=2e-6)
    finally:
        shading.close()


@pytest.mark.parametrize(
    "mfd_local_minima",
    ["rank_cordonnier", "fill_cordonnier", "carve_cordonnier",
     "reconstruct_epsilon"],
)
def test_graphflood_vanilla_two_steps(mfd_local_minima):
    side = 16
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    terrain = (100.0 - xx - 0.25 * yy).astype(np.float32)
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        mfd_local_minima=mfd_local_minima,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.precipitation.set(1.0e-5)
        flood.dt.set(0.1)
        flood.reset_h()
        flood.run(2)
        h = flood.h.to_numpy()
        qi = flood.Qi.to_numpy()
        qo = flood.Qo.to_numpy()
        assert np.isfinite(h).all() and np.isfinite(qi).all() and np.isfinite(qo).all()
        assert h.min() >= 0.0
        assert qi.max() > 0.0
        np.testing.assert_allclose(h[[0, -1], :], 0.0)
        np.testing.assert_allclose(h[:, [0, -1]], 0.0)
    finally:
        flood.close()


def test_graphflood_routes_sub_float32_head_difference():
    side = 8
    terrain = np.full((side, side), 1000.0, dtype=np.float32)
    depth = np.zeros_like(terrain)
    depth[3, 3] = 2.0e-5
    assert np.float32(terrain[3, 3] + depth[3, 3]) == terrain[3, 3]

    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=1.0,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.h.from_numpy(depth)
        flood.make_surface()
        flood.route()
        assert flood.rec.to_numpy()[3, 3] != 3 * side + 3
    finally:
        flood.close()


def test_graphflood_fill_preserves_mass_with_sub_float32_relief():
    side = 32
    terrain = np.full((side, side), 1300.0, dtype=np.float32)
    depth = (np.random.default_rng(4).random((side, side)) * 4.0e-5).astype(
        np.float32,
    )
    assert np.unique((terrain + depth).astype(np.float32)).size == 1

    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=1.0,
        mfd_local_minima="fill_cordonnier",
    )
    try:
        flood.z.from_numpy(terrain)
        flood.h.from_numpy(depth)
        flood.precipitation.set(1.0)
        flood.dt.set(0.0)
        flood.run()
        qi = flood.Qi.to_numpy()
        edge = np.zeros_like(qi, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        assert qi[edge].sum(dtype=np.float64) == pytest.approx(
            side * side, rel=3e-4,
        )
    finally:
        flood.close()


def test_graphflood_d4_full_pipeline():
    side = 16
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        topology="D4", boundary="periodic_EW",
        mfd_local_minima="rank_cordonnier",
    )
    try:
        flood.z.from_numpy((100.0 - xx - 0.25 * yy).astype(np.float32))
        flood.precipitation.set(1.0e-5)
        flood.reset_h()
        flood.run()
        assert np.isfinite(flood.h.to_numpy()).all()
        assert flood.Qi.to_numpy().max() > 0.0
    finally:
        flood.close()


@pytest.mark.parametrize("analytical_solver", ["local", "bottom_up"])
def test_graphflood_relax_two_steps(analytical_solver):
    side = 16
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    terrain = (100.0 - xx - 0.25 * yy).astype(np.float32)
    flood = GraphFloodRelax(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        mfd_local_minima="reconstruct_epsilon",
        analytical_solver=analytical_solver,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.precipitation.set(1.0e-5)
        flood.relaxation.set(0.25)
        flood.reset_h()
        flood.warmup(2)
        flood.run(2)
        h = flood.h.to_numpy()
        qi = flood.Qi.to_numpy()
        qo = flood.Qo.to_numpy()
        assert all(np.isfinite(a).all() for a in (h, qi, qo))
        assert h.min() >= 0.0
        assert qi.max() > 0.0
        np.testing.assert_allclose(h[[0, -1], :], 0.0)
        np.testing.assert_allclose(h[:, [0, -1]], 0.0)
    finally:
        flood.close()


def test_graphflood_analytical_inverse_matches_frozen_flux():
    side = 32
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    terrain = (100.0 - xx - 0.25 * yy).astype(np.float32)
    flood = GraphFloodRelax(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        mfd_local_minima="fill_cordonnier",
    )
    try:
        flood.z.from_numpy(terrain)
        flood.precipitation.set(1.0e-5)
        flood.relaxation.set(1.0)
        flood.reset_h()
        flood.run()
        qi = flood.Qi.to_numpy()[1:-1, 1:-1]
        qo = flood.Qo.to_numpy()[1:-1, 1:-1]
        # z and h are float32; forming their small hydraulic-surface
        # difference limits the attainable flux closure precision.
        np.testing.assert_allclose(qo, qi, rtol=5e-4, atol=1e-10)
    finally:
        flood.close()


def test_graphflood_bottom_up_damps_one_complete_candidate():
    side = 32
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    terrain = (100.0 - xx - 0.25 * yy).astype(np.float32)
    results = []
    for relaxation in (1.0, 0.25):
        flood = GraphFloodRelax(
            Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
            mfd_local_minima="rank_cordonnier",
            analytical_solver="bottom_up",
        )
        try:
            flood.z.from_numpy(terrain)
            flood.precipitation.set(1.0e-5)
            flood.relaxation.set(relaxation)
            flood.reset_h()
            flood.run()
            results.append(flood.h.to_numpy())
        finally:
            flood.close()
    np.testing.assert_allclose(results[1], 0.25 * results[0], rtol=2e-4)


def test_graphflood_d4_periodic_and_masked_grid():
    side = 16
    terrain = np.zeros((side, side), dtype=np.float32)
    initial = np.zeros_like(terrain)
    initial[side // 2, 0] = 1.0
    outlet = np.zeros_like(terrain, dtype=np.uint8)
    outlet[0, 0] = 1
    nodata = np.zeros_like(terrain, dtype=np.uint8)
    nodata[0, 1] = 1
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        topology="D4", boundary="periodic_EW", outlet="mask", nodata=True,
        dt=1.0e-3,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.h.from_numpy(initial)
        flood.outlet_mask.from_numpy(outlet)
        flood.nodata_mask.from_numpy(nodata)
        flood.run_transient()
        after = flood.h.to_numpy()
        # The westward flux wraps from column zero to the last column.
        assert after[side // 2, -1] > 0.0
        assert after[0, 0] == 0.0
        assert after[0, 1] == 0.0
        np.testing.assert_array_equal(flood.outlet_mask.to_numpy(), outlet)
        np.testing.assert_array_equal(flood.nodata_mask.to_numpy(), nodata)
    finally:
        flood.close()


def test_graphflood_can_initialize_depth_from_reconstructed_fill():
    side = 16
    terrain = np.full((side, side), 10.0, dtype=np.float32)
    terrain[[0, -1], :] = 0.0
    terrain[:, [0, -1]] = 0.0
    terrain[side // 2, side // 2] = 2.0
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=1.0,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.initialize_h_from_fill()
        h = flood.h.to_numpy()
        assert np.isfinite(h).all()
        assert h[side // 2, side // 2] == pytest.approx(8.0)
        np.testing.assert_allclose(h[[0, -1], :], 0.0)
        np.testing.assert_allclose(h[:, [0, -1]], 0.0)
    finally:
        flood.close()


def test_graphflood_can_fill_current_hydraulic_surface():
    side = 16
    terrain = np.full((side, side), 10.0, dtype=np.float32)
    terrain[[0, -1], :] = 0.0
    terrain[:, [0, -1]] = 0.0
    terrain[side // 2, side // 2] = 2.0
    initial_h = np.full_like(terrain, 0.25)
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=1.0,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.h.from_numpy(initial_h)
        flood.fill_hydraulic_surface()
        h = flood.h.to_numpy()
        assert np.isfinite(h).all()
        assert h[side // 2, side // 2] == pytest.approx(8.25)
        unchanged = np.ones_like(terrain, dtype=bool)
        unchanged[side // 2, side // 2] = False
        np.testing.assert_allclose(h[unchanged], 0.25)
    finally:
        flood.close()


def test_graphflood_fill_cordonnier_pipeline():
    side = 16
    terrain = np.full((side, side), 10.0, dtype=np.float32)
    terrain[[0, -1], :] = 0.0
    terrain[:, [0, -1]] = 0.0
    terrain[side // 2, side // 2] = 2.0
    flood = GraphFloodVanilla(
        Backend.from_name("cupy"), nx=side, ny=side, dx=1.0,
        mfd_local_minima="fill_cordonnier",
    )
    try:
        flood.z.from_numpy(terrain)
        flood.reset_h()
        flood.precipitation.set(1.0)
        flood.dt.set(0.0)
        flood.run()
        h = flood.h.to_numpy()
        qi = flood.Qi.to_numpy()
        assert np.isfinite(h).all() and np.isfinite(qi).all()
        assert h[side // 2, side // 2] == pytest.approx(8.0)
        edge = np.zeros((side, side), dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        assert qi[edge].sum() == pytest.approx(side * side, rel=3e-4)
    finally:
        flood.close()


@pytest.mark.parametrize("topology,boundary", [
    ("D8", "normal"), ("D4", "periodic_EW"),
])
def test_graphflood_particles_runs(topology, boundary):
    side = 32
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    terrain = (100.0 - xx - 0.25 * yy).astype(np.float32)
    flood = GraphFloodParticles(
        Backend.from_name("cupy"), nx=side, ny=side, dx=2.0,
        topology=topology, boundary=boundary, precipitation=1.0e-5,
        n_particles=4096, threads=1024,
    )
    try:
        flood.z.from_numpy(terrain)
        flood.fill_topography()
        assert flood.warmup() > 0
        stats = flood.run(2)
        result = flood.finish()
        h, qi, qo = (flood.h.to_numpy(), flood.Qi.to_numpy(),
                     flood.Qo.to_numpy())
        assert all(np.isfinite(a).all() for a in (h, qi, qo))
        assert h.min() >= 0.0
        assert stats["launched"] == pytest.approx(1.0)
        assert np.isfinite(result["residual"]) and result["cells"] > 0
    finally:
        flood.close()
