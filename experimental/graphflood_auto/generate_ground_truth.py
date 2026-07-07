"""
CLI to run graphflood to a manually-judged steady state and dump a ground-truth
h/Q/z stack for a given DEM/precip/manning combo, to be used later for
validating an automated run_to_convergence() heuristic.

Usage:
    python generate_ground_truth.py greenriver
    python generate_ground_truth.py /path/to/dem.tif --precip 2.7e-6 --manning 0.033 --every 200

Author: B.G.
"""

import argparse
import json
import os
import subprocess
import sys
from math import ceil, log2

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import topotoolbox as ttb
from matplotlib.widgets import Button

import pyfastflow.constants as cte
from pyfastflow import tp
from pyfastflow.flow import FlowContext
from pyfastflow.flood import FloodContext
from pyfastflow.grid import GridContext
from pyfastflow.visu import VisuContext

PERCENTILES = [50, 75, 80, 85, 90, 95, 98, 99]


def parse_args():
    p = argparse.ArgumentParser(description="Generate graphflood ground truth (h/Q/z) for a DEM.")
    p.add_argument("dem", type=str, help="DEM name (topotoolbox sample) or path to a .tif file")
    p.add_argument("--precip", type=float, default=10.0, help="Precipitation rate [mm/h]")
    p.add_argument("--manning", type=float, default=0.033, help="Manning roughness coefficient")
    p.add_argument("--dt", type=float, default=1e-3, help="Hydro time step [s]")
    p.add_argument("--propagate-first", action="store_true", help="Run propagate() once before the local passes each outer step")
    p.add_argument("--n-local", type=int, default=100, help="Number of local passes per outer step")
    p.add_argument("--n-solve-lm", type=int, default=0, help="solve_lm() calls per local pass")
    p.add_argument("--n-distribute", type=int, default=1, help="distribute() calls per local pass")
    p.add_argument("--n-core", type=int, default=1, help="core() calls per local pass")
    p.add_argument("--no-panel", action="store_true", help="Don't auto-launch param_panel.py as a subprocess")
    p.add_argument(
        "--lm",
        type=str,
        default=None,
        choices=["fill", "breach"],
        help="CPU depression preprocessing on the DEM before use: fill (priority-flood, slope 1e-3) or breach",
    )
    return p.parse_args()


def load_dem(dem_arg):
    root, ext = os.path.splitext(dem_arg)
    if ext == "":
        dem = ttb.load_dem(dem_arg)
        name = dem_arg
    else:
        dem = ttb.read_tif(dem_arg)
        name = os.path.basename(root)
    return dem, name


def preprocess_lm(z_2d, mode):
    """CPU depression preprocessing. Returns (processed_z_2d, dh_2d) where
    dh = processed - original (kept for info only, not applied elsewhere).
    """
    if mode is None:
        return z_2d, np.zeros_like(z_2d)

    from nbmdsa.quick import breach, priority_flood

    if mode == "fill":
        processed = priority_flood(z_2d, epsilon=1e-3).reshape(z_2d.shape)
    else:
        processed = breach(z_2d).reshape(z_2d.shape)

    dh = processed - z_2d
    return processed, dh


def main():
    args = parse_args()
    dem, dem_name = load_dem(args.dem)
    precrate = args.precip * 1e-3 / 3600  # mm/h -> m/s
    manning = args.manning
    dt_hydro = args.dt

    NX, NY, DX = dem.columns, dem.rows, dem.cellsize
    N = NX * NY

    ti.init(arch=ti.gpu, offline_cache=False)

    gridctx = GridContext(NX, NY, DX, boundary_mode="normal", topology="D8")
    flowctx = FlowContext(
        gridctx,
        weight_mode="const",
        weight=1.0,
        min_slope_mode="const",
        min_slope=1e-2,
        diagonal_partition_correction=True,
    )
    floodctx = FloodContext(
        gridctx,
        flowctx=flowctx,
        dth_mode="scalar",
        dth=dt_hydro,
        source_w_mode="const",
        source_w=precrate,
        source_w_kind="precip",
        friction_coeff_mode="const",
        friction_coeff=manning,
        boundary_h_mode="const",
        boundary_h=0.0,
        gf_min_increment=0.,
    )

    visuctx = VisuContext(gridctx)
    accum = floodctx._accum_flowctx
    logn = flowctx.logn

    # --- persistent fields ---
    z = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    h = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    receivers = tp.get_tpfield(ti.i32, N)
    Q = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    Q_next = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    Qo = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    surface = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    out_sum = tp.get_tpfield(cte.FLOAT_TYPE_TI, ())
    in_sum = tp.get_tpfield(cte.FLOAT_TYPE_TI, ())

    # reroute temps
    bid = tp.get_tpfield(ti.i32, N)
    rec_work = tp.get_tpfield(ti.i32, N)
    rec_jump = tp.get_tpfield(ti.i32, N)
    z_prime = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    is_border = tp.get_tpfield(ti.u1, N)
    outlet = tp.get_tpfield(ti.i64, N)
    basin_saddle = tp.get_tpfield(ti.i64, N)
    basin_saddlenode = tp.get_tpfield(ti.i32, N)
    tag = tp.get_tpfield(ti.u1, N)
    tag_alt = tp.get_tpfield(ti.u1, N)
    rerouted = tp.get_tpfield(ti.u1, N)

    # sfd accumulation temps
    donors = tp.get_tpfield(ti.i32, N * gridctx.n_neighbours)
    ndonors = tp.get_tpfield(ti.i32, N)
    donors_alt = tp.get_tpfield(ti.i32, N * gridctx.n_neighbours)
    ndonors_alt = tp.get_tpfield(ti.i32, N)
    Q_alt = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
    src = tp.get_tpfield(ti.i32, N)

    z_2d, lm_dh = preprocess_lm(dem.z.astype(np.float32), args.lm)
    if args.lm is not None:
        print(f"lm preprocess '{args.lm}': max|dh|={np.abs(lm_dh).max():.4f}")

    z.field.from_numpy(z_2d.ravel().astype(np.float32))
    h.field.fill(0.0)
    Q.field.fill(0.0)

    hs = visuctx.generate_hillshade(z.field, altitude_deg=45.0, azimuth_deg=315.0, z_factor=1.0)

    def propagate():
        Q.field.fill(0.0)
        floodctx.add_source_to_Q(Q.field)
        accum.set_weight(Q.field)

        floodctx.make_surface(z.field, h.field, surface.field)
        flowctx.compute_receivers(surface.field, receivers.field)

        rec_work.field.copy_from(receivers.field)
        rerouted.field.fill(False)
        ndep = flowctx.depression_counter(receivers.field)
        if ndep > 0:
            for _ in range(ceil(log2(max(1, int(ndep)))) + 1):
                ndep_bis = flowctx.depression_counter(rec_work.field)
                flowctx.basin_id_init(bid.field)
                rec_jump.field.copy_from(rec_work.field)
                for _ in range(logn + 1):
                    flowctx.propagate_basin_iter(rec_jump.field)
                flowctx.propagate_basin_final(bid.field, rec_jump.field)
                if ndep_bis == 0:
                    break
                flowctx.saddlesort(bid.field, is_border.field, z_prime.field, basin_saddle.field, basin_saddlenode.field, outlet.field, surface.field)
                flowctx.init_reroute_carve(tag.field, tag_alt.field, basin_saddlenode.field)
                receivers.field.copy_from(rec_work.field)
                rec_jump.field.copy_from(rec_work.field)
                for _ in range(logn + 1):
                    flowctx.iteration_reroute_carve(tag.field, tag_alt.field, receivers.field, rec_work.field, bid.field)
                flowctx.finalise_reroute_carve(receivers.field, rec_jump.field, tag.field, basin_saddlenode.field, outlet.field, rerouted.field)
                rec_work.field.copy_from(receivers.field)
            receivers.field.copy_from(rec_work.field)

        ndonors.field.fill(0)
        ndonors_alt.field.fill(0)
        src.field.fill(0)
        accum.init_weighted_source(Q.field)
        accum.receivers_to_donors(receivers.field, donors.field, ndonors.field)
        for iteration in range(logn + 1):
            accum.rake_compress_accum(donors.field, ndonors.field, Q.field, src.field, donors_alt.field, ndonors_alt.field, Q_alt.field, iteration)
        accum.fuse_accum_buffers(Q.field, src.field, Q_alt.field, logn)

    def distribute():
        floodctx.distribute_flow_local(z.field, h.field, Q.field, Q_next.field)
        Q.field.copy_from(Q_next.field)

    def core():
        floodctx.graphflood_core_unsafe(z.field, h.field, Q.field)

    def solve_lm():
        flowctx.solve_lm_zh(z.field, h.field)

    def distribute_1000():
        for i in range(5000):
            solve_lm()
        for _ in range(1000):
            distribute()

    def monitor_lm():
        return flowctx.monitor_lm_zh(z.field, h.field)

    # --- Qin reference: total source input over the whole domain ---
    Qin_total = precrate * NX * NY * DX ** 2

    # --- viz ---
    stop = {"flag": False}
    totime = [0.0]
    perc_hist = {p: [np.nan] for p in PERCENTILES}
    _dummy = np.zeros((NY, NX), dtype=np.float32)

    fig, ax = plt.subplots(1, 5, figsize=(24, 6))
    fig.subplots_adjust(bottom=0.27)
    for a in ax[:4]:
        a.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    imh = ax[0].imshow(_dummy, cmap="Blues", vmin=0, vmax=2.0, alpha=0.6)
    imratio = ax[1].imshow(_dummy, cmap="RdBu_r", vmin=0.95, vmax=1.05, alpha=0.6)
    imQi = ax[2].imshow(_dummy, cmap="viridis", alpha=0.6)
    imQo = ax[3].imshow(_dummy, cmap="viridis", alpha=0.6)
    ax[0].set_title("flow depth h")
    ax[1].set_title("Qout/Qin (local)")
    ax[2].set_title("log10 Qin")
    ax[3].set_title("log10 Qout")
    ax[4].set_title("dh/dt percentiles (log)")
    ax[4].set_yscale("log")
    lines = {}
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(PERCENTILES):
        (lines[p],) = ax[4].plot(totime, perc_hist[p], label=f"p{p}", color=cmap(i / (len(PERCENTILES) - 1)))
    ax[4].legend(loc="upper right", fontsize=8)
    ax[4].set_xlabel("model time [s]")

    dt_state = {"dt": dt_hydro}
    dt_text = fig.text(0.5, 0.94, f"dt = {dt_state['dt']:.3e} s", ha="center", fontsize=11)

    def set_dt(new_dt, publish=True):
        dt_state["dt"] = float(new_dt)
        floodctx.set_dth(dt_state["dt"])
        dt_text.set_text(f"dt = {dt_state['dt']:.3e} s")
        fig.canvas.draw_idle()
        if publish:
            write_param_file()

    ax_button = fig.add_axes([0.44, 0.03, 0.12, 0.06])
    btn = Button(ax_button, "Stop & save")

    def on_stop(event):
        stop["flag"] = True

    btn.on_clicked(on_stop)

    ax_x2 = fig.add_axes([0.08, 0.03, 0.06, 0.06])
    btn_x2 = Button(ax_x2, "dt x2")
    btn_x2.on_clicked(lambda event: set_dt(dt_state["dt"] * 2.0))

    ax_d2 = fig.add_axes([0.15, 0.03, 0.06, 0.06])
    btn_d2 = Button(ax_d2, "dt /2")
    btn_d2.on_clicked(lambda event: set_dt(dt_state["dt"] / 2.0))

    ax_p5 = fig.add_axes([0.22, 0.03, 0.07, 0.06])
    btn_p5 = Button(ax_p5, "+5%")
    btn_p5.on_clicked(lambda event: set_dt(dt_state["dt"] * 1.05))

    ax_m5 = fig.add_axes([0.30, 0.03, 0.07, 0.06])
    btn_m5 = Button(ax_m5, "-5%")
    btn_m5.on_clicked(lambda event: set_dt(dt_state["dt"] * 0.95))

    # --- main-loop params, live-editable from a separate process (param_panel.py) ---
    # matplotlib (TkAgg) can't share a GUI thread/process with a second panel, and the
    # compute loop can't move off the main thread either, so param_panel.py runs
    # standalone and writes this JSON; we just poll its mtime once per outer iteration.
    param_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_panel_state.json")
    loop_state = {
        "propagate_first": args.propagate_first,
        "n_local": args.n_local,
        "n_solve_lm": args.n_solve_lm,
        "n_distribute": args.n_distribute,
        "n_core": args.n_core,
    }
    param_file_mtime = [None]

    def write_param_file():
        payload = dict(loop_state)
        payload["dt"] = dt_state["dt"]
        tmp = param_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, param_file)
        param_file_mtime[0] = os.path.getmtime(param_file)

    def poll_param_file():
        try:
            mtime = os.path.getmtime(param_file)
        except FileNotFoundError:
            return
        if mtime == param_file_mtime[0]:
            return
        param_file_mtime[0] = mtime
        with open(param_file) as f:
            data = json.load(f)
        for key in ("propagate_first", "n_local", "n_solve_lm", "n_distribute", "n_core"):
            if key in data:
                loop_state[key] = data[key]
        if "dt" in data and float(data["dt"]) != dt_state["dt"]:
            set_dt(data["dt"], publish=False)

    write_param_file()  # publish CLI defaults so param_panel.py starts in sync

    panel_proc = None
    if not args.no_panel:
        panel_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_panel.py")
        panel_proc = subprocess.Popen([sys.executable, panel_script, "--file", param_file])
        print(f"launched param panel (pid={panel_proc.pid}), file: {param_file}")
    else:
        print(f"live params file: {param_file} (edit via `python param_panel.py --file {param_file}`)")

    fig.show()
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

    def save_ground_truth():
        out_dir = os.path.dirname(os.path.abspath(__file__))
        fname = f"{dem_name}_ground_truth_{args.precip}_{manning}.npy"
        out_path = os.path.join(out_dir, fname)
        payload = {
            "h": h.field.to_numpy().reshape(NY, NX),
            "Q": Q.field.to_numpy().reshape(NY, NX),
            "z": z.field.to_numpy().reshape(NY, NX),
        }
        np.save(out_path, payload, allow_pickle=True)
        print(f"\nsaved {out_path}")

    # --- initial propagation ---
    propagate()

    # --- loop ---
    ttt = 0.0
    while not stop["flag"]:
        poll_param_file()
        hm1 = h.field.to_numpy().reshape(NY, NX)

        if loop_state["propagate_first"]:
            propagate()

        n_local = loop_state["n_local"]
        for _ in range(n_local):
            for _ in range(loop_state["n_solve_lm"]):
                solve_lm()
            for _ in range(loop_state["n_distribute"]):
                distribute()
            for _ in range(loop_state["n_core"]):
                core()

        while(monitor_lm()>0):
            solve_lm()
            print(f'{monitor_lm()}', end = '                     \r')
        print()
        ti.sync()
        ttt += dt_state["dt"] * n_local

        floodctx.sum_Q_at_outlets(Q.field, out_sum.field)
        floodctx.compute_Qo(z.field, h.field, Qo.field)

        h_np = h.field.to_numpy().reshape(NY, NX)
        Q_np = Q.field.to_numpy().reshape(NY, NX)
        Qo_np = Qo.field.to_numpy().reshape(NY, NX)
        tdh = (hm1 - h_np) / dt_state["dt"]

        totime.append(ttt)
        for p in PERCENTILES:
            perc_hist[p].append(np.percentile(np.abs(tdh), p))

        ratio = np.divide(Qo_np, Q_np, out=np.ones_like(Q_np), where=Q_np > 1e-12)
        logQi = np.log10(np.maximum(Q_np, 1e-12))
        logQo = np.log10(np.maximum(Qo_np, 1e-12))

        imh.set_data(h_np)
        imratio.set_data(ratio)
        imQi.set_data(logQi)
        imQi.set_clim(logQi.min(), logQi.max())
        imQo.set_data(logQo)
        imQo.set_clim(logQo.min(), logQo.max())
        for p in PERCENTILES:
            lines[p].set_data(totime, perc_hist[p])
        ax[4].relim()
        ax[4].autoscale_view()

        balance = float(out_sum.field[None])
        print(
            f"t={ttt:.3f}s  dh p90={perc_hist[90][-1]:.3e}  Qbalance={balance:.4f} vs {Qin_total:.4f} NLM: {monitor_lm()}",
            end="          \r",
        )

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)

        if not plt.fignum_exists(fig.number):
            stop["flag"] = True

    save_ground_truth()

    if panel_proc is not None and panel_proc.poll() is None:
        panel_proc.terminate()

    # --- cleanup ---
    for f in [z, h, receivers, Q, Q_next, Qo, surface, out_sum, in_sum,
              bid, rec_work, rec_jump, z_prime, is_border, outlet,
              basin_saddle, basin_saddlenode, tag, tag_alt, rerouted,
              donors, ndonors, donors_alt, ndonors_alt, Q_alt, src]:
        f.release()
    floodctx.destroy()
    flowctx.destroy()
    gridctx.destroy()


if __name__ == "__main__":
    main()
