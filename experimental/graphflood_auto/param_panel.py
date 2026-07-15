"""
Standalone live parameter panel for generate_ground_truth.py.

Matplotlib (TkAgg) and a second GUI can't share a thread/process, and the
compute loop in generate_ground_truth.py can't be moved off the main thread
either. So this runs as an independent process: a small Tkinter window that
writes edited values to a JSON file, which generate_ground_truth.py polls
(mtime-gated) once per outer iteration.

Usage:
    python param_panel.py
    python param_panel.py --file /path/to/param_panel_state.json

Author: B.G.
"""

import argparse
import json
import os
import tkinter as tk

DEFAULTS = {
    "dt": 1e-3,
    "propagate_first": False,
    "n_local": 100,
    "n_solve_lm": 0,
    "n_distribute": 1,
    "n_core": 1,
}

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_panel_state.json")


def parse_args():
    p = argparse.ArgumentParser(description="Live parameter panel for generate_ground_truth.py")
    p.add_argument("--file", type=str, default=DEFAULT_PATH, help="Shared JSON state file")
    return p.parse_args()


def load_or_init(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    return dict(DEFAULTS)


def main():
    args = parse_args()
    state = load_or_init(args.file)

    root = tk.Tk()
    root.title("graphflood param panel")

    dt_var = tk.StringVar(value=str(state["dt"]))
    prop_var = tk.BooleanVar(value=state["propagate_first"])
    n_local_var = tk.StringVar(value=str(state["n_local"]))
    n_lm_var = tk.StringVar(value=str(state["n_solve_lm"]))
    n_dist_var = tk.StringVar(value=str(state["n_distribute"]))
    n_core_var = tk.StringVar(value=str(state["n_core"]))
    status_var = tk.StringVar(value="")

    def write(*_):
        try:
            payload = {
                "dt": float(dt_var.get()),
                "propagate_first": bool(prop_var.get()),
                "n_local": int(n_local_var.get()),
                "n_solve_lm": int(n_lm_var.get()),
                "n_distribute": int(n_dist_var.get()),
                "n_core": int(n_core_var.get()),
            }
        except ValueError:
            status_var.set("invalid value, not saved")
            return
        tmp = args.file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, args.file)
        status_var.set(f"saved -> {os.path.basename(args.file)}")

    def row(label, var, r, is_check=False):
        tk.Label(root, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        if is_check:
            w = tk.Checkbutton(root, variable=var, command=write)
        else:
            w = tk.Entry(root, textvariable=var, width=12)
            w.bind("<Return>", write)
            w.bind("<FocusOut>", write)
        w.grid(row=r, column=1, padx=6, pady=4)

    row("dt [s]", dt_var, 0)
    row("propagate first", prop_var, 1, is_check=True)
    row("n_local", n_local_var, 2)
    row("n_solve_lm", n_lm_var, 3)
    row("n_distribute", n_dist_var, 4)
    row("n_core", n_core_var, 5)

    tk.Button(root, text="Apply", command=write).grid(row=6, column=0, columnspan=2, pady=8)
    tk.Label(root, textvariable=status_var, fg="gray").grid(row=7, column=0, columnspan=2)

    write()  # make sure the file exists so the compute script sees consistent initial state

    root.mainloop()


if __name__ == "__main__":
    main()
