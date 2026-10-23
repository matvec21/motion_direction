"""Main figure of the project: delay of the vision-only direction estimate vs the fused one.

Reads the per-frame dump written by `run_experiment.py` (columns t, sx/sy: vision,
fx/fy: fused, rx/ry: raw measurement) and draws
  top    : angle over a short fragment: reference, vision-only, fused;
  bottom : RMS difference to the reference as a function of the time shift tau.
The minimum of each curve is the delay of the channel.

    python make_main_figure.py results/dump_tuning.csv -o ../slides/figs/main_figure.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_direction.metrics import rms_vs_shift, zero_phase  # noqa: E402

FPS = 30.0
MAX_SHIFT = 45     # frames
C_REF, C_VIS, C_FUS = "#222222", "#d55e00", "#0072b2"


def busiest_fragment(reference, length, start_from):
    """Start index of the fragment where the reference changes the most."""
    best, best_start = -1.0, start_from
    for start in range(start_from, len(reference) - length):
        chunk = reference[start:start + length]
        if np.isfinite(chunk).all() and chunk.std() > best:
            best, best_start = chunk.std(), start
    return best_start


def draw(dump_csv, out, axis="y", span=12.0):
    """Draw the figure from a dump file; return delays (ms) and minimal RMS (deg)."""
    d = pd.read_csv(dump_csv)
    vis = d["s" + axis].to_numpy(float).copy()
    fus = d["f" + axis].to_numpy(float).copy()
    ref = zero_phase(d["r" + axis].to_numpy(float))
    nonzero = np.flatnonzero((d["sx"].abs() + d["sy"].abs()).to_numpy() > 0)
    warm = int(nonzero[0]) if len(nonzero) else 0     # frames before the first estimate
    vis[:warm] = fus[:warm] = np.nan
    t = d["t"].to_numpy(float)

    curves = {"vis": rms_vs_shift(vis, ref, MAX_SHIFT), "fus": rms_vs_shift(fus, ref, MAX_SHIFT)}
    lag_ms = {k: int(np.nanargmin(c)) * 1000.0 / FPS for k, c in curves.items()}

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.7), gridspec_kw={"height_ratios": [1.25, 1]})

    length = int(span * FPS)
    s0 = busiest_fragment(ref, length, warm)
    sl = slice(s0, s0 + length)
    tt = t[sl] - t[s0]
    a1.plot(tt, vis[sl], color=C_VIS, lw=1.2, label=r"зрение $\hat d$ (foe_pose)")
    a1.plot(tt, fus[sl], color=C_FUS, lw=1.2, label=r"фьюз $\hat d$")
    a1.plot(tt, ref[sl], color=C_REF, lw=1.2, ls="--", label=r"опора $\bar z$")
    a1.set_xlabel("время, с")
    a1.set_ylabel("угол по вертикали, °" if axis == "y" else "угол по горизонтали, °")
    a1.legend(frameon=False, loc="best", fontsize=8)

    shifts_ms = np.arange(MAX_SHIFT + 1) * 1000.0 / FPS
    for key, color in (("vis", C_VIS), ("fus", C_FUS)):
        a2.plot(shifts_ms, curves[key], color=color, lw=1.4)
        i = int(np.nanargmin(curves[key]))
        a2.plot(shifts_ms[i], curves[key][i], "o", color=color, ms=5)
        a2.annotate(rf"$\tau={shifts_ms[i]:.0f}$ мс", (shifts_ms[i], curves[key][i]),
                    textcoords="offset points", xytext=(6, 8), color=color, fontsize=8.5)
    a2.set_xlabel(r"сдвиг $\tau$, мс")
    a2.set_ylabel(r"ошибка к $\bar z$, °")
    a2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {"lag_ms": lag_ms, "min_rms_deg": {k: float(np.nanmin(c)) for k, c in curves.items()},
            "fragment_s": (float(t[s0]), float(t[s0] + span))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="CSV written by run_experiment.py")
    ap.add_argument("-o", "--out", default="main_figure.pdf")
    ap.add_argument("--axis", default="y", choices=("x", "y"))
    ap.add_argument("--span", type=float, default=12.0, help="length of the time fragment, s")
    args = ap.parse_args()
    print(draw(args.dump, args.out, args.axis, args.span))


if __name__ == "__main__":
    main()
