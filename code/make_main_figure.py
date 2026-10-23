"""Main figure of the project: delay of the vision-only direction estimate vs the fused one.

Reads the per-frame dump written by `foe_replay.py --fuse --show-raw --dump FILE.csv`
(columns: t, sy, fy, ry, ...) and draws
  top    : vertical angle over a short fragment: reference, vision-only, fused;
  bottom : RMS difference to the reference as a function of the time shift tau.
The minimum of each curve is the delay of the channel.

    python make_main_figure.py DUMP.csv -o ../slides/figs/main_figure.pdf
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FPS = 30.0
SKIP = 12          # frames before the tracker produces its first estimate
MAX_SHIFT = 45     # frames

C_REF, C_VIS, C_FUS = "#222222", "#d55e00", "#0072b2"


def zero_phase(values, window=9):
    """Symmetric moving average: suppresses noise, adds no delay (offline reference only)."""
    out = np.full(len(values), np.nan)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half):i + half + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


def rms_vs_shift(channel, reference, max_shift=MAX_SHIFT):
    """RMS(channel[t] - reference[t - tau]) for tau = 0..max_shift frames."""
    curve = []
    for shift in range(max_shift + 1):
        a = channel[shift:] if shift else channel
        b = reference[:len(reference) - shift] if shift else reference
        m = np.isfinite(a) & np.isfinite(b)
        curve.append(np.sqrt(np.mean((a[m] - b[m]) ** 2)) if m.sum() >= 30 else np.nan)
    return np.array(curve)


def busiest_fragment(reference, length):
    """Start index of the fragment where the reference changes the most."""
    best, best_start = -1.0, 0
    for start in range(SKIP, len(reference) - length):
        chunk = reference[start:start + length]
        if np.isfinite(chunk).all() and chunk.std() > best:
            best, best_start = chunk.std(), start
    return best_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="CSV from foe_replay.py --dump")
    ap.add_argument("-o", "--out", default="main_figure.pdf")
    ap.add_argument("--axis", default="y", choices=("x", "y"))
    ap.add_argument("--span", type=float, default=12.0, help="length of the time fragment, s")
    args = ap.parse_args()

    d = pd.read_csv(args.dump)
    ax_name = args.axis
    vis = d["s" + ax_name].to_numpy(float).copy()
    fus = d["f" + ax_name].to_numpy(float).copy()
    ref = zero_phase(d["r" + ax_name].to_numpy(float))
    vis[:SKIP] = fus[:SKIP] = np.nan
    t = d["t"].to_numpy(float)

    curves = {"vis": rms_vs_shift(vis, ref), "fus": rms_vs_shift(fus, ref)}
    lag_ms = {k: int(np.nanargmin(c)) * 1000.0 / FPS for k, c in curves.items()}

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.7), gridspec_kw={"height_ratios": [1.25, 1]})

    length = int(args.span * FPS)
    s0 = busiest_fragment(ref, length)
    sl = slice(s0, s0 + length)
    tt = t[sl] - t[s0]
    a1.plot(tt, vis[sl], color=C_VIS, lw=1.2, label=r"зрение $\hat d$ (foe_pose)")
    a1.plot(tt, fus[sl], color=C_FUS, lw=1.2, label=r"фьюз $\hat d$")
    a1.plot(tt, ref[sl], color=C_REF, lw=1.2, ls="--", label=r"опора $\bar z$")
    a1.set_xlabel("время, с")
    a1.set_ylabel("угол по вертикали, °")
    a1.legend(frameon=False, loc="best", fontsize=8)

    shifts_ms = np.arange(MAX_SHIFT + 1) * 1000.0 / FPS
    for key, color, name in (("vis", C_VIS, "зрение"), ("fus", C_FUS, "фьюз")):
        a2.plot(shifts_ms, curves[key], color=color, lw=1.4)
        i = int(np.nanargmin(curves[key]))
        a2.plot(shifts_ms[i], curves[key][i], "o", color=color, ms=5)
        a2.annotate(rf"$\tau={shifts_ms[i]:.0f}$ мс", (shifts_ms[i], curves[key][i]),
                    textcoords="offset points", xytext=(6, 8), color=color, fontsize=8.5)
    a2.set_xlabel(r"сдвиг $\tau$, мс")
    a2.set_ylabel(r"ошибка к $\bar z$, °")
    a2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"fragment {t[s0]:.1f}-{t[s0] + args.span:.1f} s, lag_ms = {lag_ms}")
    print("min rms, deg:", {k: float(np.nanmin(c)) for k, c in curves.items()})


if __name__ == "__main__":
    main()
