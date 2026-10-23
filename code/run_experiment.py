"""Main entry point: one command reproduces numbers, tables and figures.

    python run_experiment.py                       # all windows from configs/experiment.json
    python run_experiment.py --window tuning       # one window
    python run_experiment.py --no-figures          # numbers only

For every window the vision-only estimator (`foe_pose`) and the gyroscope/accelerometer
filter (`VelocityFuser`) are run on the same frames. Outputs go to `results/`:
    dump_<window>.csv     per-frame channels (t, sx, sy: vision; fx, fy: fused; rx, ry: raw, ...)
    metrics_<window>.json delay, jitter and error to the reference for every channel
Figures are drawn by make_main_figure.py from these files only; the experiment is not rerun.

The video and the telemetry are not part of the repository: set their paths in the config.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from motion_direction import foe_pose, fusion                      # noqa: E402
from motion_direction.attitude_log import AttitudeLogReader        # noqa: E402
from motion_direction.metrics import jitter, lag_vs_reference, zero_phase  # noqa: E402

CHANNELS = {"vision": ("sx", "sy"), "fusion": ("fx", "fy"), "raw": ("rx", "ry")}


class VideoClock:
    """time-like object: perf_counter() follows the video time, not the wall clock."""

    def __init__(self, real):
        self._real = real
        self.t = 0.0

    def perf_counter(self):
        return self.t

    def __getattr__(self, name):
        return getattr(self._real, name)


class PoseSpy:
    """Proxy over cv2 for foe_pose: passes everything through but remembers recoverPose.

    The raw (unsmoothed) direction lives in a local variable inside foe_pose; it is the
    honest, delay-free reference, so it is captured here instead of editing foe_pose."""

    def __init__(self, real):
        self._cv2 = real
        self.dir = None
        self.inliers = 0

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def recoverPose(self, *args, **kwargs):
        result = self._cv2.recoverPose(*args, **kwargs)
        _, _, t, mask_pose = result
        d = (-t).flatten()
        norm = float(np.linalg.norm(d))
        self.dir = d / norm if norm > 1e-12 else None
        self.inliers = int(mask_pose.ravel().astype(bool).sum())
        return result


def resolve(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def effective_pose_config(path, out_dir):
    """Copy of the estimator config with an absolute intrinsics path."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["intrinsics_path"] = resolve(cfg["intrinsics_path"], os.path.dirname(os.path.abspath(path)))
    eff = os.path.join(out_dir, ".effective_config.json")
    with open(eff, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return eff


def run_window(cfg, window, pose_cfg_path):
    """Run the estimator and the filter on one window of the video; return per-frame records."""
    fcfg = cfg["fusion"]
    cap = cv2.VideoCapture(resolve(cfg["video"], HERE))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {cfg['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    clock = VideoClock(time)
    foe_pose.time = clock
    spy = PoseSpy(cv2)
    foe_pose.cv2 = spy
    telemetry = resolve(cfg["telemetry"], HERE)
    reader = AttitudeLogReader(telemetry)
    foe_pose.AttitudeShmReader = lambda name=None, **_kw: reader

    est = foe_pose.FoePoseEstimator(pose_cfg_path)
    mount = est.config["camera_mount_pitch_deg"]
    src, _, val = fcfg["speed_source"].partition(":")
    fuser = fusion.VelocityFuser(
        fusion.FlightLog(telemetry), mount, k_vis=fcfg["k_vis"], k_spd=fcfg["k_spd"],
        lag_s=fcfg["vision_lag"], speed_source=src, const_speed=float(val) if val else 8.0,
        gate_deg=fcfg["gate"], ref_window=est.config["ref_window"],
        inlier_ref_min=est.config["inlier_ref_min"], use_accel=fcfg["use_accel"],
        meas_median=fcfg["meas_median"], corr_max_deg=fcfg["corr_max"])

    cap.set(cv2.CAP_PROP_POS_MSEC, window["start"] * 1000.0)
    records, started = [], time.time()
    while True:
        t_v = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok or t_v > window["end"]:
            break
        clock.t = t_v
        prev_fix = est.last_slerp_time
        spy.dir = None
        sx, sy = est.process(frame, t_v)
        vision_ok = est.last_slerp_time != prev_fix        # a new measurement on this frame

        rec = {"t": t_v, "sx": sx, "sy": sy}
        raw_dir = spy.dir if vision_ok else None
        if raw_dir is not None:
            rec["rx"], rec["ry"] = fusion.cam_angles(raw_dir)
        inliers = spy.inliers if raw_dir is not None else None
        v_fused, v_imu = fuser.step(t_v, raw_dir, inliers)
        rec["fx"], rec["fy"] = fuser.angles(v_fused)
        rec["ix"], rec["iy"] = fuser.angles(v_imu)
        rec["inliers"] = inliers if inliers is not None else np.nan
        records.append(rec)
        if len(records) % 500 == 0:
            print(f"  {window['name']}: {len(records)} frames, t={t_v:.1f}s, "
                  f"{len(records) / (time.time() - started):.1f} fps", flush=True)
    cap.release()
    stats = {"frames": len(records), "fps_video": fps, "vision_updates": fuser.updates,
             "frames_without_correction": fuser.skips, "gate_rejected": fuser.rejected,
             "wall_seconds": round(time.time() - started, 1)}
    return pd.DataFrame(records), stats


def window_metrics(dump, fps, stats):
    """Delay (ms), RMS to the reference (deg) and jitter (deg/s) per channel and axis."""
    frame = dump
    # drop the warm-up: before the first non-zero output the estimator has produced nothing
    nonzero = np.flatnonzero((frame["sx"].abs() + frame["sy"].abs()).to_numpy() > 0)
    warm = int(nonzero[0]) if len(nonzero) else len(frame)
    result = {"warmup_frames": warm, **stats, "channels": {}}
    for axis in "xy":
        reference = zero_phase(frame["r" + axis].to_numpy(float))
        for name, cols in CHANNELS.items():
            channel = frame[cols[0] if axis == "x" else cols[1]].to_numpy(float).copy()
            channel[:warm] = np.nan
            shift, rms = lag_vs_reference(channel, reference)
            result["channels"].setdefault(name, {})[axis] = {
                "lag_ms": None if shift is None else round(1000.0 * shift / fps, 1),
                "rms_deg": rms,
                "jitter_deg_s": jitter(channel, fps),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "configs", "experiment.json"))
    ap.add_argument("--window", help="run only the window with this name")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    results_dir = resolve(cfg["results_dir"], HERE)
    os.makedirs(results_dir, exist_ok=True)
    pose_cfg = effective_pose_config(resolve(cfg["foe_pose_config"], HERE), results_dir)

    for window in cfg["windows"]:
        if args.window and window["name"] != args.window:
            continue
        print(f"window '{window['name']}': {window['start']}-{window['end']} s", flush=True)
        dump, stats = run_window(cfg, window, pose_cfg)
        dump_path = os.path.join(results_dir, f"dump_{window['name']}.csv")
        dump.to_csv(dump_path, index=False)
        metrics = window_metrics(dump, stats["fps_video"], stats)
        metrics["window"] = window
        metrics["fusion_params"] = cfg["fusion"]
        with open(os.path.join(results_dir, f"metrics_{window['name']}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        for name, axes in metrics["channels"].items():
            print(f"  {name:7s} " + "   ".join(
                f"{ax}: lag {axes[ax]['lag_ms']} ms, jitter {axes[ax]['jitter_deg_s']:.1f} deg/s"
                for ax in "xy"))
        if not args.no_figures:
            import make_main_figure
            out = os.path.join(resolve(cfg["figures_dir"], HERE), f"main_figure_{window['name']}.pdf")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            make_main_figure.draw(dump_path, out, axis="y")
            print(f"  figure: {out}")


if __name__ == "__main__":
    main()
