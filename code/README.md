# Code

Source code for the paper: the experiment (vision-only estimator and the filter run on the same frames), the metrics, and the figures. Computation and plotting are separate: the experiment writes numbers to `results/`, the figures are drawn from those files.

## Layout

```
configs/
  experiment.json      # paths to the video and telemetry, windows, filter parameters
  foe_pose.json        # parameters of the vision-only estimator (640 px, frame gap 15)
  foe_pose_gap5_1200px.json  # the later onboard configuration (1200 px, frame gap 5), not used for the results
  calibration_data_full.npz  # camera intrinsics
motion_direction/
  foe_pose.py          # vision-only estimator: tracks, essential matrix, smoothing
  fusion.py            # complementary filter on the unit sphere (gyroscope, accelerometer, vision)
  attitude_log.py      # attitude and angular rates from the telemetry CSV
  metrics.py           # delay (zero-phase reference) and jitter
run_experiment.py      # main entry point: numbers and figures with one command
make_main_figure.py    # main figure, reads results/dump_*.csv
tests/                 # unit tests of the metrics (synthetic delay of known size)
results/               # metrics_*.json (tracked), dump_*.csv (generated)
```

## Installation

Clone the repo:
```bash
git clone https://github.com/matvec21/motion_direction.git
cd motion_direction/code
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The video and the telemetry are not part of the repository. Set their paths in `configs/experiment.json` (`video`, `telemetry`; the telemetry CSV must contain the column `video_t_s`, the time of each row on the video clock).

Run both windows and draw the figures:
```bash
python run_experiment.py
```

Only one window, numbers only:
```bash
python run_experiment.py --window tuning --no-figures
```

Redraw the main figure from a saved dump without rerunning the experiment:
```bash
python make_main_figure.py results/dump_tuning.csv -o ../slides/figs/main_figure.pdf
```

Run the tests:
```bash
python -m pytest tests
```

## Notes

- The estimator and the filter use the video clock instead of the wall clock, so the result does not depend on the speed of the run.
- The first frames, before the estimator produces its first non-zero output, are excluded from the delay and jitter.
- The filter parameters were tuned on the window `tuning`; the window `validation` was not used for tuning.
