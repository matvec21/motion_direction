import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motion_direction.metrics import jitter, lag_vs_reference, zero_phase  # noqa: E402


def _signal(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)) * 0.5


def test_known_delay_is_recovered():
    truth = _signal()
    for delay in (0, 3, 7, 12):
        channel = np.concatenate([np.full(delay, truth[0]), truth[:len(truth) - delay]])
        shift, rms = lag_vs_reference(channel, truth)
        assert shift == delay
        assert rms < 1e-9


def test_delay_survives_noise_in_reference():
    truth = _signal()
    rng = np.random.default_rng(1)
    reference = zero_phase(truth + rng.normal(scale=0.5, size=len(truth)), window=9)
    channel = np.concatenate([np.full(5, truth[0]), truth[:-5]])
    shift, _ = lag_vs_reference(channel, reference)
    assert shift == 5


def test_zero_phase_adds_no_delay():
    ramp = np.arange(100, dtype=float)
    smoothed = zero_phase(ramp, window=9)
    assert np.allclose(smoothed[10:-10], ramp[10:-10])


def test_jitter_of_ramp_and_constant():
    assert jitter(np.full(50, 3.0), fps=30) == 0.0
    assert abs(jitter(np.arange(50) * 2.0, fps=30) - 60.0) < 1e-9


def test_nan_are_ignored():
    values = np.array([np.nan, 0.0, 1.0, 2.0, np.nan, 3.0])
    assert np.isfinite(jitter(values, fps=10))
