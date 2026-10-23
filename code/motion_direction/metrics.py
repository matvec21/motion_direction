"""Quality metrics of a direction channel: delay and jitter.

The reference for the delay is the raw visual measurement smoothed by a symmetric
window (zero phase): the window looks into the future, so it cannot run on board,
but offline it adds no delay of its own. The delay of a channel is the time shift
that minimises its RMS difference to the reference.
"""
import numpy as np


def zero_phase(values, window=9):
    """Symmetric moving average over finite values; adds no delay."""
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half):i + half + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


def jitter(values, fps):
    """RMS of the time derivative of a channel, deg/s: what shakes the control loop."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float("nan")
    return float(np.sqrt(np.mean((np.diff(a) * fps) ** 2)))


def rms_vs_shift(channel, reference, max_shift=45, min_points=30):
    """RMS(channel[t] - reference[t - shift]) for shift = 0..max_shift frames."""
    channel = np.asarray(channel, float)
    reference = np.asarray(reference, float)
    curve = []
    for shift in range(max_shift + 1):
        a = channel[shift:] if shift else channel
        b = reference[:len(reference) - shift] if shift else reference
        m = np.isfinite(a) & np.isfinite(b)
        curve.append(float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() >= min_points else np.nan)
    return np.array(curve)


def lag_vs_reference(channel, reference, max_shift=45):
    """Return (shift in frames, RMS at that shift) with the smallest RMS."""
    curve = rms_vs_shift(channel, reference, max_shift)
    if np.isnan(curve).all():
        return None, float("nan")
    shift = int(np.nanargmin(curve))
    return shift, float(curve[shift])
