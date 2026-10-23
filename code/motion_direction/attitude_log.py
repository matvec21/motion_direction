# -*- coding: utf-8 -*-
"""Замена AttitudeShmReader: ориентация борта берётся не из shared memory, а из CSV-лога.

На борту foe_pose.py читает углы из кольцевого буфера в /dev/shm, который пишет
mavlink-процесс. При оффлайновом прогоне писать в shm некому, поэтому здесь тот же
самый интерфейс (`at(t)` / `latest()` / `attached()` / `close()`) отдаёт сэмплы из
телеметрийного CSV.

Время. Слоты в shm лежат в шкале time.time() борта, а лог синхронизирован с видео
колонкой `video_t_s` (0 = первый кадр). Поэтому ts сэмпла - это video_t_s, и
вызывающий код обязан передавать в FoePoseEstimator.process() время кадра по PTS.
Тогда `at(t - camera_latency_s)` внутри foe_pose работает ровно как на борту.

Углы - градусы, скорости - град/с, оси body FRD: как в AttitudeShmWriter.push().
"""
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd


class Attitude(NamedTuple):
    ts: float
    roll: float
    pitch: float
    yaw: float
    wx: float
    wy: float
    wz: float


# колонки лога -> поля слота shm
DEFAULT_COLUMNS = {
    "ts":    "video_t_s",
    "roll":  "mavlink_roll",
    "pitch": "mavlink_pitch",
    "yaw":   "mavlink_yaw",
    "wx":    "mavlink_roll_rate",
    "wy":    "mavlink_pitch_rate",
    "wz":    "mavlink_yaw_rate",
}


class AttitudeLogReader:
    """Тот же контракт, что у AttitudeShmReader, но поверх CSV.

    max_staleness_s держим таким же, как у shm-читателя: если ближайший сэмпл дальше
    этого от запрошенного момента, отдаём None - и foe_pose честно останется без
    маски на этом кадре, вместо того чтобы построить её по протухшим углам.
    """

    def __init__(self, csv_path: str, max_staleness_s: float = 0.5,
                 columns: dict = None, time_offset_s: float = 0.0):
        cols = dict(DEFAULT_COLUMNS)
        if columns:
            cols.update(columns)

        frame = pd.read_csv(csv_path, usecols=list(cols.values()))
        missing = [name for name in cols.values() if name not in frame.columns]
        if missing:
            raise SystemExit(f"в {csv_path} нет колонок: {', '.join(missing)}")

        data = frame[[cols[k] for k in ("ts", "roll", "pitch", "yaw", "wx", "wy", "wz")]]
        data = data.astype(float).to_numpy()
        data = data[np.isfinite(data).all(axis=1)]
        data = data[np.argsort(data[:, 0], kind="stable")]   # at() бинарным поиском

        self.path = csv_path
        self.max_staleness_s = max_staleness_s
        self._data = data
        self._ts = np.ascontiguousarray(data[:, 0]) + time_offset_s
        self._closed = False

    # -- публичное (совпадает с AttitudeShmReader) ---------------------------
    def at(self, t: float) -> Optional[Attitude]:
        """Ориентация на момент t - ближайший по времени сэмпл, как в кольце shm."""
        if self._closed or len(self._ts) == 0:
            return None
        i = int(np.searchsorted(self._ts, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self._ts):
                if best is None or abs(self._ts[j] - t) < abs(self._ts[best] - t):
                    best = j
        if best is None or abs(self._ts[best] - t) > self.max_staleness_s:
            return None
        return self._row(best)

    def latest(self) -> Optional[Attitude]:
        if self._closed or len(self._ts) == 0:
            return None
        return self._row(len(self._ts) - 1)

    def attached(self) -> bool:
        return not self._closed and len(self._ts) > 0

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -- служебное ----------------------------------------------------------
    def _row(self, i: int) -> Attitude:
        r = self._data[i]
        return Attitude(float(self._ts[i]), *(float(v) for v in r[1:]))

    def __len__(self) -> int:
        return len(self._ts)

    @property
    def duration(self) -> float:
        return float(self._ts[-1] - self._ts[0]) if len(self._ts) else 0.0

    @property
    def rate_hz(self) -> float:
        d = np.diff(self._ts)
        d = d[d > 0]
        return float(1.0 / np.median(d)) if len(d) else 0.0
