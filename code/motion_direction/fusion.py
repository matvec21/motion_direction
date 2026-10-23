# -*- coding: utf-8 -*-
"""Фьюз инерциалки (аксель + гироскоп) с визуальным вектором скорости.

ЧТО С ЧЕМ И ЗАЧЕМ

Зрение (foe_pose) даёт ТОЛЬКО направление: единичный вектор в системе камеры, без
модуля скорости. И оно медленное: blend по инлайерам, ограничение поворота
step_max_deg град/с, плюс база frame_gap кадров - измерение относится к середине
окна, а не к текущему кадру. И проваливается на плохой геометрии: foe_pose тогда
просто отдаёт прошлое значение, кадр за кадром.

Инерциалка даёт производную без задержки на 47 Гц, но её интеграл уплывает за
секунды. Отсюда комплементарный фильтр: аксель с гироскопом ведут вектор между
измерениями, зрение медленно тянет его к себе.

ПОЧЕМУ В BODY, А НЕ В NED. Курс (mavlink_yaw) в этом логе не держится: подбор
постоянного сдвига yaw по окнам даёт от -64 до +134 градусов от окна к окну, то
есть направление скорости из EKF в горизонте доверия не заслуживает. В body-фрейме
курс не нужен вовсе:

    v̇_body = a_meas + g_body(roll, pitch) - ω × v_body

Здесь roll/pitch нужны только чтобы вычесть проекцию g, а они не дрейфуют
(акселерометр их подпирает). Гироскоп даёт ω. Из EKF берётся ровно один скаляр -
модуль скорости, он от курса не зависит.

ЗАЧЕМ ЧЛЕН ω × v. В кадре FOE ездит в основном не потому, что борт разгоняется, а
потому что борт ВРАЩАЕТСЯ: вектор скорости в мире стоит, а камера крутится.
Именно -ω × v это и описывает, и именно поэтому фьюз реагирует на манёвр сразу, а
зрение - с задержкой и упираясь в step_max_deg.

ПРО МОДУЛЬ СКОРОСТИ. Чтобы перевести ускорение (м/с^2) в скорость поворота вектора
(град/с), нужен модуль: поперечное ускорение a при скорости |v| крутит вектор на
a/|v| рад/с. Из акселерометра |v| не берётся (интеграл уплывает), из зрения тоже
(только направление). Поэтому модуль якорится к |v| из EKF - это честная часть
постановки: инерциалка отвечает за динамику направления, EKF только за масштаб.
Ключ --speed-source const:V позволяет обойтись и без EKF, фиксированным числом.

ПРО ЗАДЕРЖКУ ЗРЕНИЯ. Казалось бы, измерение с базой frame_gap должно относиться к
середине окна, то есть в прошлое на frame_gap/(2*fps) ~ 0.25 с. Померили: для
каждой гипотезы tau брали сырое измерение в момент t, поворачивали его гироскопом
на интервал [t-tau, t+dt-tau] и сравнивали с сырым измерением в t+dt. Минимум
ошибки вышел на tau = 0 (9.59 против 10.53 град без предсказания, на кадрах с
поворотом корпуса > 6 град), и дальше монотонно росла. То есть измерение
фактически относится к текущему кадру, и lag_s по умолчанию 0.

Механизм задержанного состояния при этом оставлен: состояние держится в двух
экземплярах - задержанное (в него бьёт коррекция) и текущее, перепрогоняемое от
задержанного вперёд по накопленной инерциалке. При lag_s = 0 это вырождается в
обычный фильтр, но ключ --vision-lag остаётся рабочим для другой камеры или
другого frame_gap.

ЧТО ИЗ ЭТОГО РЕАЛЬНО РАБОТАЕТ (замер на auc_21.08.26, окно 60-160 с, 3002 кадра):

  * гироскоп полезен: предсказание поворотом даёт 9.59 град против 10.53 без него
    (на кадрах с заметным вращением);
  * АКСЕЛЬ - в пределах шума. Интеграл ускорения за 0.3 с ухудшал предсказание
    (10.81 против 7.13 град), то есть смещение нуля съедает весь выигрыш. На
    рабочей точке фильтра разница между `с акселем` и `--no-accel` меньше
    полуградуса СКО, зато без акселя фильтр устойчив при вдвое более слабой
    коррекции. Аксель оставлен включённым по умолчанию, но честный ответ: на этих
    данных всю работу делает гироскоп;
  * зрение приходится держать сильным (k_vis ~ 16 1/с): при k_vis < 10 интеграл
    успевает уехать между коррекциями, ворота начинают резать измерения, и канал
    разваливается.
"""
import math
from collections import deque

import numpy as np
import pandas as pd

G = 9.80665

# body FRD (x вперёд, y вправо, z вниз) -> камера (x вправо, y вниз, z вперёд)
_BODY_TO_CAM = np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0]])


def r_nb(roll_deg, pitch_deg, yaw_deg):
    """Матрица body FRD -> NED, последовательность Z-Y-X (как в MAVLink ATTITUDE)."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def r_cb(mount_pitch_deg):
    """Матрица body -> камера с наклоном подвеса.

    Сверено с foe_pose.gravity_in_camera: r_cb(m) @ g_body совпадает с ней
    бит-в-бит на любых углах, то есть конвенция ровно та же."""
    m = math.radians(mount_pitch_deg)
    cm, sm = math.cos(m), math.sin(m)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cm, -sm],
                     [0.0, sm,  cm]]) @ _BODY_TO_CAM


def gravity_body(roll_deg, pitch_deg):
    """Вектор g в body FRD. Курс не участвует - в этом весь смысл body-фрейма."""
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return G * np.array([-math.sin(p), math.cos(p) * math.sin(r), math.cos(p) * math.cos(r)])


def cam_angles(v_cam):
    """Вектор в системе камеры -> (speed_x, speed_y), градусы, как в foe_pose."""
    x, y, z = v_cam
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(-y, z))


def cam_dir_from_angles(ax_deg, ay_deg):
    """Обратное к cam_angles для |углов| < 90."""
    v = np.array([math.tan(math.radians(ax_deg)), -math.tan(math.radians(ay_deg)), 1.0])
    return v / np.linalg.norm(v)


def angle_between(a, b):
    """Угол между направлениями, градусы."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))))


def flight_path_angle(v_body, roll_deg, pitch_deg):
    """Наклон траектории: 0 = горизонт, -90 = отвесно вниз.

    Считается через направление g в body, то есть НЕ ЗАВИСИТ ОТ КУРСА - этим и
    ценен: единственная величина, по которой зрение и EKF можно сравнивать, не
    упираясь в дрейф yaw."""
    return angle_between(v_body, gravity_body(roll_deg, pitch_deg)) - 90.0


def slerp(a, b, t):
    """Поворот a к b на долю t угла между ними; модуль a сохраняется."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return a
    a1, b1 = a / na, b / nb
    dot = float(np.clip(np.dot(a1, b1), -1.0, 1.0))
    ang = math.acos(dot)
    if ang < 1e-9:
        return a
    s = math.sin(ang)
    if s < 1e-9:
        return b1 * na
    v = (math.sin((1 - t) * ang) * a1 + math.sin(t * ang) * b1) / s
    return v / np.linalg.norm(v) * na


# ------------------------------------------------------------------------ лог
class FlightLog:
    """Аксель, гироскоп, ориентация и скорость EKF из CSV, по шкале video_t_s."""

    COLUMNS = {
        "t":    "video_t_s",
        "ax":   "mavlink_accel_x", "ay": "mavlink_accel_y", "az": "mavlink_accel_z",
        "wx":   "mavlink_roll_rate", "wy": "mavlink_pitch_rate", "wz": "mavlink_yaw_rate",
        "roll": "mavlink_roll", "pitch": "mavlink_pitch", "yaw": "mavlink_yaw",
        "vn":   "mavlink_north_speed", "ve": "mavlink_east_speed", "vd": "mavlink_down_speed",
    }

    def __init__(self, path, columns=None):
        cols = dict(FlightLog.COLUMNS)
        if columns:
            cols.update(columns)
        frame = pd.read_csv(path, usecols=lambda c: c in set(cols.values()))
        missing = [c for c in cols.values() if c not in frame.columns]
        if missing:
            raise SystemExit(f"в {path} нет колонок: {', '.join(missing)}")

        t = frame[cols["t"]].to_numpy(float)
        order = np.argsort(t, kind="stable")
        pick = lambda *keys: frame[[cols[k] for k in keys]].to_numpy(float)[order]
        self.t = t[order]
        self.accel = pick("ax", "ay", "az")
        self.gyro = np.radians(pick("wx", "wy", "wz"))     # град/с -> рад/с
        self.att = pick("roll", "pitch", "yaw")
        self.vned = pick("vn", "ve", "vd")
        # предпосчитанная g_body: gravity_body на каждый сэмпл - самая горячая точка
        r = np.radians(self.att[:, 0])
        p = np.radians(self.att[:, 1])
        self.g_body = G * np.stack([-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)], 1)

    def index(self, t):
        return int(np.searchsorted(self.t, t))

    def nearest(self, t):
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        return i

    def speed(self, t):
        """Модуль скорости из EKF (скаляр, от курса не зависит)."""
        return float(np.linalg.norm(self.vned[self.nearest(t)]))

    def attitude(self, t):
        return self.att[self.nearest(t)]


# --------------------------------------------------------------------- фильтр
class VelocityFuser:
    """Комплементарный фильтр по вектору скорости в body FRD.

    k_vis - темп подтягивания направления к зрению, 1/с;
    k_spd - темп подтягивания модуля к опорной скорости, 1/с;
    lag_s - на сколько визуальное измерение отстоит в прошлое.
    """

    def __init__(self, log: FlightLog, mount_pitch_deg: float,
                 k_vis: float = 1.5, k_spd: float = 0.5, lag_s: float = 0.25,
                 speed_source: str = "ekf", const_speed: float = 8.0,
                 gate_deg: float = 35.0, ref_window: int = 90,
                 inlier_ref_min: float = 60.0, use_accel: bool = True,
                 corr_max_deg: float = 0.0, meas_median: int = 1):
        self.log = log
        self.r_cb = r_cb(mount_pitch_deg)
        self.k_vis, self.k_spd, self.lag = k_vis, k_spd, lag_s
        self.speed_source, self.const_speed = speed_source, const_speed
        # Кормить фильтр СЫРЫМ измерением, а не сглаженным выходом foe_pose:
        # иначе фильтруем дважды и получаем сумму двух задержек. Плата - выбросы
        # (сырое иногда прыгает на десятки градусов), поэтому ворота по невязке
        # и вес по инлайерам, как в самом foe_pose.
        self.gate_deg = gate_deg
        self.inlier_hist = deque(maxlen=ref_window)
        self.inlier_ref_min = inlier_ref_min
        self.rejected = 0
        # Аксель можно выключить: тогда вектор скорости считается неподвижным в
        # мире и только вращается вместе с корпусом (член -ω × v). На этом логе
        # так выходит точнее - см. README, интеграл ускорения тащит смещение нуля.
        self.use_accel = use_accel
        # Ограничение скорости КОРРЕКЦИИ (град/с), 0 = без ограничения. Шум сырого
        # измерения - 470-730 град/с по производной, и коррекция тащит его часть в
        # выход. Гироскопный путь под ограничение НЕ попадает, поэтому настоящее
        # вращение проходит мгновенно, а дёргается только медленная подстройка.
        self.corr_max_deg = corr_max_deg
        # Медиана по последним N измерениям. Против выбросов она сильно лучше
        # усреднения: у сырого канала p95 производной 2300 град/с, это редкие
        # одиночные всплески, а не широкополосный шум. Медиана из 3 их убирает
        # ценой одного кадра задержки, среднее - размазывает.
        self.meas_median = max(1, int(meas_median))
        self.meas_hist = deque(maxlen=self.meas_median)

        self.v_del = None      # состояние на t - lag, в него бьёт зрение
        self.t_del = 0.0
        self.v_now = None      # выход: v_del, догнанный инерциалкой до кадра
        self.v_imu = None      # чистый интеграл, без коррекций - для сравнения
        self.t_imu = 0.0
        self.t_prev = None
        self.updates = 0
        self.skips = 0

    # -- интегрирование ----------------------------------------------------
    def _integrate(self, v, t_from, t_to):
        """v̇ = a + g_body - ω × v по сэмплам лога."""
        if t_to <= t_from:
            return v, t_from
        i = self.log.index(t_from)
        n = len(self.log.t)
        t_cur = t_from
        while i < n and self.log.t[i] <= t_to:
            dt = self.log.t[i] - t_cur
            if 0.0 < dt < 0.5:
                dv = -np.cross(self.log.gyro[i], v)
                if self.use_accel:
                    dv = dv + self.log.accel[i] + self.log.g_body[i]
                v = v + dv * dt
            t_cur = self.log.t[i]
            i += 1
        return v, t_cur

    def _ref_speed(self, t):
        if self.speed_source == "const":
            return self.const_speed
        return self.log.speed(t)

    def reset(self, t0, dir_body=None):
        """Старт: направление - из зрения, если оно уже есть, иначе из EKF."""
        speed = max(self._ref_speed(t0), 1.0)
        if dir_body is not None:
            v0 = np.asarray(dir_body, float) * speed
        else:
            v_ned = self.log.vned[self.log.nearest(t0)]
            att = self.log.attitude(t0)
            v0 = r_nb(*att).T @ v_ned
            if np.linalg.norm(v0) < 0.5:
                v0 = np.array([speed, 0.0, 0.0])
        self.v_del, self.t_del = v0.copy(), t0 - self.lag
        self.v_now = v0.copy()
        self.v_imu, self.t_imu = v0.copy(), t0
        self.t_prev = t0

    # -- шаг кадра ---------------------------------------------------------
    def step(self, t_frame, vision_dir_cam, inliers=None):
        """vision_dir_cam - единичный вектор измерения (система камеры) или None,
        если кадр не дал валидной геометрии. inliers - число инлайеров этого
        измерения, если известно: слабое измерение тянет слабее.

        Возвращает (v_fused_body, v_imu_only_body)."""
        dir_body = None if vision_dir_cam is None else self.r_cb.T @ np.asarray(vision_dir_cam, float)
        if dir_body is not None and self.meas_median > 1:
            self.meas_hist.append(dir_body)
            if len(self.meas_hist) == self.meas_median:
                med = np.median(np.stack(self.meas_hist), axis=0)
                n = np.linalg.norm(med)
                if n > 1e-9:
                    dir_body = med / n
        if self.v_del is None:
            self.reset(t_frame, dir_body)
            return self.v_now, self.v_imu

        dt = max(1e-3, min(0.5, t_frame - self.t_prev))
        self.t_prev = t_frame
        t_meas = t_frame - self.lag

        # 1. задержанное состояние - вперёд до момента, к которому относится зрение
        self.v_del, self.t_del = self._integrate(self.v_del, self.t_del, t_meas)

        # 2. коррекция направления по зрению - именно в задержанной точке
        if dir_body is not None and angle_between(self.v_del, dir_body) > self.gate_deg:
            self.rejected += 1          # выброс: инерциалка ведёт кадр сама
            dir_body = None
        if dir_body is not None:
            weight = 1.0
            if inliers is not None:
                self.inlier_hist.append(int(inliers))
                ref = max(float(np.median(self.inlier_hist)), self.inlier_ref_min)
                weight = min(1.0, inliers / ref)
            frac = 1.0 - math.exp(-self.k_vis * weight * dt)
            if self.corr_max_deg > 0.0:
                innov = angle_between(self.v_del, dir_body)
                if innov > 1e-6:
                    frac = min(frac, self.corr_max_deg * dt / innov)
            self.v_del = slerp(self.v_del, dir_body, frac)
            self.updates += 1
        else:
            self.skips += 1

        # 3. якорь модуля
        ref = self._ref_speed(t_meas)
        cur = float(np.linalg.norm(self.v_del))
        if cur > 1e-6 and ref > 0.2:
            w = 1.0 - math.exp(-self.k_spd * dt)
            self.v_del = self.v_del / cur * (cur + (ref - cur) * w)

        # 4. выход - догоняем инерциалкой от задержанной точки до текущего кадра
        self.v_now, _ = self._integrate(self.v_del.copy(), self.t_del, t_frame)

        # 5. отдельный канал: тот же интеграл, но зрение его никогда не поправляло
        self.v_imu, self.t_imu = self._integrate(self.v_imu, self.t_imu, t_frame)
        return self.v_now, self.v_imu

    # -- вывод -------------------------------------------------------------
    def angles(self, v_body):
        return cam_angles(self.r_cb @ np.asarray(v_body, float))


class Running:
    """Скользящее и полное СКО ряда."""

    def __init__(self, window=None):
        self.sq, self.n = 0.0, 0
        self.recent = deque(maxlen=window) if window else None

    def add(self, value):
        if value is None or not np.isfinite(value):
            return
        self.sq += value * value
        self.n += 1
        if self.recent is not None:
            self.recent.append(value)

    @property
    def rms(self):
        return math.sqrt(self.sq / self.n) if self.n else float("nan")

    @property
    def rms_recent(self):
        if not self.recent:
            return float("nan")
        a = np.asarray(self.recent, float)
        return float(np.sqrt(np.mean(a * a)))
