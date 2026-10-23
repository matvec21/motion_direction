
import json
import math
import time
from collections import deque

import cv2
import numpy as np

try:
    from attitude_shm import AttitudeShmReader
except ImportError:
    AttitudeShmReader = None

cv2.setNumThreads(4)

class FoePoseEstimator:
    def __init__(self, config_path : str):
        with open(config_path, 'r', encoding = 'utf-8') as f:
            self.config = json.load(f)

        self.camera_mtx, self.camera_dist, self.camera_width, self.camera_height = load_intrinsics(self.config['intrinsics_path'])
        self.width = self.config['resize_width']
        self.height = int(self.config['resize_width'] * (self.camera_height / self.camera_width))
        self.lk_params = dict(winSize = (21, 21), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.camera_mtx = scale_camera_matrix(self.camera_mtx, self.camera_width, self.camera_height, self.width, self.height)

        self.smooth_dirvec = np.array([0, 0, 1], dtype = np.float64)
        self.tracks : list[deque[tuple[float, float]]] = []
        self.prev_gray = None
        self.last_slerp_time = None
        self.inlier_hist = deque(maxlen = self.config['ref_window'])

        self.speed_x = 0.0
        self.speed_y = 0.0

        self.attitude = None
        if self.config['sky_mask_enabled'] and AttitudeShmReader is not None:
            self.attitude = AttitudeShmReader(name = self.config['attitude_shm_name'])
        self.ground_mask = None
        self.horizon = None

    def detect_new(self, gray : np.ndarray):
        need = self.config['max_corners'] - len(self.tracks)
        if need <= 0:
            return

        mask = np.full(gray.shape, 255, dtype = np.uint8) if self.ground_mask is None else self.ground_mask.copy()
        for track in self.tracks:
            x, y = track[-1]
            cv2.circle(mask, (int(x), int(y)), self.config['min_distance'], 0, -1)

        corners = cv2.goodFeaturesToTrack(gray, maxCorners = need,
                                           qualityLevel = self.config['quality_level'],
                                           minDistance = self.config['min_distance'],
                                           mask = mask)
        if corners is None:
            return

        for corner in corners.reshape(-1, 2):
            self.tracks.append(deque([(float(corner[0]), float(corner[1]))], self.config['frame_gap'] + 1))

    def track(self, gray : np.ndarray):
        if not self.tracks or self.prev_gray is None:
            return

        pts_prev = np.array([track[-1] for track in self.tracks], dtype = np.float32).reshape(-1, 1, 2)
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts_prev, None, **self.lk_params)
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, pts_new, None, **self.lk_params)
        fb_errors = np.linalg.norm((pts_prev - pts_back).reshape(-1, 2), axis = 1)

        survivors = []
        for i, track in enumerate(self.tracks):
            if status[i, 0] != 1 or status_back[i, 0] != 1 or fb_errors[i] > self.config['fb_thresh']:
                continue
            x, y = pts_new[i, 0]
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            if self.ground_mask is not None and not self.ground_mask[int(y), int(x)]:
                continue
            track.append((float(x), float(y)))
            survivors.append(track)
        self.tracks = survivors

    def process(self, frame : np.ndarray, timestamp : float = None) -> tuple:
        self._process(frame, timestamp)
        return self.speed_x, self.speed_y

    def update_ground_mask(self, timestamp : float = None) -> None:
        self.ground_mask = None
        self.horizon = None
        if self.attitude is None:
            return

        now = time.time() if timestamp is None else timestamp
        att = self.attitude.at(now - self.config['camera_latency_s'])
        if att is None:
            return

        g = gravity_in_camera(att.roll, att.pitch, self.config['camera_mount_pitch_deg'])
        self.ground_mask, self.horizon = ground_mask(g, self.camera_mtx, self.width, self.height,
                                                     self.config['sky_margin_px'])

    def _process(self, frame : np.ndarray, timestamp : float = None) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.width, self.height), interpolation = cv2.INTER_AREA)

        self.update_ground_mask(timestamp)
        self.track(gray)
        self.prev_gray = gray

        if len(self.tracks) < self.config['min_corners']:
            self.detect_new(gray)

        full_tracks = [track for track in self.tracks if len(track) == self.config['frame_gap'] + 1]
        if len(full_tracks) < self.config['min_corr_points']:
            return

        pts_old = np.array([track[ 0] for track in full_tracks], dtype = np.float64)
        pts_new = np.array([track[-1] for track in full_tracks], dtype = np.float64)

        pts_old = cv2.undistortPoints(pts_old.reshape(-1, 1, 2), self.camera_mtx, self.camera_dist, P = self.camera_mtx).reshape(-1, 2)
        pts_new = cv2.undistortPoints(pts_new.reshape(-1, 1, 2), self.camera_mtx, self.camera_dist, P = self.camera_mtx).reshape(-1, 2)

        E, mask_e = cv2.findEssentialMat(pts_old, pts_new, self.camera_mtx,
                                         method = cv2.RANSAC, prob = 0.999, threshold = 1.0,
                                         maxIters = self.config['max_iters'])
        if E is None or E.shape != (3, 3):
            return

        _, _, t, mask_pose = cv2.recoverPose(E, pts_old, pts_new, self.camera_mtx, mask = mask_e)
        inlier_mask = mask_pose.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        if inliers < self.config['min_inliers']:
            return

        dirvec = (-t).flatten()
        dirvec = dirvec / (np.linalg.norm(dirvec) + 1e-12)

        now = time.perf_counter()
        if self.last_slerp_time is not None:
            dt = max(0.0, now - self.last_slerp_time)

            self.inlier_hist.append(inliers)
            ref = max(float(np.median(self.inlier_hist)), self.config['inlier_ref_min'])
            weight = self.config['blend'] * min(1.0, inliers / ref)
            blended = weight * dirvec + (1.0 - weight) * self.smooth_dirvec
            blended = blended / (np.linalg.norm(blended) + 1e-12)

            self.smooth_dirvec = slerp_towards(self.smooth_dirvec, blended, math.radians(self.config['step_max_deg'] * dt))
        else:
            self.smooth_dirvec = dirvec
        self.last_slerp_time = now

        x, y, z = self.smooth_dirvec
        self.speed_x = math.degrees(math.atan2(x, z))
        self.speed_y = math.degrees(math.atan2(-y, z))

def gravity_in_camera(roll_deg, pitch_deg, mount_pitch_deg):
    r, p, m = math.radians(roll_deg), math.radians(pitch_deg), math.radians(mount_pitch_deg)
    gx = -math.sin(p)
    gy = math.cos(p) * math.sin(r)
    gz = math.cos(p) * math.cos(r)
    cx, cy, cz = gy, gz, gx
    cm, sm = math.cos(m), math.sin(m)
    return np.array([cx, cm * cy - sm * cz, sm * cy + cm * cz], dtype = np.float64)

def ground_mask(g_cam, K, width, height, margin_px):
    line = np.linalg.inv(K).T @ g_cam
    scale = math.hypot(line[0], line[1])
    if scale < 1e-9:
        full = 255 if line[2] > 0 else 0
        return np.full((height, width), full, dtype = np.uint8), None

    thresh = margin_px * scale
    poly = clip_halfplane([(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], line, thresh)
    mask = np.zeros((height, width), dtype = np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)

    horizon = None
    if 0 < len(poly) < 6:
        on_line = [p for p in poly if abs(line[0] * p[0] + line[1] * p[1] + line[2] - thresh) < 1e-6 * max(1.0, scale)]
        if len(on_line) == 2:
            horizon = (on_line[0], on_line[1])
    return mask, horizon

def clip_halfplane(poly, line, thresh):
    out = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        fa = line[0] * a[0] + line[1] * a[1] + line[2] - thresh
        fb = line[0] * b[0] + line[1] * b[1] + line[2] - thresh
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            k = fa / (fa - fb)
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
    return out

def load_intrinsics(path):
    data = np.load(path)
    mtx = np.asarray(data['matrix'], dtype=np.float64)
    dist = np.asarray(data['distortion'], dtype=np.float64)
    width = int(data['width'])
    height = int(data['height'])
    return mtx, dist, width, height

def scale_camera_matrix(mtx, from_w, from_h, to_w, to_h):
    sx = to_w / from_w
    sy = to_h / from_h
    scaled = mtx.copy()
    scaled[0, 0] *= sx  # fx
    scaled[1, 1] *= sy  # fy
    scaled[0, 2] *= sx  # cx
    scaled[1, 2] *= sy  # cy
    return scaled

def slerp_towards(a, b, max_angle_rad):
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1e-9:
        return a
    t = min(1.0, max_angle_rad / angle)
    sin_angle = math.sin(angle)
    if sin_angle < 1e-9:
        return b
    s0 = math.sin((1 - t) * angle) / sin_angle
    s1 = math.sin(t * angle) / sin_angle
    v = s0 * a + s1 * b
    return v / (np.linalg.norm(v) + 1e-12)
