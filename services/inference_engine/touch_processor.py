"""
Touch & Rhythm Interaction Feature Extractor for ArtSpeak
=========================================================
Extracts 13 kinematic & rhythm features from session touch events for Model 3.
"""

import numpy as np
import pandas as pd


class TouchFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "avg_pressure",
            "avg_speed_px_s",
            "avg_curvature",
            "avg_pressure_variance",
            "avg_tap_interval_ms",
            "avg_tap_regularity",
            "stereotypy_burst_present",
            "tap_cadence_hz",
            "kinetic_touch_intensity",
            "rhythm_stability_index",
            "motor_fluidity_score",
            "dysregulation_ratio",
            "pressure_velocity_ratio",
        ]

    def extract_features(self, touch_events: list[dict]) -> dict:
        """
        Processes touch pointer events (e.g. from Supabase session_events or canvas).
        Each event: {"x": float, "y": float, "t": float (ms/sec), "pressure": float}
        """
        if not touch_events:
            # Baseline gentle interaction fallback
            return {
                "avg_pressure": 0.50,
                "avg_speed_px_s": 140.0,
                "avg_curvature": 0.55,
                "avg_pressure_variance": 0.08,
                "avg_tap_interval_ms": 380.0,
                "avg_tap_regularity": 0.68,
                "stereotypy_burst_present": 0.0,
                "tap_cadence_hz": 2.63,
                "kinetic_touch_intensity": 0.70,
                "rhythm_stability_index": 0.62,
                "motor_fluidity_score": 0.37,
                "dysregulation_ratio": 0.11,
                "pressure_velocity_ratio": 0.35,
            }

        pressures = [e.get("pressure", 0.5) for e in touch_events]
        xs = [e.get("x", 0.0) for e in touch_events]
        ys = [e.get("y", 0.0) for e in touch_events]
        ts = [e.get("t", 0.0) for e in touch_events]

        avg_pressure = float(np.mean(pressures))
        avg_pressure_var = float(np.var(pressures))

        # Speed calculation
        if len(xs) > 1:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(ts)
            # Ensure dt is in seconds
            if np.mean(dt) > 10.0:  # milliseconds
                dt = dt / 1000.0
            dt = np.clip(dt, 1e-4, 10.0)
            speeds = np.sqrt(dx**2 + dy**2) / dt
            avg_speed = float(np.mean(speeds))
        else:
            avg_speed = 120.0

        # Curvature calculation (smoothness of trajectory angles)
        if len(xs) > 2:
            v1_x, v1_y = np.diff(xs[:-1]), np.diff(ys[:-1])
            v2_x, v2_y = np.diff(xs[1:]), np.diff(ys[1:])
            dots = v1_x * v2_x + v1_y * v2_y
            norms = np.sqrt(v1_x**2 + v1_y**2) * np.sqrt(v2_x**2 + v2_y**2) + 1e-5
            cos_angles = np.clip(dots / norms, -1.0, 1.0)
            angles = np.arccos(cos_angles)
            avg_curvature = float(np.clip(1.0 - (np.mean(angles) / np.pi), 0.0, 1.0))
        else:
            avg_curvature = 0.60

        # Tap intervals & rhythm regularity
        if len(ts) > 1:
            intervals_ms = np.diff(ts)
            if np.mean(intervals_ms) < 5.0:  # already in seconds
                intervals_ms = intervals_ms * 1000.0
            avg_interval = float(np.mean(intervals_ms))
            interval_std = float(np.std(intervals_ms))
            avg_regularity = float(np.clip(1.0 - (interval_std / (avg_interval + 1e-4)), 0.0, 1.0))
            # Stereotypy burst (rapid tapping < 150ms)
            stereotypy = 1.0 if np.sum(intervals_ms < 150.0) >= 3 else 0.0
        else:
            avg_interval = 400.0
            avg_regularity = 0.70
            stereotypy = 0.0

        # Derived Kinematic Features
        tap_cadence_hz = 1000.0 / (avg_interval + 1e-5)
        kinetic_intensity = avg_pressure * (avg_speed / 100.0)
        rhythm_stability = avg_regularity * np.clip(1.0 - avg_pressure_var, 0.0, 1.0)
        motor_fluidity = avg_curvature * avg_regularity
        dysregulation = avg_pressure_var / (avg_regularity + 1e-4)
        pressure_vel_ratio = avg_pressure / (avg_speed / 100.0 + 1e-4)

        return {
            "avg_pressure": avg_pressure,
            "avg_speed_px_s": avg_speed,
            "avg_curvature": avg_curvature,
            "avg_pressure_variance": avg_pressure_var,
            "avg_tap_interval_ms": avg_interval,
            "avg_tap_regularity": avg_regularity,
            "stereotypy_burst_present": stereotypy,
            "tap_cadence_hz": tap_cadence_hz,
            "kinetic_touch_intensity": kinetic_intensity,
            "rhythm_stability_index": rhythm_stability,
            "motor_fluidity_score": motor_fluidity,
            "dysregulation_ratio": dysregulation,
            "pressure_velocity_ratio": pressure_vel_ratio,
        }
