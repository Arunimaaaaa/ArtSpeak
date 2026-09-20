"""
Video & Computer Vision Feature Extractor for ArtSpeak
======================================================
Processes session video recordings to extract:
1. Aligned Face Crops (RGB 224x224) for Model 1 (RAF-DB Facial Emotion & VAD).
2. Gaze Coordinates (x, y, t) & Kinematic Gaze Metrics for Model 2 (Gaze Attention).
"""

import math
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class VideoFeatureExtractor:
    def __init__(self, target_fps: float = 5.0):
        self.target_fps = target_fps
        # Load OpenCV Haar Cascade for face and eye tracking (fast, lightweight, cross-platform)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # Transforms for RAF-DB Model (ImageNet / RAF-DB normalized)
        self.face_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.584, 0.457, 0.413],
                std=[0.260, 0.237, 0.232]
            ),
        ])

    def process_video(self, video_path: str) -> dict:
        """
        Extract face crops and gaze coordinates from a video file.
        Returns:
            {
                "face_tensors": torch.Tensor [N, 3, 224, 224],
                "timestamps": list[float],
                "gaze_points": list[dict(x, y, t)],
                "gaze_features": dict,
                "n_frames_analyzed": int
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file at: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(video_fps / self.target_fps)))

        face_tensors = []
        timestamps = []
        gaze_points = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                t_sec = frame_idx / video_fps
                h, w, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 1. Detect Face
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) > 0:
                    # Select largest face (primary participant)
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]

                    # Add gentle margin
                    pad = int(0.1 * fw)
                    y1 = max(0, fy - pad)
                    y2 = min(h, fy + fh + pad)
                    x1 = max(0, fx - pad)
                    x2 = min(w, fx + fw + pad)

                    face_crop_bgr = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    tensor = self.face_transform(pil_img)
                    face_tensors.append(tensor)
                    timestamps.append(t_sec)

                    # 2. Track Eyes within Face ROI for Gaze Estimation
                    face_gray = gray[fy:fy + fh, fx:fx + fw]
                    eyes = self.eye_cascade.detectMultiScale(
                        face_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
                    )

                    if len(eyes) >= 1:
                        # Estimate gaze center from detected eye centers relative to face center
                        eye_centers_x = [(ex + ew / 2.0) / fw for (ex, ey, ew, eh) in eyes]
                        eye_centers_y = [(ey + eh / 2.0) / fh for (ex, ey, ew, eh) in eyes]
                        norm_gaze_x = float(np.mean(eye_centers_x))
                        norm_gaze_y = float(np.mean(eye_centers_y))
                    else:
                        norm_gaze_x = float((fx + fw / 2.0) / w)
                        norm_gaze_y = float((fy + fh / 2.0) / h)

                    gaze_points.append({
                        "x": norm_gaze_x,
                        "y": norm_gaze_y,
                        "t": t_sec,
                    })

            frame_idx += 1

        cap.release()

        # If no face detected (or empty video), create fallback baseline tensor
        if len(face_tensors) == 0:
            dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
            face_tensors.append(self.face_transform(dummy_img))
            timestamps.append(0.0)
            gaze_points.append({"x": 0.5, "y": 0.5, "t": 0.0})

        stacked_faces = torch.stack(face_tensors)

        # 3. Compute Kinematic Gaze Features for Model 2
        gaze_features = self.compute_kinematic_gaze_features(gaze_points)

        return {
            "face_tensors": stacked_faces,
            "timestamps": timestamps,
            "gaze_points": gaze_points,
            "gaze_features": gaze_features,
            "n_frames_analyzed": len(timestamps),
        }

    def compute_kinematic_gaze_features(self, gaze_points: list[dict]) -> dict:
        """
        Compute kinematic features required by Gaze Classifier (Model 2).
        Features:
          - n_fixations, mean_x_norm, mean_y_norm, spatial_spread, scanpath_length
          - total_dwell_ms_log, mean_fixation_ms_log, std_fixation_ms_log
          - saccade_length_per_fixation, dwell_per_fixation, fixation_variability_ratio
          - center_dist, exploration_density, spread_to_path_ratio, dwell_spread_product, fixation_entropy_proxy
        """
        if not gaze_points:
            gaze_points = [{"x": 0.5, "y": 0.5, "t": 0.0}]

        xs = np.array([p["x"] for p in gaze_points])
        ys = np.array([p["y"] for p in gaze_points])
        ts = np.array([p["t"] for p in gaze_points])

        # Cluster into fixations (simple dispersion/velocity threshold)
        n_points = len(xs)
        dx = np.diff(xs)
        dy = np.diff(ys)
        dt = np.diff(ts) + 1e-5
        velocities = np.sqrt(dx**2 + dy**2) / dt

        # Velocity threshold for fixation vs saccade
        is_fixation = (velocities < 0.8)
        n_fixations = max(3, int(np.sum(is_fixation)))

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))

        # Spatial spread (std dev of coordinates)
        spatial_spread = float(np.sqrt(np.var(xs) + np.var(ys)) + 1e-4)

        # Scanpath length (sum of Euclidean shifts)
        scanpath_length = float(np.sum(np.sqrt(dx**2 + dy**2)) if len(dx) > 0 else 0.05)

        # Dwell times in ms
        total_duration_ms = max(1000.0, float((ts[-1] - ts[0]) * 1000.0) if len(ts) > 1 else 1000.0)
        mean_fix_ms = total_duration_ms / (n_fixations + 1e-5)
        std_fix_ms = mean_fix_ms * 0.35

        total_dwell_ms_log = float(np.log1p(total_duration_ms))
        mean_fixation_ms_log = float(np.log1p(mean_fix_ms))
        std_fixation_ms_log = float(np.log1p(std_fix_ms))

        # Kinematic derived features
        saccade_length_per_fix = scanpath_length / (n_fixations + 1e-5)
        dwell_per_fix = total_dwell_ms_log / (n_fixations + 1e-5)
        fix_variability_ratio = std_fixation_ms_log / (mean_fixation_ms_log + 1e-5)
        center_dist = float(np.sqrt((mean_x - 0.5)**2 + (mean_y - 0.5)**2))
        exploration_density = n_fixations / (spatial_spread + 1e-5)
        spread_to_path_ratio = spatial_spread / (scanpath_length + 1e-5)
        dwell_spread_product = total_dwell_ms_log * spatial_spread
        fixation_entropy_proxy = (spatial_spread * std_fixation_ms_log) / (mean_fixation_ms_log + 1e-5)

        return {
            "n_fixations": n_fixations,
            "mean_x_norm": mean_x,
            "mean_y_norm": mean_y,
            "spatial_spread": spatial_spread,
            "scanpath_length": scanpath_length,
            "total_dwell_ms_log": total_dwell_ms_log,
            "mean_fixation_ms_log": mean_fixation_ms_log,
            "std_fixation_ms_log": std_fixation_ms_log,
            "saccade_length_per_fixation": saccade_length_per_fix,
            "dwell_per_fixation": dwell_per_fix,
            "fixation_variability_ratio": fix_variability_ratio,
            "center_dist": center_dist,
            "exploration_density": exploration_density,
            "spread_to_path_ratio": spread_to_path_ratio,
            "dwell_spread_product": dwell_spread_product,
            "fixation_entropy_proxy": fixation_entropy_proxy,
        }
