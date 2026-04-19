"""
Keypoint Extractor — extends ai.core.extractor with training/demo methods.

The shared KeypointExtractor class lives in ai.core.extractor.
This module adds extract_frame_and_draw() and extract_video() as methods.
"""

import cv2
import numpy as np

import config

from ai.core.extractor import KeypointExtractor  # noqa: F401
from ai.core.preprocessing import compute_hand_velocity  # noqa: F401

# Hand connections for drawing (21 landmarks)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Pose connections for upper body (using full 17 COCO body landmarks)
_POSE_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),                  # Torso
]


def _draw_points(frame, points, connections, line_color, point_color, thickness=2, radius=3):
    """Draw [[x,y], ...] normalized landmarks on a BGR frame."""
    if not points:
        return
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in points]

    for i, j in connections:
        if i < len(px) and j < len(px):
            cv2.line(frame, px[i], px[j], line_color, thickness)

    for pt in px:
        cv2.circle(frame, pt, radius, point_color, -1)


def _extract_frame_and_draw(self, frame_bgr, frame_draw=None, timestamp_ms=None):
    """
    Extract keypoints AND draw landmarks on frame.

    Args:
        frame_bgr: BGR frame (passed directly to RTMPose, no conversion)
        frame_draw: BGR frame to draw on (if None, draws on frame_bgr copy)

    Returns:
        (keypoints, annotated_frame)
    """
    kp, landmarks = self.extract_frame(frame_bgr, timestamp_ms=timestamp_ms)

    # Draw on a contiguous copy to avoid ONNX memory reference leak
    annotated = np.ascontiguousarray(frame_draw if frame_draw is not None else frame_bgr).copy()

    if landmarks.get("pose"):
        _draw_points(annotated, landmarks["pose"], _POSE_CONNECTIONS,
                     (0, 255, 0), (0, 0, 255))

    if landmarks.get("left_hand"):
        _draw_points(annotated, landmarks["left_hand"], _HAND_CONNECTIONS,
                     (0, 0, 200), (0, 255, 0))

    if landmarks.get("right_hand"):
        _draw_points(annotated, landmarks["right_hand"], _HAND_CONNECTIONS,
                     (200, 0, 0), (0, 255, 0))

    return kp, annotated


def _extract_video(self, video_path, seq_len=None):
    """Extract keypoints from all frames of a video file."""
    seq_len = seq_len or config.SEQ_LEN
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Pass BGR directly — no conversion needed
        timestamp_ms = int(frame_idx * 1000 / fps)
        kp, _ = self.extract_frame(frame, timestamp_ms=timestamp_ms)
        frames.append(kp)
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        return np.zeros((seq_len, config.FEATURE_DIM), dtype=np.float32)

    frames = np.array(frames)

    if len(frames) < seq_len:
        pad = np.zeros((seq_len - len(frames), config.FEATURE_DIM), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    elif len(frames) > seq_len:
        indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
        frames = frames[indices]

    return frames


# Patch methods onto the shared KeypointExtractor class
KeypointExtractor.extract_frame_and_draw = _extract_frame_and_draw
KeypointExtractor.extract_video = _extract_video
