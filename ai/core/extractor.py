"""
Keypoint Extractor — shared core implementation.

Extracts upper body pose (12 points) + left hand (21 points) + right hand (21 points)
= 54 keypoints x 2 coordinates = 108 features per frame.

Uses RTMPose (via rtmlib) WholeBody model for 133-keypoint detection,
then selects the 54 keypoints matching our pipeline format.

Output is compatible with models trained on MediaPipe keypoints —
same anatomical landmarks, same normalized 0-1 coordinate space.
"""

import logging

import numpy as np

from .constants import (
    KEYPOINT_DIM,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
)

logger = logging.getLogger(__name__)

# ── COCO-WholeBody keypoint layout (133 total) ─────────────────────────
_COCO_UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_COCO_LEFT_HAND = slice(91, 112)
_COCO_RIGHT_HAND = slice(112, 133)

_MIN_SCORE = 0.3
_ZERO_KP = np.zeros(108, dtype=np.float32)
_EMPTY_LANDMARKS = {"pose": None, "left_hand": None, "right_hand": None}


class KeypointExtractor:
    """
    Extract pose + hand landmarks using RTMPose WholeBody (via rtmlib).

    Accepts BGR frames directly (avoids double conversion).
    Returns (keypoints, landmarks) with normalized 0-1 coordinates.
    """

    def __init__(self, static_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        from rtmlib import Wholebody

        self._wholebody = Wholebody(mode='lightweight')
        # Pre-allocate output buffer to avoid per-frame allocation
        self._kp_buf = np.zeros(108, dtype=np.float32)
        self._frame_count = 0
        logger.info("RTMPose WholeBody extractor initialized (lightweight mode)")

    def extract_frame(self, frame, timestamp_ms=None):
        """
        Extract keypoints from a single frame.

        Args:
            frame: BGR or RGB numpy array (H, W, 3). BGR preferred to avoid conversion.
            timestamp_ms: ignored (kept for API compat)

        Returns:
            (keypoints, landmarks) where keypoints is np.array(108,) and
            landmarks is dict with pose/left_hand/right_hand as [[x,y], ...].
        """
        h, w = frame.shape[:2]

        # rtmlib expects BGR — if frame looks like it might be RGB from Django,
        # the model still works (color doesn't affect skeleton detection)
        all_kp, all_scores = self._wholebody(frame)

        self._frame_count += 1

        if len(all_kp) == 0:
            return _ZERO_KP.copy(), dict(_EMPTY_LANDMARKS)

        # Use first detected person. Copy to break ONNX memory pool reference.
        kp = np.array(all_kp[0], dtype=np.float64)
        scores = np.array(all_scores[0], dtype=np.float64)

        buf = self._kp_buf
        buf[:] = 0.0
        landmarks = {"pose": None, "left_hand": None, "right_hand": None}

        # ── Upper body pose (12 from 17 body keypoints) ────────────
        pose_scores = scores[_COCO_UPPER_BODY]
        if np.mean(pose_scores) > _MIN_SCORE:
            landmarks["pose"] = [[kp[i][0] / w, kp[i][1] / h] for i in range(17)]
            for j, idx in enumerate(_COCO_UPPER_BODY):
                buf[j * 2] = kp[idx][0] / w
                buf[j * 2 + 1] = kp[idx][1] / h

        # ── Left hand (21 keypoints) ───────────────────────────────
        left_hand_kp = kp[_COCO_LEFT_HAND]
        left_hand_scores = scores[_COCO_LEFT_HAND]
        offset = NUM_POSE_LANDMARKS * KEYPOINT_DIM  # 24
        if np.mean(left_hand_scores) > _MIN_SCORE:
            landmarks["left_hand"] = [[p[0] / w, p[1] / h] for p in left_hand_kp]
            for j, p in enumerate(left_hand_kp):
                buf[offset + j * 2] = p[0] / w
                buf[offset + j * 2 + 1] = p[1] / h

        # ── Right hand (21 keypoints) ──────────────────────────────
        right_hand_kp = kp[_COCO_RIGHT_HAND]
        right_hand_scores = scores[_COCO_RIGHT_HAND]
        offset2 = offset + NUM_HAND_LANDMARKS * KEYPOINT_DIM  # 66
        if np.mean(right_hand_scores) > _MIN_SCORE:
            landmarks["right_hand"] = [[p[0] / w, p[1] / h] for p in right_hand_kp]
            for j, p in enumerate(right_hand_kp):
                buf[offset2 + j * 2] = p[0] / w
                buf[offset2 + j * 2 + 1] = p[1] / h

        # Return a COPY — breaks reference to internal buffer and ONNX arrays
        return buf.copy(), landmarks

    def close(self):
        """Release resources."""
        self._wholebody = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
