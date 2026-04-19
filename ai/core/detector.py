"""
Sign Boundary Detector — shared core implementation.

Segments continuous webcam stream into individual sign attempts.

Logic:
1. Track hand velocity frame-to-frame
2. When velocity drops below threshold for PAUSE_FRAMES consecutive frames -> sign boundary
3. Buffer the keypoint frames between boundaries
4. If buffer has >= MIN_SIGN_FRAMES, send to classifier
"""

import numpy as np
from collections import deque

from .constants import (
    FEATURE_DIM,
    MIN_SIGN_FRAMES,
    PAUSE_FRAMES,
    SEQ_LEN,
    VELOCITY_THRESHOLD,
)
from .preprocessing import compute_hand_velocity


class SignDetector:
    """
    Stateful sign boundary detector for real-time webcam stream.

    Accumulates keypoint frames and emits completed sign segments
    when a pause (hand stillness) is detected.
    """

    def __init__(self):
        self.frame_buffer = []
        self.prev_keypoints = None
        self.still_count = 0
        self.is_signing = False
        self.velocity_history = deque(maxlen=30)

    def feed_frame(self, keypoints):
        """
        Feed one frame of keypoints. Returns a completed sign segment or None.

        Args:
            keypoints: np.array of shape (FEATURE_DIM,)

        Returns:
            np.array of shape (SEQ_LEN, FEATURE_DIM) if a sign was completed, else None
        """
        kp = np.array(keypoints, dtype=np.float32)
        velocity = compute_hand_velocity(kp, self.prev_keypoints)
        self.velocity_history.append(velocity)
        self.prev_keypoints = kp.copy()

        is_still = velocity < VELOCITY_THRESHOLD

        if is_still:
            self.still_count += 1
        else:
            self.still_count = 0
            self.is_signing = True

        if self.is_signing:
            self.frame_buffer.append(kp)

        if self.still_count >= PAUSE_FRAMES and self.is_signing:
            return self._emit_sign()

        return None

    def _emit_sign(self):
        """Package the buffered frames as a completed sign and reset."""
        self.is_signing = False

        buffer = (
            self.frame_buffer[:-PAUSE_FRAMES]
            if len(self.frame_buffer) > PAUSE_FRAMES
            else self.frame_buffer
        )
        self.frame_buffer = []

        if len(buffer) < MIN_SIGN_FRAMES:
            return None

        frames = np.array(buffer, dtype=np.float32)

        if len(frames) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(frames), FEATURE_DIM), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        elif len(frames) > SEQ_LEN:
            indices = np.linspace(0, len(frames) - 1, SEQ_LEN, dtype=int)
            frames = frames[indices]

        return frames

    def force_emit(self):
        """Force emit whatever is in the buffer (for end-of-session)."""
        if len(self.frame_buffer) >= MIN_SIGN_FRAMES:
            self.is_signing = True
            self.still_count = PAUSE_FRAMES
            return self._emit_sign()
        self.frame_buffer = []
        return None

    def reset(self):
        """Clear all state."""
        self.frame_buffer = []
        self.prev_keypoints = None
        self.still_count = 0
        self.is_signing = False
        self.velocity_history.clear()

    @property
    def avg_velocity(self):
        """Current average hand velocity (for UI display)."""
        if not self.velocity_history:
            return 0.0
        return float(np.mean(self.velocity_history))

    @property
    def buffer_length(self):
        """Current number of buffered frames."""
        return len(self.frame_buffer)
