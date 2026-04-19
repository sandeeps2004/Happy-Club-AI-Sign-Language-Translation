"""
Shared preprocessing functions for the ISL inference pipeline.

Used by both ai/ (training/demo) and apps/sign_language/ (Django inference).
"""

import numpy as np
from .constants import (
    NUM_POSE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    SHOULDER_LEFT_IDX,
    SHOULDER_RIGHT_IDX,
)


def normalize_keypoints(keypoints_flat):
    """
    Normalize a single frame of keypoints relative to shoulder center.

    Applies SignSpace normalization:
    1. Center all points on the midpoint between shoulders
    2. Scale by shoulder distance (so all bodies are "same size")
    3. Min-max normalize hand keypoints within their bounding box

    Args:
        keypoints_flat: (FEATURE_DIM,) = (108,) array of [x,y] pairs

    Returns:
        (FEATURE_DIM,) normalized array
    """
    kp = keypoints_flat.copy().reshape(-1, 2)  # (54, 2)

    # Extract shoulder positions from pose landmarks
    left_shoulder = kp[SHOULDER_LEFT_IDX]    # (2,)
    right_shoulder = kp[SHOULDER_RIGHT_IDX]  # (2,)

    # Shoulder center and distance
    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

    # Avoid division by zero (no shoulders detected)
    if shoulder_dist < 1e-6:
        return keypoints_flat

    # Center all keypoints on shoulder midpoint, scale by shoulder distance
    kp = (kp - center) / shoulder_dist

    # Additional hand normalization: min-max within each hand's bounding box
    pose_end = NUM_POSE_LANDMARKS  # 12
    left_hand_end = pose_end + NUM_HAND_LANDMARKS  # 33
    right_hand_end = left_hand_end + NUM_HAND_LANDMARKS  # 54

    for start, end in [(pose_end, left_hand_end), (left_hand_end, right_hand_end)]:
        hand = kp[start:end]
        # Only normalize if hand was detected (not all zeros)
        hand_magnitude = np.abs(hand).sum()
        if hand_magnitude > 1e-4:
            h_min = hand.min(axis=0)
            h_max = hand.max(axis=0)
            h_range = h_max - h_min
            h_range = np.where(h_range < 1e-6, 1.0, h_range)
            kp[start:end] = (hand - h_min) / h_range - 0.5

    return kp.flatten()


def normalize_sequence(sequence):
    """
    Normalize a full sequence of keypoints.

    Args:
        sequence: (SEQ_LEN, FEATURE_DIM) array

    Returns:
        (SEQ_LEN, FEATURE_DIM) normalized array
    """
    result = np.zeros_like(sequence)
    for i in range(len(sequence)):
        result[i] = normalize_keypoints(sequence[i])
    return result


def add_velocity_features(sequence):
    """
    Append frame-to-frame velocity (delta) features to the sequence.

    Args:
        sequence: (SEQ_LEN, FEATURE_DIM) normalized keypoints

    Returns:
        (SEQ_LEN, FEATURE_DIM * 2) with [position, velocity] per frame
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, velocity], axis=1)


def compute_hand_velocity(keypoints_current, keypoints_previous):
    """
    Compute the Euclidean displacement of HAND landmarks only between frames.

    Only measures the 42 hand keypoints (left + right), ignoring pose landmarks
    whose micro-movements (breathing, body sway, tracking jitter) create a noise
    floor that prevents sign boundary detection.

    Args:
        keypoints_current: (FEATURE_DIM,) array for the current frame
        keypoints_previous: (FEATURE_DIM,) array for the previous frame, or None

    Returns:
        float: mean Euclidean displacement across hand keypoints, or 1.0 if no previous frame
    """
    if keypoints_previous is None:
        return 1.0

    current = keypoints_current.reshape(-1, 2)
    previous = keypoints_previous.reshape(-1, 2)

    # Only measure hand landmarks: skip first 12 pose landmarks
    # Indices 12-32 = left hand (21), 33-53 = right hand (21) → 42 hand keypoints
    hand_current = current[NUM_POSE_LANDMARKS:]
    hand_previous = previous[NUM_POSE_LANDMARKS:]
    displacement = np.linalg.norm(hand_current - hand_previous, axis=1)
    return float(displacement.mean())
