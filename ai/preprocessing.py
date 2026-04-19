"""
Preprocessing — Keypoint normalization and data augmentation.

Normalization makes keypoints invariant to camera position, body size, and distance.
Augmentation increases effective dataset size for better generalization.

Shared normalization functions are re-exported from ai.core.preprocessing.
augment_sequence() is training-only and not shared to Django.
"""

import numpy as np
import config

# Re-export shared preprocessing functions from ai.core
from ai.core.preprocessing import (  # noqa: F401
    normalize_keypoints,
    normalize_sequence,
    add_velocity_features,
    compute_hand_velocity,
)


# ── Data Augmentation ─────────────────────────────────────────────────

def augment_sequence(sequence, p=0.5):
    """
    Apply random augmentations to a keypoint sequence.
    Each augmentation is applied independently with probability p.

    Args:
        sequence: (SEQ_LEN, FEATURE_DIM) array (already normalized)
        p: probability of applying each augmentation

    Returns:
        Augmented (SEQ_LEN, FEATURE_DIM) array
    """
    seq = sequence.copy()

    # 1. Mirror (left-right flip) — swap left/right hand + flip x coordinates
    if np.random.random() < p:
        seq = _mirror_sequence(seq)

    # 2. Gaussian noise injection
    if np.random.random() < p:
        noise_scale = np.random.uniform(0.01, 0.05)
        noise = np.random.randn(*seq.shape).astype(np.float32) * noise_scale
        seq = seq + noise

    # 3. Temporal stretching (speed variation)
    if np.random.random() < p:
        seq = _temporal_stretch(seq)

    # 4. Random scaling (body size variation)
    if np.random.random() < p:
        scale = np.random.uniform(0.85, 1.15)
        seq = seq * scale

    # 5. Random rotation (small angle)
    if np.random.random() < p:
        angle = np.random.uniform(-15, 15)  # degrees
        seq = _rotate_sequence(seq, angle)

    return seq


def _mirror_sequence(sequence):
    """Flip left/right by negating x coordinates and swapping hands."""
    seq = sequence.copy().reshape(sequence.shape[0], -1, 2)  # (T, 54, 2)

    # Negate x coordinates
    seq[:, :, 0] = -seq[:, :, 0]

    # Swap left hand (indices 12-32) and right hand (indices 33-53)
    pose_end = config.NUM_POSE_LANDMARKS
    lh_end = pose_end + config.NUM_HAND_LANDMARKS
    rh_end = lh_end + config.NUM_HAND_LANDMARKS

    left_hand = seq[:, pose_end:lh_end, :].copy()
    right_hand = seq[:, lh_end:rh_end, :].copy()
    seq[:, pose_end:lh_end, :] = right_hand
    seq[:, lh_end:rh_end, :] = left_hand

    return seq.reshape(sequence.shape)


def _temporal_stretch(sequence):
    """Randomly speed up or slow down the sequence."""
    seq_len = len(sequence)
    factor = np.random.uniform(0.7, 1.3)
    new_len = max(1, int(seq_len * factor))
    indices = np.linspace(0, seq_len - 1, new_len).astype(int)
    stretched = sequence[indices]

    # Pad or truncate back to original length
    if len(stretched) < seq_len:
        pad = np.zeros((seq_len - len(stretched), sequence.shape[1]), dtype=np.float32)
        stretched = np.concatenate([stretched, pad], axis=0)
    elif len(stretched) > seq_len:
        idx = np.linspace(0, len(stretched) - 1, seq_len).astype(int)
        stretched = stretched[idx]

    return stretched


def _rotate_sequence(sequence, angle_deg):
    """Apply 2D rotation to all keypoints."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    seq = sequence.copy().reshape(sequence.shape[0], -1, 2)
    x = seq[:, :, 0]
    y = seq[:, :, 1]
    seq[:, :, 0] = x * cos_a - y * sin_a
    seq[:, :, 1] = x * sin_a + y * cos_a

    return seq.reshape(sequence.shape)
