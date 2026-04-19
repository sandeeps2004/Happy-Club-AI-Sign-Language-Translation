"""
LSTM Sign Language Classifier — shared core implementation.

Input:  (batch, seq_len, feature_dim) — keypoint sequences
Output: (batch, num_classes) — word class probabilities
"""

import threading

import torch
import torch.nn as nn

from .constants import (
    BIDIRECTIONAL,
    DROPOUT,
    FEATURE_DIM,
    FEATURE_DIM_FINAL,
    HIDDEN_SIZE,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_LAYERS,
)

# Thread-safe singleton state
_model = None
_glosses = []
_device = "cpu"
_lock = threading.Lock()


class SignLanguageLSTM(nn.Module):
    """
    Bidirectional LSTM for classifying sign language keypoint sequences.

    Architecture:
        LayerNorm -> BiLSTM (2 layers) -> Attention pooling -> FC -> Dropout -> FC
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.layer_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_size * self.num_directions

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def attention_pool(self, lstm_out, mask=None):
        """Weighted sum of LSTM outputs using learned attention."""
        attn_scores = self.attention(lstm_out).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, feature_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)

        mask = None
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        context = self.attention_pool(lstm_out, mask)
        logits = self.classifier(context)
        return logits


def load_model(model_path=None, device=None):
    """Load trained model (thread-safe singleton). Returns (model, glosses, device)."""
    global _model, _glosses, _device

    if _model is not None:
        return _model, _glosses, _device

    with _lock:
        if _model is not None:
            return _model, _glosses, _device

        path = model_path or MODEL_PATH
        if not path.exists():
            import logging
            logging.getLogger(__name__).warning("No checkpoint found at %s", path)
            return None, [], _device

        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=_device, weights_only=False)
        _glosses = checkpoint.get("glosses", [])
        ckpt_config = checkpoint.get("config", {})

        num_classes = ckpt_config.get("num_classes", len(_glosses)) or NUM_CLASSES
        input_dim = ckpt_config.get("input_dim", FEATURE_DIM_FINAL)

        _model = SignLanguageLSTM(input_dim=input_dim, num_classes=num_classes)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.train(False)

        import logging
        logging.getLogger(__name__).info(
            "Loaded sign language model: %d classes, device=%s", num_classes, _device
        )
        return _model, _glosses, _device
