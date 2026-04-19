"""
WebSocket Consumer for real-time sign language inference.

Server-side MediaPipe processing — exact same pipeline as ai/demo.py:
  1. Client sends raw JPEG frames from webcam
  2. Server flips frame (mirror, same as demo.py)
  3. Server runs MediaPipe HolisticLandmarker (Tasks API)
  4. Server feeds keypoints to SignDetector (velocity-based boundary detection)
  5. Server runs BiLSTM inference on completed signs
  6. Server sends back landmark positions + predictions + sentence

Protocol (JSON over WebSocket):

Client -> Server:
  {"type": "frame", "image": "<base64_jpeg>"}
  {"type": "start_recording"}
  {"type": "stop_recording"}
  {"type": "clear"}

Server -> Client:
  {"type": "model_info", ...}
  {"type": "frame_result", "landmarks": {...}, "status": {...}, "prediction": {...}}
  {"type": "prediction", "word": "...", "confidence": 0.95, "buffer": [...]}
  {"type": "sentence", "text": "...", "glosses": [...]}
  {"type": "recording_started" / "recording_stopped" / "cleared"}
  {"type": "error", "message": "..."}
"""

import base64
import json
import logging
import time

import cv2
import numpy as np
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .inference import (
    FEATURE_DIM,
    KeypointExtractor,
    SignDetector,
    assemble_sentence,
    get_model,
    predict_sign,
)

logger = logging.getLogger(__name__)


class SignLanguageConsumer(AsyncWebsocketConsumer):
    """Handles real-time sign language inference over WebSocket with server-side MediaPipe."""

    async def connect(self):
        self.extractor = None
        self.detector = SignDetector()
        self.word_buffer = []
        self.recording = False
        self.frame_count = 0
        self._start_time = time.monotonic()
        self._user = self.scope.get("user")
        self._recording_start_time = None
        self._prediction_log = []
        self._last_prediction_time = 0.0
        self._last_predicted_word = None

        await self.accept()

        # Initialize server-side MediaPipe extractor (VIDEO mode — same as ai/demo.py)
        try:
            self.extractor = await sync_to_async(KeypointExtractor)(static_mode=False)
            logger.info("KeypointExtractor initialized (VIDEO mode)")
        except Exception as e:
            logger.error("Failed to initialize KeypointExtractor: %s", e)
            await self.send_json({
                "type": "error",
                "message": f"Failed to initialize MediaPipe: {e}",
            })

        # Send model info on connect
        model, glosses, device = await sync_to_async(get_model)()
        if model is not None:
            await self.send_json({
                "type": "model_info",
                "num_classes": len(glosses),
                "glosses": glosses,
                "device": device,
            })
        else:
            await self.send_json({
                "type": "error",
                "message": "Model not loaded. Ensure ai/checkpoints/isl_lstm_best.pt exists.",
            })

    async def disconnect(self, close_code):
        if self.extractor:
            await sync_to_async(self.extractor.close)()
            self.extractor = None
        self.detector.reset()
        self.word_buffer.clear()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self.send_json({"type": "error", "message": "Invalid JSON"})
            return

        msg_type = data.get("type")

        if msg_type == "frame":
            await self._handle_frame(data.get("image", ""))
        elif msg_type == "start_recording":
            await self._handle_start()
        elif msg_type == "stop_recording":
            await self._handle_stop()
        elif msg_type == "clear":
            await self._handle_clear()

    def _process_frame_sync(self, base64_image):
        """Synchronous frame processing: decode, flip, extract keypoints, detect signs.

        Returns a response dict to be sent to the client, or None on failure.
        """
        if not self.extractor or not base64_image:
            return None

        # Decode base64 JPEG → numpy BGR
        img_bytes = base64.b64decode(base64_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None

        # Mirror — exact same as ai/demo.py: frame = cv2.flip(frame, 1)
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Extract keypoints + raw landmarks (RTMPose accepts BGR directly)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        keypoints, landmarks = self.extractor.extract_frame(frame_bgr, timestamp_ms=timestamp_ms)

        self.frame_count += 1

        # Flip landmark x-coordinates back for client display
        # (server processes flipped frame, but client shows CSS-mirrored raw video)
        display_landmarks = self._flip_landmarks_x(landmarks)

        # Build response
        response = {
            "type": "frame_result",
            "landmarks": display_landmarks,
        }

        # Run sign detection + inference only when recording
        if self.recording:
            completed_sign = self.detector.feed_frame(keypoints)

            response["status"] = {
                "velocity": round(self.detector.avg_velocity, 4),
                "buffer_length": self.detector.buffer_length,
                "signing": self.detector.is_signing,
            }

            # If a sign was completed, run inference
            if completed_sign is not None:
                word, confidence = predict_sign(completed_sign)
                # Time-based dedup: same word within 1s is likely noise,
                # but same word after a real pause is intentional repetition
                now = time.monotonic()
                is_rapid_repeat = (
                    word == self._last_predicted_word
                    and (now - self._last_prediction_time) < 1.0
                )
                if word is not None and not is_rapid_repeat:
                    self._last_predicted_word = word
                    self._last_prediction_time = now
                    order = len(self.word_buffer)
                    self.word_buffer.append(word)
                    self._prediction_log.append((word, confidence, order))
                    response["prediction"] = {
                        "word": word,
                        "confidence": round(confidence, 3),
                        "buffer": list(self.word_buffer),
                    }
                    logger.info("Predicted: %s (%.1f%%)", word, confidence * 100)

        return response

    async def _handle_frame(self, base64_image):
        """Decode JPEG frame, flip (mirror), run MediaPipe, detect signs, predict."""
        try:
            response = await sync_to_async(self._process_frame_sync)(base64_image)
            if response is not None:
                await self.send_json(response)
        except Exception as e:
            logger.error("Frame processing error: %s", e, exc_info=True)

    @staticmethod
    def _flip_landmarks_x(landmarks):
        """Flip landmark x-coordinates back (1.0 - x) for client display alignment."""
        flipped = {}
        for key in ("pose", "left_hand", "right_hand"):
            pts = landmarks.get(key)
            if pts:
                flipped[key] = [[1.0 - x, y] for x, y in pts]
            else:
                flipped[key] = None
        return flipped

    async def _handle_start(self):
        self.recording = True
        self.frame_count = 0
        self.detector.reset()
        self.word_buffer.clear()
        self._last_predicted_word = None
        self._last_prediction_time = 0.0
        self._prediction_log = []
        self._recording_start_time = time.monotonic()
        logger.info("Recording started")
        await self.send_json({"type": "recording_started"})

    async def _handle_stop(self):
        self.recording = False
        duration = (
            time.monotonic() - self._recording_start_time
            if self._recording_start_time is not None
            else None
        )
        logger.info("Recording stopped. Word buffer: %s", self.word_buffer)

        # Force emit any remaining sign in buffer
        remaining = self.detector.force_emit()
        if remaining is not None:
            word, confidence = await sync_to_async(predict_sign)(remaining)
            if word is not None:
                order = len(self.word_buffer)
                self.word_buffer.append(word)
                self._prediction_log.append((word, confidence, order))
                await self.send_json({
                    "type": "prediction",
                    "word": word,
                    "confidence": round(confidence, 3),
                    "buffer": list(self.word_buffer),
                })

        # Assemble sentence from word buffer
        sentence_text = ""
        if self.word_buffer:
            sentence_text = await sync_to_async(assemble_sentence)(self.word_buffer)
            logger.info("Assembled sentence: %s", sentence_text)
            await self.send_json({
                "type": "sentence",
                "text": sentence_text,
                "glosses": list(self.word_buffer),
            })
        else:
            logger.info("No words to assemble.")

        # Persist session to DB
        await self._save_session(sentence_text, duration)

        await self.send_json({"type": "recording_stopped"})

    async def _handle_clear(self):
        self.detector.reset()
        self.word_buffer.clear()
        self._prediction_log = []
        self.frame_count = 0
        await self.send_json({"type": "cleared"})

    async def _save_session(self, sentence_text, duration):
        """Persist the interpretation session and predictions."""
        if not self._user or not self._user.is_authenticated:
            return
        if not self.word_buffer and not sentence_text:
            return
        try:
            from .models import InterpretationSession, Prediction
            session = await sync_to_async(InterpretationSession.objects.create)(
                user=self._user,
                sentence=sentence_text,
                glosses=list(self.word_buffer),
                duration_seconds=round(duration, 2) if duration else None,
                word_count=len(self.word_buffer),
            )
            predictions = [
                Prediction(session=session, word=word, confidence=conf, order=order)
                for word, conf, order in self._prediction_log
            ]
            if predictions:
                await sync_to_async(Prediction.objects.bulk_create)(predictions)
            logger.info("Saved session %d with %d predictions", session.pk, len(predictions))
        except Exception as e:
            logger.error("Failed to save session: %s", e)

    async def send_json(self, data):
        await self.send(text_data=json.dumps(data))
