"""
Happy Club Demo — Real-time ISL Sign -> Text Pipeline.

Modes:
    python demo.py              # Live webcam mode
    python demo.py --test       # Test on sample videos
    python demo.py --benchmark  # Run accuracy benchmark on test set

Controls (webcam mode):
    - R = Start/Stop recording signs
    - During recording: sign in front of camera, pause between words
    - On stop: all recorded signs are assembled into a sentence
    - C = clear word buffer and sentence
    - Q = quit
"""

import argparse
import sys
import time

import numpy as np
import torch

import config
from model import SignLanguageLSTM, load_model
from preprocessing import normalize_sequence, add_velocity_features


def _preprocess_for_inference(keypoints_seq):
    """Apply the same normalization + velocity used during training."""
    keypoints_seq = normalize_sequence(keypoints_seq)
    if config.USE_VELOCITY:
        keypoints_seq = add_velocity_features(keypoints_seq)
    return keypoints_seq


def run_webcam(model, glosses, device):
    """Live webcam sign recognition with landmarks + record/stop flow."""
    import cv2
    from keypoint_extractor import KeypointExtractor
    from sign_detector import SignDetector
    from sentence_assembler import assemble_sentence

    extractor = KeypointExtractor(static_mode=False, min_detection_confidence=0.5)
    detector = SignDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("  Run with --test to test on sample data instead.")
        return

    # State
    recording = False
    word_buffer = []
    last_prediction = ""
    last_confidence = 0.0
    assembled_sentence = ""
    processing = False  # True while LLM is assembling

    print("\n" + "=" * 60)
    print("  Happy Club — ISL Sign -> Text (Live)")
    print("  R = Record/Stop | C = Clear | Q = Quit")
    print("  Landmarks shown on hands and pose.")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # Extract keypoints AND draw landmarks on frame (BGR directly, no conversion)
        keypoints, frame = extractor.extract_frame_and_draw(frame)

        # Feed to sign boundary detector (only classify during recording)
        if recording:
            completed_sign = detector.feed_frame(keypoints)

            if completed_sign is not None:
                processed = _preprocess_for_inference(completed_sign)
                x = torch.from_numpy(processed).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = probs.max(dim=1)
                    confidence = confidence.item()
                    pred_idx = pred_idx.item()

                if confidence >= config.CONFIDENCE_THRESHOLD and pred_idx < len(glosses):
                    word = glosses[pred_idx]
                    last_prediction = word
                    last_confidence = confidence
                    word_buffer.append(word)
                    print(f"  [REC] Recognized: {word} ({confidence:.1%})")

        # ── Draw UI overlay ──
        h, w = frame.shape[:2]

        # ── Top bar: recording status + detection info ──
        cv2.rectangle(frame, (0, 0), (w, 65), (30, 30, 30), -1)

        if recording:
            # Flashing red recording indicator
            rec_color = (0, 0, 255)
            cv2.circle(frame, (25, 20), 10, rec_color, -1)
            cv2.putText(frame, "REC", (42, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)
        else:
            cv2.putText(frame, "READY", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 2)

        vel_text = f"Hand vel: {detector.avg_velocity:.4f}"
        sign_text = "SIGNING" if detector.is_signing else "idle"
        sign_color = (0, 255, 0) if detector.is_signing else (100, 100, 100)
        cv2.putText(frame, vel_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(frame, sign_text, (w - 80, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sign_color, 1)
        cv2.putText(frame, f"Buf: {detector.buffer_length}", (w - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Last prediction (right side) ──
        if last_prediction:
            cv2.putText(frame, f"{last_prediction}", (w - 260, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
            cv2.putText(frame, f"{last_confidence:.0%}", (w - 260, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        # ── Bottom panel: words + sentence ──
        cv2.rectangle(frame, (0, h - 100), (w, h), (30, 30, 30), -1)

        # Recognized words
        if word_buffer:
            words_display = " | ".join(word_buffer[-10:])
            cv2.putText(frame, f"Signs: {words_display}", (10, h - 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            hint = "Press R to start recording signs..." if not recording else "Sign in front of camera..."
            cv2.putText(frame, hint, (10, h - 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # Assembled sentence
        if assembled_sentence:
            cv2.putText(frame, f"Sentence: {assembled_sentence}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)

        # Controls hint
        cv2.putText(frame, "R=Record/Stop | C=Clear | Q=Quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.imshow("Happy Club - ISL Sign Language", frame)

        # ── Handle keys ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"):
            if not recording:
                # Start recording
                recording = True
                word_buffer = []
                assembled_sentence = ""
                last_prediction = ""
                detector.reset()
                print("\n  >>> Recording started. Sign now!")
            else:
                # Stop recording -> emit any remaining sign, then assemble
                recording = False
                remaining = detector.force_emit()
                if remaining is not None:
                    processed = _preprocess_for_inference(remaining)
                    x = torch.from_numpy(processed).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        confidence, pred_idx = probs.max(dim=1)
                    if confidence.item() >= config.CONFIDENCE_THRESHOLD and pred_idx.item() < len(glosses):
                        word = glosses[pred_idx.item()]
                        word_buffer.append(word)
                        print(f"  [REC] Final sign: {word} ({confidence.item():.1%})")

                print(f"  >>> Recording stopped. Signs: {word_buffer}")
                if word_buffer:
                    print("  >>> Assembling sentence...")
                    assembled_sentence = assemble_sentence(word_buffer)
                    print(f"  >>> Sentence: {assembled_sentence}\n")
                else:
                    assembled_sentence = "(no signs detected)"
                    print("  >>> No signs were detected during recording.\n")
                detector.reset()

        elif key == ord("c"):
            word_buffer = []
            last_prediction = ""
            last_confidence = 0.0
            assembled_sentence = ""
            detector.reset()
            if recording:
                recording = False
            print("  [Cleared]")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


def run_test(model, glosses, device):
    """Test pipeline on sample keypoint files from the dataset."""
    from pathlib import Path
    from sentence_assembler import assemble_sentence

    print("\n" + "=" * 60)
    print("  Happy Club — Pipeline Test (no webcam)")
    print("=" * 60)

    # Find test keypoint files
    test_dir = config.KEYPOINTS_DIR / "test"
    if not test_dir.exists():
        test_dir = config.KEYPOINTS_DIR / "val"
    if not test_dir.exists():
        test_dir = config.KEYPOINTS_DIR / "train"
    if not test_dir.exists():
        print("[ERROR] No keypoint data found. Run setup_dataset.py first.")
        return

    # Load a few samples from different classes
    correct = 0
    total = 0
    predictions = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        npy_files = sorted(class_dir.glob("*.npy"))[:3]  # 3 samples per class
        for npy_file in npy_files:
            keypoints = np.load(npy_file).astype(np.float32)

            # Ensure shape
            if keypoints.shape[0] < config.SEQ_LEN:
                pad = np.zeros((config.SEQ_LEN - keypoints.shape[0], config.FEATURE_DIM), dtype=np.float32)
                keypoints = np.concatenate([keypoints, pad], axis=0)
            elif keypoints.shape[0] > config.SEQ_LEN:
                indices = np.linspace(0, keypoints.shape[0] - 1, config.SEQ_LEN, dtype=int)
                keypoints = keypoints[indices]

            # Apply same preprocessing as training
            keypoints = _preprocess_for_inference(keypoints)

            x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = probs.max(dim=1)

            true_label = class_dir.name
            pred_label = glosses[pred_idx.item()] if pred_idx.item() < len(glosses) else "?"
            is_correct = pred_label == true_label

            correct += int(is_correct)
            total += 1
            predictions.append(pred_label)

            status = "✓" if is_correct else "✗"
            print(f"  {status} True: {true_label:20s} | Pred: {pred_label:20s} ({confidence.item():.1%})")

    if total > 0:
        print(f"\n  Accuracy: {correct}/{total} = {correct / total:.1%}")

    # Demo sentence assembly with a few predictions
    if predictions:
        sample_words = predictions[:5]
        print(f"\n  Sample word buffer: {sample_words}")
        sentence = assemble_sentence(sample_words)
        print(f"  Assembled sentence: {sentence}")


def run_benchmark(model, glosses, device):
    """Run full accuracy benchmark on test/val set."""
    from torch.utils.data import DataLoader
    from train import KeypointDataset

    print("\n  Running benchmark...")

    for split in ["test", "val"]:
        try:
            ds = KeypointDataset(config.KEYPOINTS_DIR, split=split, glosses=glosses)
            break
        except FileNotFoundError:
            continue
    else:
        print("[ERROR] No test or val data found.")
        return

    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"  {split} accuracy: {correct}/{total} = {correct / total:.1%}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Happy Club ISL Demo")
    parser.add_argument("--test", action="store_true", help="Test on saved keypoint files")
    parser.add_argument("--benchmark", action="store_true", help="Run accuracy benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print("\nLoading model...")
    model, glosses, device = load_model(config.MODEL_PATH, device)

    if not glosses:
        print("[WARN] No trained model found. Checking for untrained glosses...")
        # Try to infer glosses from data directory
        for split in ["train", "val", "test"]:
            split_dir = config.KEYPOINTS_DIR / split
            if split_dir.exists():
                glosses = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
                break

        if not glosses:
            print("[ERROR] No model and no data found.")
            print("  Step 1: python setup_dataset.py --sample  (create test data)")
            print("  Step 2: python train.py                    (train model)")
            print("  Step 3: python demo.py --test              (test pipeline)")
            return

        print(f"  Found {len(glosses)} classes from data directory")
        # Create untrained model with correct class count
        model = SignLanguageLSTM(num_classes=len(glosses)).to(device)
        model.eval()

    print(f"  Classes: {len(glosses)}")
    print(f"  Sample glosses: {glosses[:10]}")

    if args.benchmark:
        run_benchmark(model, glosses, device)
    elif args.test:
        run_test(model, glosses, device)
    else:
        run_webcam(model, glosses, device)


if __name__ == "__main__":
    main()
