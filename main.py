"""Entry point: webcam -> Mediapipe Face Landmarker -> expression detection -> Tkinter popup.

Run with: python main.py
Press 'q' in the webcam preview window (or close the popup) to quit.
"""
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from detector import STATE_NEUTRAL, ExpressionDetector, compute_ratios, compute_tongue_redness
from gesture import GestureDetector, classify_gesture
from popup import ReactionPopup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# state -> reaction image/gif shown when that state is detected.
# "confused" and "strong"/"thumbsup" can each be reached two ways: a facial
# expression (detector.py) or a hand gesture (gesture.py) - see main loop.
IMAGE_MAP = {
    STATE_NEUTRAL: os.path.join(IMAGES_DIR, "neutral.gif"),
    "happy": os.path.join(IMAGES_DIR, "happy.gif"),
    "surprised": os.path.join(IMAGES_DIR, "scared.webp"),
    "confused": os.path.join(IMAGES_DIR, "nerd.gif"),
    "strong": os.path.join(IMAGES_DIR, "strong.gif"),
    "thumbsup": os.path.join(IMAGES_DIR, "thumbsup.gif"),
    "silly": os.path.join(IMAGES_DIR, "silly.gif"),
}

CALIBRATION_SECONDS = 3
FRAME_INTERVAL_MS = 15
FACE_SMOOTHING_FRAMES = 10
GESTURE_SMOOTHING_FRAMES = 6


def _download(path, url, label):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {label} model (one-time)...")
    urllib.request.urlretrieve(url, path)
    print(f"{label} model downloaded.")


def ensure_models():
    _download(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarker")
    _download(GESTURE_MODEL_PATH, GESTURE_MODEL_URL, "gesture recognizer")


class App:
    def __init__(self):
        ensure_models()

        capture_backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        self.cap = cv2.VideoCapture(0, capture_backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. On Windows, check Settings > Privacy & security "
                "> Camera > 'Let apps access your camera', and make sure no other app "
                "(Zoom, Teams, etc.) is holding the camera."
            )

        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        gesture_options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

        self._start_time = time.monotonic()

        self.detector = ExpressionDetector(smoothing_frames=FACE_SMOOTHING_FRAMES)
        self.gesture_detector = GestureDetector(smoothing_frames=GESTURE_SMOOTHING_FRAMES)
        self.popup = ReactionPopup(IMAGE_MAP, on_close=self._shutdown)

        self._calibration_samples = []
        self._calibration_start = None
        self._calibrated = False
        self._running = True

        self.popup.show_state(STATE_NEUTRAL)
        self.popup.set_status("Starting camera...")

    def _process_frame(self):
        """One webcam frame -> (frame, face ratios or None, raw gesture state or None)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        ratios = None
        tongue_redness = 0.0
        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            ratios = compute_ratios(face_landmarks)
            tongue_redness = compute_tongue_redness(frame, face_landmarks, ratios)

        gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
        raw_gesture = None
        if gesture_result.gestures and gesture_result.gestures[0]:
            top_category = gesture_result.gestures[0][0]
            hand_landmarks = gesture_result.hand_landmarks[0] if gesture_result.hand_landmarks else None
            raw_gesture = classify_gesture(top_category, hand_landmarks)

        return frame, ratios, tongue_redness, raw_gesture

    def _tick(self):
        if not self._running:
            return

        frame, ratios, tongue_redness, raw_gesture = self._process_frame()

        if not self._calibrated:
            if ratios is not None:
                self._run_calibration(ratios)
        else:
            # A held gesture is a more deliberate signal than a facial
            # expression, so it takes priority when present.
            face_state = (
                self.detector.update(ratios, tongue_redness)
                if ratios is not None
                else self.detector.confirmed_state
            )
            gesture_state = self.gesture_detector.update(raw_gesture)
            self.popup.show_state(gesture_state if gesture_state is not None else face_state)

        if frame is not None:
            cv2.putText(
                frame, f"state: {self.popup.current_state or '...'}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._shutdown()
                return

        if self._running:
            self.popup.after(FRAME_INTERVAL_MS, self._tick)

    def _run_calibration(self, ratios):
        now = time.monotonic()
        if self._calibration_start is None:
            self._calibration_start = now

        self._calibration_samples.append(ratios)
        remaining = max(0.0, CALIBRATION_SECONDS - (now - self._calibration_start))
        self.popup.set_status(f"Calibrating... hold a neutral face ({remaining:.1f}s)")

        if remaining <= 0 and len(self._calibration_samples) >= 5:
            self.detector.calibrate(self._calibration_samples)
            self._calibrated = True
            self.popup.set_status("Calibrated! Watching your face...")
            self.popup.show_state(STATE_NEUTRAL)

    def _shutdown(self):
        if not self._running:
            return
        self._running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.popup.destroy()

    def run(self):
        self.popup.after(FRAME_INTERVAL_MS, self._tick)
        self.popup.mainloop()


def main():
    try:
        app = App()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
