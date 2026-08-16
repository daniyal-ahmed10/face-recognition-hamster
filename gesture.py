"""Rule-based mapping from Mediapipe's built-in hand gestures to reaction states.

Uses the pretrained Gesture Recognizer task bundle (no training of our own,
same spirit as the face landmark model) which classifies each detected hand
into one of a fixed set of canned gestures: None, Closed_Fist, Open_Palm,
Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.

A held gesture is a more deliberate, intentional signal than a facial
expression, so main.py lets a confirmed gesture override the facial state.
"""
from collections import Counter, deque

GESTURE_THUMBS_UP = "Thumb_Up"
GESTURE_POINTING_UP = "Pointing_Up"
GESTURE_CLOSED_FIST = "Closed_Fist"

WRIST = 0

# Wrist must be in the upper part of the frame (0.0 = top, 1.0 = bottom) for a
# closed fist to count as a "flex" rather than just a fist resting at your side.
FLEX_MAX_WRIST_Y = 0.6


def classify_gesture(top_category, hand_landmarks):
    """One frame of Gesture Recognizer output -> a reaction state name, or None."""
    if top_category is None:
        return None

    name = top_category.category_name
    if name == GESTURE_THUMBS_UP:
        return "thumbsup"
    if name == GESTURE_POINTING_UP:
        return "confused"
    if name == GESTURE_CLOSED_FIST and hand_landmarks:
        if hand_landmarks[WRIST].y < FLEX_MAX_WRIST_Y:
            return "strong"
    return None


class GestureDetector:
    """Smooths raw per-frame gesture states over a short window (majority vote).

    Also naturally hands control back to the facial detector: once the
    gesture is released, `None` becomes the majority again and
    `confirmed_state` reverts to None.
    """

    def __init__(self, smoothing_frames=6):
        self._history = deque(maxlen=smoothing_frames)
        self.confirmed_state = None

    def update(self, raw_state):
        self._history.append(raw_state)
        if len(self._history) == self._history.maxlen:
            state, count = Counter(self._history).most_common(1)[0]
            if count > self._history.maxlen // 2:
                self.confirmed_state = state
        return self.confirmed_state
