"""Rule-based mapping from hand gestures/pose to reaction states.

Thumb_Up and Pointing_Up use Mediapipe's pretrained Gesture Recognizer task
bundle (no training of our own, same spirit as the face landmark model),
which classifies each detected hand into one of a fixed set of canned
gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
Victory, ILoveYou.

The "flex" (strong) gesture does NOT use the canned Closed_Fist category -
that classifier is appearance-based and tuned for a fist facing the camera,
so it stops recognizing it once the fist is rotated (e.g. knuckles turned
toward your head, like a real bicep-flex pose). Instead we detect a fist
geometrically from the hand landmarks (rotation-invariant: are the fingers
curled toward the palm, regardless of which way the hand is facing) and
require it to be near your head, using the face landmarks main.py already
tracks each frame.

A held gesture is a more deliberate, intentional signal than a facial
expression, so main.py lets a confirmed gesture override the facial state.
"""
import math
from collections import Counter, deque

GESTURE_THUMBS_UP = "Thumb_Up"
GESTURE_POINTING_UP = "Pointing_Up"

WRIST = 0
MIDDLE_MCP = 9  # roughly the center of the knuckles/fist

# A finger counts as "curled" if its tip sits within this fraction of the
# wrist-to-knuckle distance (an extended finger reaches well past that).
# Requires ALL 4 fingers curled - Pointing_Up also curls 3 of them (all but
# the index), so requiring only 3 here swallowed that gesture as a "fist".
FIST_CURL_RATIO = 0.85
FIST_MIN_CURLED_FINGERS = 4

# How close the fist must be to your head to count as a flex, in units of
# inter-eye distance (the same scale-invariant unit detector.py uses).
# Generous by design since arm length/camera framing varies a lot - tune
# after testing your own setup.
FLEX_MAX_HAND_TO_HEAD_DIST = 5.0


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _is_fist(hand_landmarks):
    wrist = (hand_landmarks[WRIST].x, hand_landmarks[WRIST].y)
    tips = (8, 12, 16, 20)
    mcps = (5, 9, 13, 17)
    curled = 0
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip = (hand_landmarks[tip_idx].x, hand_landmarks[tip_idx].y)
        mcp = (hand_landmarks[mcp_idx].x, hand_landmarks[mcp_idx].y)
        if _dist(tip, wrist) < _dist(mcp, wrist) * FIST_CURL_RATIO:
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS


def classify_gesture(top_category, hand_landmarks, head_center=None, head_scale=None):
    """One frame of gesture + hand landmark output -> a reaction state name, or None.

    head_center: (x, y) in the same normalized image coords as hand_landmarks,
    e.g. the midpoint between the eyes. head_scale: inter-eye distance in
    that same coordinate space, used to normalize the proximity check.
    """
    if hand_landmarks and _is_fist(hand_landmarks):
        if head_center is not None and head_scale:
            fist_center = (hand_landmarks[MIDDLE_MCP].x, hand_landmarks[MIDDLE_MCP].y)
            if _dist(fist_center, head_center) / head_scale < FLEX_MAX_HAND_TO_HEAD_DIST:
                return "strong"
        return None  # a fist that isn't near your head isn't a flex

    if top_category is None:
        return None

    name = top_category.category_name
    if name == GESTURE_THUMBS_UP:
        return "thumbsup"
    if name == GESTURE_POINTING_UP:
        return "confused"
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
