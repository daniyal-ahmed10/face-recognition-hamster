"""Rule-based facial expression classification from Mediapipe Face Landmarker points.

No ML training involved: we compute a handful of geometric ratios from the 468+
face landmarks each frame, compare them against a per-user neutral baseline
recorded at calibration time, and classify based on how far things have moved.
"""
import math
from collections import Counter, deque

# Landmark indices (Mediapipe Face Mesh / Face Landmarker topology).
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
LEFT_EYEBROW_TOP = 105
RIGHT_EYEBROW_TOP = 334
LEFT_EYEBROW_INNER = 55
RIGHT_EYEBROW_INNER = 285
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UP_INNER = 13
MOUTH_LOW_INNER = 14
MOUTH_TOP_OUTER = 0
MOUTH_BOTTOM_OUTER = 17
MOUTH_INNER_LEFT = 78
MOUTH_INNER_RIGHT = 308

STATE_NEUTRAL = "neutral"
STATE_HAPPY = "happy"
STATE_SURPRISED = "surprised"
STATE_CONFUSED = "confused"
STATE_SILLY = "silly"

# Tongue-out detection: the face mesh has no points on the tongue itself, so
# geometry can't see it. Instead we sample pixel color inside the open mouth
# and check whether it reads as pink/red (tongue) rather than dark (open
# cavity) or white (teeth). MOUTH_OPEN_GATE avoids sampling lip color on a
# closed/near-closed mouth.
MOUTH_OPEN_GATE = 0.15
TONGUE_REDNESS_THRESHOLD = 0.12


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def compute_ratios(landmarks):
    """Turn one frame of landmarks into scale-invariant geometric ratios.

    Everything is normalized by inter-eye distance so results stay stable as
    you move closer to / further from the camera.
    """
    lm = landmarks
    d = _dist(lm[LEFT_EYE_OUTER], lm[RIGHT_EYE_OUTER])
    if d < 1e-6:
        d = 1e-6

    mouth_width = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / d
    mouth_open = _dist(lm[MOUTH_UP_INNER], lm[MOUTH_LOW_INNER]) / d

    mouth_center_y = (lm[MOUTH_TOP_OUTER].y + lm[MOUTH_BOTTOM_OUTER].y) / 2
    corner_avg_y = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2
    # Positive => corners sit above the lip midline (image y grows downward) => smile curve.
    corner_raise = (mouth_center_y - corner_avg_y) / d

    left_brow_raise = _dist(lm[LEFT_EYEBROW_TOP], lm[LEFT_EYE_TOP]) / d
    right_brow_raise = _dist(lm[RIGHT_EYEBROW_TOP], lm[RIGHT_EYE_TOP]) / d
    eyebrow_raise = (left_brow_raise + right_brow_raise) / 2
    eyebrow_asym = abs(left_brow_raise - right_brow_raise)

    eyebrow_furrow = _dist(lm[LEFT_EYEBROW_INNER], lm[RIGHT_EYEBROW_INNER]) / d

    dx = lm[RIGHT_EYE_OUTER].x - lm[LEFT_EYE_OUTER].x
    dy = lm[RIGHT_EYE_OUTER].y - lm[LEFT_EYE_OUTER].y
    head_tilt = math.degrees(math.atan2(dy, dx))

    return {
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
        "corner_raise": corner_raise,
        "eyebrow_raise": eyebrow_raise,
        "eyebrow_asym": eyebrow_asym,
        "eyebrow_furrow": eyebrow_furrow,
        "head_tilt": head_tilt,
    }


def compute_tongue_redness(frame_bgr, landmarks, ratios):
    """Rough color heuristic for 'tongue sticking out'. Samples the pixels
    inside the open mouth and scores how pink/red they are. Returns 0.0 if
    the mouth isn't open enough to bother sampling.
    """
    if ratios["mouth_open"] < MOUTH_OPEN_GATE:
        return 0.0

    h, w = frame_bgr.shape[:2]
    lm = landmarks
    x1 = int(min(lm[MOUTH_INNER_LEFT].x, lm[MOUTH_INNER_RIGHT].x) * w)
    x2 = int(max(lm[MOUTH_INNER_LEFT].x, lm[MOUTH_INNER_RIGHT].x) * w)
    y1 = int(min(lm[MOUTH_UP_INNER].y, lm[MOUTH_LOW_INNER].y) * h)
    y2 = int(max(lm[MOUTH_UP_INNER].y, lm[MOUTH_LOW_INNER].y) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0

    patch = frame_bgr[y1:y2, x1:x2].reshape(-1, 3).astype(float)
    b, g, r = patch[:, 0].mean(), patch[:, 1].mean(), patch[:, 2].mean()
    if (r + g + b) / 3 < 25:  # too dark - likely open-cavity shadow, not tongue
        return 0.0
    return max(0.0, ((r - g) + (r - b)) / 255.0)


class ExpressionDetector:
    """Classifies ratios against a calibrated baseline, smoothed over N frames."""

    def __init__(self, smoothing_frames=10):
        self.baseline = None
        self._history = deque(maxlen=smoothing_frames)
        self.confirmed_state = STATE_NEUTRAL

    def calibrate(self, samples):
        """samples: list of ratio dicts collected while the user holds a neutral face."""
        keys = samples[0].keys()
        self.baseline = {k: sum(s[k] for s in samples) / len(samples) for k in keys}
        self._history.clear()
        self.confirmed_state = STATE_NEUTRAL

    def classify_raw(self, ratios, tongue_redness=0.0):
        """Single-frame classification against baseline. Noisy by design -
        `update()` smooths this out over time."""
        if self.baseline is None:
            return STATE_NEUTRAL
        b = self.baseline

        # Checked first: an open mouth with tongue showing would otherwise
        # get misread as "surprised" (both involve a wide-open mouth).
        if tongue_redness > TONGUE_REDNESS_THRESHOLD:
            return STATE_SILLY

        is_surprised = (
            ratios["mouth_open"] > b["mouth_open"] + 0.06
            and ratios["eyebrow_raise"] > b["eyebrow_raise"] * 1.12
        )
        is_happy = (
            ratios["mouth_width"] > b["mouth_width"] * 1.10
            and ratios["corner_raise"] > b["corner_raise"] + 0.015
        )
        is_confused = (
            ratios["eyebrow_furrow"] < b["eyebrow_furrow"] * 0.90
            or ratios["eyebrow_asym"] > b["eyebrow_asym"] + 0.02
            or abs(ratios["head_tilt"] - b["head_tilt"]) > 12
        )

        # Priority order matters when multiple trigger at once (e.g. a big
        # surprised face can also look "wide"). Tune thresholds above first.
        if is_surprised:
            return STATE_SURPRISED
        if is_happy:
            return STATE_HAPPY
        if is_confused:
            return STATE_CONFUSED
        return STATE_NEUTRAL

    def update(self, ratios, tongue_redness=0.0):
        """Feed one frame's ratios in. Returns the current smoothed state.

        Uses majority vote over the last `smoothing_frames` frames so a
        single noisy frame can't flip the displayed reaction.
        """
        raw = self.classify_raw(ratios, tongue_redness)
        self._history.append(raw)

        if len(self._history) == self._history.maxlen:
            state, count = Counter(self._history).most_common(1)[0]
            if count > self._history.maxlen // 2:
                self.confirmed_state = state
        return self.confirmed_state
