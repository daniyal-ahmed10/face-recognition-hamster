"""Rule-based facial expression classification from Mediapipe Face Landmarker points.

No ML training involved: we compute a handful of geometric ratios from the 468+
face landmarks each frame, compare them against a per-user neutral baseline
recorded at calibration time, and classify based on how far things have moved.
"""
import math
from collections import Counter, deque

import cv2

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
CHEEK_LEFT = 50
CHEEK_RIGHT = 280

STATE_NEUTRAL = "neutral"
STATE_HAPPY = "happy"
STATE_SURPRISED = "surprised"
STATE_CONFUSED = "confused"
STATE_SILLY = "silly"

# Mouth-content detection: the face mesh has no points on the tongue or teeth
# themselves, so geometry can't see them. Instead we sample pixel color inside
# the open mouth (in HSV, which is far more lighting-robust than raw RGB) and
# check what fraction reads as pink/red (tongue) vs. white (teeth).
# MOUTH_SAMPLE_GATE avoids sampling lip/skin color on a closed mouth.
#
# Tongue color is compared *relative to your own cheek skin* (sampled at
# calibration time), not an absolute hue cutoff: skin tone and tongue color
# sit in overlapping hue ranges, especially warmer skin tones, so a fixed
# threshold reads ordinary mouth-corner/chin skin as "tongue" for some
# people. Saturation - how vivid vs. pale a color is - is what actually
# separates them: a tongue reads distinctly more saturated than resting skin.
DEFAULT_SKIN_SATURATION = 60.0
TONGUE_SATURATION_MARGIN = 35.0
MOUTH_SAMPLE_GATE = 0.08
TONGUE_FRACTION_THRESHOLD = 0.15
TEETH_FRACTION_THRESHOLD = 0.10


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
        "scale": d,  # raw (un-normalized) inter-eye distance, for callers that need it
    }


def _hsv_patch(frame_bgr, x1, x2, y1, y2, w, h):
    x1, x2 = int(max(0.0, x1) * w), int(min(1.0, x2) * w)
    y1, y2 = int(max(0.0, y1) * h), int(min(1.0, y2) * h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)


def compute_skin_saturation(frame_bgr, landmarks):
    """Average HSV saturation of a small patch on each cheek - the resting-skin
    baseline that tongue detection compares against. Call during calibration
    and average across a few frames.
    """
    h, w = frame_bgr.shape[:2]
    lm = landmarks
    radius = 0.02
    sats = []
    for idx in (CHEEK_LEFT, CHEEK_RIGHT):
        cx, cy = lm[idx].x, lm[idx].y
        patch = _hsv_patch(frame_bgr, cx - radius, cx + radius, cy - radius, cy + radius, w, h)
        if patch is not None:
            sats.append(float(patch[..., 1].mean()))
    return sum(sats) / len(sats) if sats else DEFAULT_SKIN_SATURATION


def sample_mouth_colors(frame_bgr, landmarks, ratios, skin_saturation=DEFAULT_SKIN_SATURATION):
    """Samples pixels inside the mouth and returns (teeth_fraction, tongue_fraction) -
    what fraction of the sampled patch reads as white (teeth) vs. pink/red and
    notably more saturated than resting skin (tongue). Both are 0.0 if the
    mouth isn't open enough to bother sampling.

    Uses two different boxes: teeth are measured with a tight box right at
    the mouth opening (a smile's teeth band sits exactly there - padding
    would just dilute the fraction with skin/chin pixels), while tongue is
    measured with a box padded well beyond it, especially downward, since a
    protruding tongue extends past where the tracked inner-lip landmarks sit.
    """
    if ratios["mouth_open"] < MOUTH_SAMPLE_GATE:
        return 0.0, 0.0

    h, w = frame_bgr.shape[:2]
    lm = landmarks
    x1 = min(lm[MOUTH_INNER_LEFT].x, lm[MOUTH_INNER_RIGHT].x)
    x2 = max(lm[MOUTH_INNER_LEFT].x, lm[MOUTH_INNER_RIGHT].x)
    y1 = min(lm[MOUTH_UP_INNER].y, lm[MOUTH_LOW_INNER].y)
    y2 = max(lm[MOUTH_UP_INNER].y, lm[MOUTH_LOW_INNER].y)
    box_w, box_h = x2 - x1, y2 - y1

    teeth_fraction = 0.0
    teeth_hsv = _hsv_patch(frame_bgr, x1, x2, y1, y2, w, h)
    if teeth_hsv is not None:
        sat, val = teeth_hsv[..., 1].astype(float), teeth_hsv[..., 2].astype(float)
        teeth_fraction = float(((sat < 90) & (val > 90)).mean())

    tongue_fraction = 0.0
    tongue_hsv = _hsv_patch(
        frame_bgr, x1 - box_w * 0.15, x2 + box_w * 0.15, y1, y2 + box_h * 0.45, w, h
    )
    if tongue_hsv is not None:
        hue = tongue_hsv[..., 0].astype(float)
        sat, val = tongue_hsv[..., 1].astype(float), tongue_hsv[..., 2].astype(float)
        is_tongue = (
            ((hue < 20) | (hue > 160))
            & (sat > skin_saturation + TONGUE_SATURATION_MARGIN)
            & (val > 40)
        )
        tongue_fraction = float(is_tongue.mean())

    return teeth_fraction, tongue_fraction


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

    def classify_raw(self, ratios, teeth_fraction=0.0, tongue_fraction=0.0):
        """Single-frame classification against baseline. Noisy by design -
        `update()` smooths this out over time."""
        if self.baseline is None:
            return STATE_NEUTRAL
        b = self.baseline

        # Checked first: an open mouth with tongue showing would otherwise
        # get misread as "surprised" (both involve a wide-open mouth).
        if tongue_fraction > TONGUE_FRACTION_THRESHOLD:
            return STATE_SILLY

        # Teeth visibility is the real "smiling" signal; mouth_width tells a
        # smile (wide, corners pulled back) apart from a shocked wide-open
        # mouth (drops vertically without necessarily widening) - both can
        # show teeth, so this is what keeps them from colliding.
        is_happy = (
            teeth_fraction > TEETH_FRACTION_THRESHOLD
            and ratios["mouth_width"] > b["mouth_width"] * 1.05
        )
        is_surprised = (
            ratios["mouth_open"] > b["mouth_open"] + 0.10
            or (
                ratios["mouth_open"] > b["mouth_open"] + 0.05
                and ratios["eyebrow_raise"] > b["eyebrow_raise"] * 1.10
            )
        )
        is_confused = (
            ratios["eyebrow_furrow"] < b["eyebrow_furrow"] * 0.90
            or ratios["eyebrow_asym"] > b["eyebrow_asym"] + 0.02
            or abs(ratios["head_tilt"] - b["head_tilt"]) > 12
        )

        # Priority order matters when multiple trigger at once (e.g. a big
        # surprised face can also look "wide"). Tune thresholds above first.
        if is_happy:
            return STATE_HAPPY
        if is_surprised:
            return STATE_SURPRISED
        if is_confused:
            return STATE_CONFUSED
        return STATE_NEUTRAL

    def update(self, ratios, teeth_fraction=0.0, tongue_fraction=0.0):
        """Feed one frame's ratios in. Returns the current smoothed state.

        Uses majority vote over the last `smoothing_frames` frames so a
        single noisy frame can't flip the displayed reaction.
        """
        raw = self.classify_raw(ratios, teeth_fraction, tongue_fraction)
        self._history.append(raw)

        if len(self._history) == self._history.maxlen:
            state, count = Counter(self._history).most_common(1)[0]
            if count > self._history.maxlen // 2:
                self.confirmed_state = state
        return self.confirmed_state
