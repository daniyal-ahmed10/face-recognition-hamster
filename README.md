# Face Reaction Popup

A real-time webcam app that watches your face and hands, and pops up a matching
reaction image based on what you're doing — smile and a happy gif shows up,
flex your arm and you get a strength meme, stick your tongue out and it knows.

Rule-based detection throughout — no ML training or datasets involved. It uses
Google's pretrained Mediapipe models (face landmarks, hand gestures) purely for
tracking, then classifies expressions with geometry and simple thresholds.

## Features

| Trigger | Reaction |
|---|---|
| Neutral face | `neutral.gif` |
| Smile (mouth widens + corners raise) | `happy.gif` |
| Surprised (mouth drops open + eyebrows raise) | `scared.webp` |
| Confused (eyebrows furrow/asymmetric, or head tilts) | `nerd.gif` |
| 👆 Point your index finger up | `nerd.gif` |
| 👍 Thumbs up | `thumbsup.gif` |
| ✊ Raised fist (flex) | `strong.gif` |
| 👅 Tongue out | `silly.gif` |

A held hand gesture always overrides whatever your face is doing, since it's a
more deliberate signal. Everything is smoothed over several frames so it
doesn't flicker between states on noisy single-frame reads.

## Setup

Requires Python 3.10+ and a webcam.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python main.py
```

On first run it downloads two small Mediapipe model bundles (~12MB total,
one-time, cached in `models/`).

You'll get two windows:
- **Face Reaction** — the popup showing the current reaction image.
- **Webcam (press q to quit)** — a live debug feed with the detected state
  overlaid. Press `q` there, or just close the popup, to exit.

The app starts with a **3-second calibration**: hold a neutral, relaxed face
so it can record your personal baseline (everyone's resting face geometry is
different). It uses that baseline instead of hardcoded thresholds.

## How it works

- **`detector.py`** — computes geometric ratios (mouth width/openness, corner
  raise, eyebrow raise/asymmetry/furrow, head tilt) from the 468+ face
  landmarks each frame, normalized by inter-eye distance so it's stable
  regardless of how close you are to the camera. Classifies against your
  calibrated baseline. Tongue-out is a separate color heuristic — the face
  mesh has no landmarks on the tongue itself, so it instead samples the
  pixels inside an open mouth and checks if they read pink/red rather than
  dark (open cavity) or white (teeth).
- **`gesture.py`** — maps Mediapipe's pretrained hand gesture classifier
  (`Thumb_Up`, `Pointing_Up`, `Closed_Fist`, etc.) to reaction states. A
  closed fist only counts as a "flex" if it's held up near shoulder/head
  height, not resting at your side.
- **`popup.py`** — the Tkinter window that swaps the displayed image (static
  or animated) whenever the confirmed state changes.
- **`main.py`** — wires it all together: webcam capture, calibration,
  per-frame face + gesture inference, and state resolution.

## Tuning

Everything is a constant at the top of `detector.py` and `gesture.py` —
mouth/eyebrow thresholds, the tongue color threshold, the fist-height cutoff,
and the smoothing window sizes. If a state is over- or under-triggering,
that's the place to adjust.

## Project structure

```
main.py           entry point, webcam loop, calibration, state resolution
detector.py       facial landmark math + expression/tongue classification
gesture.py        hand gesture classification
popup.py          Tkinter popup display logic
images/           reaction images/gifs
models/           auto-downloaded Mediapipe model bundles (gitignored)
requirements.txt
```
