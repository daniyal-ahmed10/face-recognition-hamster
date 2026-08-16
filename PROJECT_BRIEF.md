# Face Reaction Popup — Project Brief

## What this is
A real-time webcam app that detects my facial expression and pops up a matching image (meme-style) based on what I'm doing — e.g. confused face → confused image, happy face → happy image.

## Goal for this session
Build a working v1 fast. Rule-based expression detection (no ML training needed). Get something running end-to-end today, then iterate.

## Tech stack
- Python
- OpenCV — webcam capture
- Mediapipe (Face Mesh) — facial landmark detection (468 points per frame)
- Tkinter — popup window to display the reaction image

## Core behavior
1. Open webcam feed, run Mediapipe Face Mesh on each frame.
2. Compute geometric ratios from landmarks each frame:
   - **Happy**: mouth corners raised relative to mouth center + mouth width increase
   - **Surprised**: vertical mouth opening exceeds threshold + eyebrows raised (distance from eyebrow to eye landmarks increases)
   - **Confused**: eyebrows furrowed/asymmetric (inner eyebrow points closer together) and/or head tilt
   - **Neutral**: default state when nothing else triggers
3. Smooth detection over ~10 frames (or a few hundred ms) so it doesn't flicker between states on noisy single-frame reads.
4. On a *state change* (not every frame), swap the image shown in a Tkinter window to the corresponding reaction image.
5. Include a **calibration step** at startup: sit neutral for ~3 seconds, record baseline ratios (everyone's resting face geometry is different), and use that baseline to set thresholds instead of hardcoding fixed numbers.

## Project structure (suggested — adjust as needed)
```
/images/          <- reaction images go here (happy.png, confused.png, surprised.png, neutral.png)
main.py           <- entry point, runs the loop
detector.py        <- landmark math / expression classification logic
popup.py           <- Tkinter display logic
requirements.txt
```

## What I'll provide
- Reaction images (I'll drop them in /images/ — for now use placeholder colored squares with text labels so the app is testable before I add real images)

## Non-goals for v1
- No ML model training
- No dataset collection
- No deployment/packaging — just needs to run locally via `python main.py`

## Notes
- I'm on Windows — please account for webcam permission quirks for that OS.
- Prioritize getting a runnable end-to-end loop first, even if detection accuracy is rough — I'll tune thresholds after seeing it work.