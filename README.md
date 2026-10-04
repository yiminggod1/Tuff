# Tuff · Gesture Canvas

Browser-based hand-tracking canvas built around natural human drawing kinematics.

## Gesture controls

Tuff starts directly in the gesture canvas after camera and hand-tracker startup.

- Index finger only → draw a persistent neon stroke
- Thumb + index pinch → grab and move the selected object; pinch on empty space pans the canvas
- Pinky only → select the hovered stroke and hold briefly to delete that **single** stroke
- Open palm → tracking only
- S → export PNG

Pinky deletion is intentionally target-scoped: it can remove one hovered stroke at a time and never clears the whole canvas.

## Motion model

Gesture geometry is computed from MediaPipe World Landmarks when available.

### Industrial anti-false-trigger pipeline

1. **Dynamic PalmScale normalization** — PalmScale is the 3D distance from wrist landmark 0 to middle-finger MCP landmark 9. Gesture distance thresholds are expressed as ratios of PalmScale rather than fixed hand-size distances.
2. **1€ filtering** — every X/Y/Z axis has an independent filter; the World Landmark bank uses `minCutoff = 0.30` and `beta = 0.007` for strong stationary jitter suppression with adaptive high-speed response.
3. **Palm normal compensation** — the normal from landmarks 0/5/17 estimates palm orientation. When the hand is side-on, Pinch uses a tighter PalmScale threshold and gesture entry requires 92% temporal agreement instead of 85%.
4. **Multi-finger coupling** — Pinch requires thumb/index bending plus middle/ring/pinky foldback; Pinky delete requires pinky dominance while the other fingers are explicitly folded; Index draw requires index dominance and the other three long fingers folded.
5. **15-frame Schmitt state machine** — entering a gesture requires at least 85% of the last 15 frames to agree (92% side-on). Once active, the gesture is held until its 15-frame agreement falls below 30%, then `onEnd()` is fired.
6. **Motion gate** — a new DRAW state is rejected while the index fingertip is moving faster than the configured PalmScale-per-second threshold. Once DRAW is active, intentional fast drawing remains responsive.

### Tuning guide

- `worldFilterMinCutoff`: lower = smoother at rest, higher = less latency.
- `worldFilterBeta`: higher = more responsive at speed, lower = more damping.
- `gestureEnterRatio`: higher = fewer false entries, but more deliberate gestures.
- `gestureMaintainRatio`: lower = stronger hold hysteresis; higher = faster release.
- `sideOnEnterRatio`: higher = safer when palm orientation is ambiguous.
- `pinchMaxPalmRatio`: lower = harder to pinch accidentally.
- `pinchSideOnPalmRatio`: lower = extra protection for side-on hands.
- `drawMaxStartPalmSpeeds`: lower = harder to enter DRAW while repositioning quickly.

Captured samples are resampled for stable spacing and rendered as continuous Bézier segments. Slow micro-movements are damped while intentional motion receives lower latency.
## Stack

- HTML / CSS / JavaScript
- MediaPipe Tasks Vision HandLandmarker
- Canvas 2D
- GitHub Pages via Actions

## Deployment

The pages workflow deploys the site to GitHub Pages on pushes to main.
