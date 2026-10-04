# Tuff · Gesture Canvas

Browser-based hand-tracking canvas built around natural human drawing kinematics.

## Gesture controls

Tuff does not drop the user directly into drawing. After the camera starts, a guided personal calibration appears over the **live camera feed**.

The calibration teaches one pose at a time:

1. **Open hand** → tracking
2. **Thumb + index pinch** → move
3. **Pinky only** → select/delete one object
4. **Index only** → draw

For every step, the user sees a finger-by-finger pose guide, the real camera image, the detected 21-point hand skeleton, a true 3D world-landmark inset, and five live checks:

- hand visibility / framing
- gesture-shape match
- 3D depth quality
- movement stability
- personal fit

A pose is accepted only after all required checks are healthy and the user holds still through the confirmation bar.

## Personal multi-dimensional calibration

The calibration is intentionally more than a gesture-name test. Tuff builds a per-session personal hand profile from multiple 3D dimensions:

- 21 landmarks in canonical 3D coordinates
- wrist-centered and palm-axis normalized hand shape
- scale-normalized finger geometry
- finger extension/curl baselines
- thumb-index pinch ratio
- short-term motion stability
- filtered fingertip motion speed
- hand framing and depth structure

The learned 3D template is then blended with the anatomical gesture detector. This lets the app be stricter about **who is being tracked** and less dependent on one fixed set of hand-angle thresholds.

No raw camera frames are uploaded by Tuff. The personal calibration profile lives in browser memory for the current session.

## Interaction

- Index finger only → draw a persistent neon stroke
- Thumb + index pinch → grab and move the selected object; pinch on empty space pans the canvas
- Pinky only → select the hovered stroke and hold briefly to delete that **single** stroke
- Open palm → tracking only
- S → export PNG

Pinky deletion is intentionally target-scoped: it can remove one hovered stroke at a time and never clears the whole canvas.

## Motion model

The pointer is not treated as a raw mouse cursor. It uses adaptive filtering driven by estimated fingertip speed and local path curvature. Slow micro-movements are attenuated while intentional fast movement receives a faster response. Tight turns receive slightly more damping, reflecting the documented speed–curvature relationship in human drawing movements.

Captured samples are resampled for stable spacing and rendered as continuous Bézier segments. The cursor uses mirrored image coordinates, while gesture geometry uses MediaPipe 3D world landmarks when available.

## Stack

- HTML / CSS / JavaScript
- MediaPipe Tasks Vision HandLandmarker
- Canvas 2D
- GitHub Pages via Actions

## Deployment

The pages workflow deploys the site to GitHub Pages on pushes to main.


## Anti-false-trigger pipeline

Gesture transitions use several independent gates:

1. **1€ landmark filtering** smooths the 21 image/world landmark streams before geometry is evaluated.
2. **300 ms temporal voting** requires at least 5 observations and an 80% qualifying majority before entering a gesture.
3. **Buffered release** prevents a single noisy frame from dropping an active gesture.
4. **Draw speed gate** blocks a new DRAW activation while the index fingertip is moving too fast; once drawing is already engaged, normal intentional motion is allowed.
5. **Pinch geometry** combines 3D tip distance, thumb/index joint bending, and palm-normal orientation.
6. **Pinky geometry** requires the pinky to dominate while index/middle/ring are curled and the thumb is not in an open-palm state.

These gates are deliberately redundant: a frame must survive filtering, temporal consistency, anatomy, and motion checks before it can trigger an action.
