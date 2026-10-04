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

The pointer is not treated as a raw mouse cursor. It uses adaptive filtering driven by estimated fingertip speed and local path curvature. Slow micro-movements are attenuated while intentional fast movement receives a faster response. Tight turns receive slightly more damping, reflecting the documented speed–curvature relationship in human drawing movements.

Captured samples are resampled for stable spacing and rendered as continuous Bézier segments. Gesture geometry uses MediaPipe 3D world landmarks when available, while the on-screen cursor uses mirrored image coordinates.

## Anti-false-trigger pipeline

1. **1€ landmark filtering** smooths the 21 image/world landmark streams before geometry is evaluated.
2. **300 ms temporal voting** requires at least 5 observations and an 80% qualifying majority before entering a gesture.
3. **Buffered release** prevents a single noisy frame from dropping an active gesture.
4. **Draw speed gate** blocks a new DRAW activation while the index fingertip is moving too fast; once drawing is already engaged, normal intentional motion is allowed.
5. **Pinch geometry** combines 3D tip distance, thumb/index joint bending, and palm-normal orientation.
6. **Pinky geometry** requires the pinky to dominate while index/middle/ring are curled and the thumb is not in an open-palm state.

These gates are deliberately redundant: a frame must survive filtering, temporal consistency, anatomy, and motion checks before it can trigger an action.

## Stack

- HTML / CSS / JavaScript
- MediaPipe Tasks Vision HandLandmarker
- Canvas 2D
- GitHub Pages via Actions

## Deployment

The pages workflow deploys the site to GitHub Pages on pushes to main.
