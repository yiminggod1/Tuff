# Tuff · Gesture Canvas

Browser-based hand-tracking canvas built around natural human drawing kinematics.

## Gesture controls

- Index finger only → draw a persistent neon stroke
- Thumb + index pinch → grab and move the selected object; pinch on empty space pans the canvas
- Pinky only → select the hovered stroke and hold briefly to delete that single stroke
- Open palm → tracking only
- S → export PNG

Pinky deletion is intentionally target-scoped: it can remove one hovered stroke at a time and never clears the whole canvas.

## Motion model

The pointer is not treated as a raw mouse cursor. It uses adaptive filtering driven by estimated fingertip speed and local path curvature. Slow micro-movements are attenuated while intentional fast movement receives a faster response. Tight turns receive slightly more damping, reflecting the documented speed–curvature relationship in human drawing movements.

Captured samples are resampled for stable spacing and rendered as continuous Bézier segments. Particle emission follows movement speed and is reduced around tighter turns.

## Stack

- HTML / CSS / JavaScript
- MediaPipe Tasks Vision HandLandmarker
- Canvas 2D
- GitHub Pages via Actions

## Deployment

The pages workflow deploys the site to GitHub Pages on pushes to main.
