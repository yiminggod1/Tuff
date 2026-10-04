# Tuff · Gesture Canvas

Browser-based hand-tracking canvas built around natural human drawing kinematics.

## Web interaction

- Open palm → tracking only
- Index finger only → draw
- Thumb + index pinch → grab / drag
- Fist over a stroke → delete that stroke
- C → clear
- S → export PNG

## Motion model

The pointer is not treated as a raw mouse cursor. It uses adaptive filtering driven by estimated fingertip speed and local path curvature. Slow micro-movements are attenuated while intentional fast movement receives a faster response. Tight turns receive slightly more damping, reflecting the documented speed–curvature relationship in human drawing movements.

Captured samples are resampled for stable spacing and rendered as continuous Bézier segments. Particle emission also follows motion speed and is reduced around tighter turns.

## Stack

- HTML / CSS / JavaScript
- MediaPipe Tasks Vision HandLandmarker
- Canvas 2D
- GitHub Pages via Actions

## Deployment

The pages.yml workflow deploys the root site to GitHub Pages on pushes to main.
