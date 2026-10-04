import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const COLORS = {
  draw: "#FF4F91",
  hover: "#D8FF5A",
  hand: "#29E8D4",
  handPoint: "#85FFF3",
  white: "#FFFFFF",
};

const SETTINGS = {
  detectIntervalMs: 26,
  lostGraceMs: 110,
  gestureHoldMs: 58,
  pinchStart: 0.48,
  pinchRelease: 0.63,
  collisionRadius: 28,
  hoverExitRadius: 43,
  deleteHoldMs: 620,
  drawStartMove: 3.0,
  drawMinMove: 1.35,
  resampleStep: 3.0,
  brushWidth: 3.15,
  maxParticles: 180,
};

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

const video = document.querySelector("#video");
const stage = document.querySelector("#stage");
const drawingCanvas = document.querySelector("#drawingCanvas");
const effectsCanvas = document.querySelector("#effectsCanvas");
const drawCtx = drawingCanvas.getContext("2d");
const fxCtx = effectsCanvas.getContext("2d");
const startScreen = document.querySelector("#startScreen");
const startButton = document.querySelector("#startButton");
const startError = document.querySelector("#startError");
const status = document.querySelector("#status");
const motionLabel = document.querySelector("#motion");

let landmarker = null;
let stream = null;
let running = false;
let lastVideoTime = -1;
let lastDetectAt = 0;
let latestHand = null;
let activePoints = [];
let stableGesture = "IDLE";
let gestureCandidate = "IDLE";
let gestureCandidateAt = 0;
let pinchActive = false;

let rawPoint = null;
let filteredPoint = null;
let velocity = { x: 0, y: 0 };
let lastSampleAt = 0;
let motionSamples = [];
let lastCurvature = 0;
let lastSpeed = 0;

let strokes = [];
let strokeOffsets = [];
let strokeBounds = [];
let currentStroke = null;
let drawAnchor = null;
let hoveredStrokeIndex = -1;
let grabbedStrokeIndex = -1;
let previousGrabPoint = null;
let canvasPanning = false;
let deleteTarget = -1;
let deleteStartedAt = 0;
let deleteConsumed = false;
let panX = 0;
let panY = 0;

let lastFrameAt = performance.now();
const particles = [];
const flashes = [];

function size() {
  return { width: stage.clientWidth, height: stage.clientHeight };
}

function resizeCanvas() {
  const { width, height } = size();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  for (const canvas of [drawingCanvas, effectsCanvas]) {
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
  }

  drawCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fxCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  fxCtx.lineCap = "round";
  fxCtx.lineJoin = "round";
  redraw();
}

function displayPoint(x, y) {
  const { width, height } = size();
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const scale = Math.max(width / vw, height / vh);
  const renderedW = vw * scale;
  const renderedH = vh * scale;
  const cropX = (renderedW - width) / 2;
  const cropY = (renderedH - height) / 2;

  return {
    x: width - (x * renderedW - cropX),
    y: y * renderedH - cropY,
  };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function jointAngle(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const abz = (a.z || 0) - (b.z || 0);
  const cbx = c.x - b.x;
  const cby = c.y - b.y;
  const cbz = (c.z || 0) - (b.z || 0);
  const denom = Math.hypot(abx, aby, abz) * Math.hypot(cbx, cby, cbz);
  if (!denom) return 0;
  return Math.acos(Math.max(-1, Math.min(1, (abx * cbx + aby * cby + abz * cbz) / denom))) * 180 / Math.PI;
}

function fingerScore(hand, mcp, pip, dip, tip) {
  const palm = Math.max(.0001, dist(hand[0], hand[9]));
  const direct = dist(hand[mcp], hand[tip]);
  const chain = dist(hand[mcp], hand[pip]) + dist(hand[pip], hand[dip]) + dist(hand[dip], hand[tip]);
  const straightness = chain ? direct / chain : 0;
  const reach = direct / palm;
  const a1 = jointAngle(hand[mcp], hand[pip], hand[dip]);
  const a2 = jointAngle(hand[pip], hand[dip], hand[tip]);

  const reachScore = Math.max(0, Math.min(1, (reach - .68) / .62));
  const straightScore = Math.max(0, Math.min(1, (straightness - .70) / .25));
  const angleScore = Math.max(0, Math.min(1, ((a1 - 108) / 62 + (a2 - 112) / 68) * .5));

  return reachScore * .42 + straightScore * .30 + angleScore * .28;
}

function fingerFlags(hand) {
  const scores = {
    index: fingerScore(hand, 5, 6, 7, 8),
    middle: fingerScore(hand, 9, 10, 11, 12),
    ring: fingerScore(hand, 13, 14, 15, 16),
    pinky: fingerScore(hand, 17, 18, 19, 20),
  };

  const palm = Math.max(.0001, dist(hand[0], hand[9]));
  return {
    thumb: dist(hand[4], hand[5]) / palm > .43,
    index: scores.index >= .50 && scores.index >= (scores.middle + scores.ring + scores.pinky) / 3 + .04,
    middle: scores.middle >= .50,
    ring: scores.ring >= .50,
    pinky: scores.pinky >= .50,
  };
}

function pinchRatio(hand) {
  return dist(hand[4], hand[8]) / Math.max(.0001, dist(hand[0], hand[9]));
}

function updatePinch(ratio) {
  if (pinchActive) {
    if (ratio >= SETTINGS.pinchRelease) pinchActive = false;
  } else if (ratio <= SETTINGS.pinchStart) {
    pinchActive = true;
  }
  return pinchActive;
}

function classifyGesture(hand) {
  const f = fingerFlags(hand);
  if (updatePinch(pinchRatio(hand))) return "GRAB";
  if (f.index && !f.middle && !f.ring && !f.pinky) return "DRAW";
  if (f.thumb && f.index && f.middle && f.ring && f.pinky) return "TRACK";
  if (!f.index && !f.middle && !f.ring && !f.pinky) return "DELETE";
  return "IDLE";
}

function commitGesture(next, now) {
  if (next !== gestureCandidate) {
    gestureCandidate = next;
    gestureCandidateAt = now;
    return;
  }

  if (next !== stableGesture && now - gestureCandidateAt >= SETTINGS.gestureHoldMs) {
    const prev = stableGesture;
    stableGesture = next;
    onGestureChanged(prev, next);
  }
}

function onGestureChanged(prev, next) {
  if (prev === "DRAW" && next !== "DRAW") stopStroke();

  if (prev === "GRAB" && next !== "GRAB") {
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;
  }

  if (next === "DRAW") {
    drawAnchor = filteredPoint ? { ...filteredPoint } : null;
    grabbedStrokeIndex = -1;
    canvasPanning = false;
    previousGrabPoint = null;
    deleteTarget = -1;
    deleteConsumed = false;
    currentStroke = null;
  }

  if (next === "GRAB") {
    grabbedStrokeIndex = hoveredStrokeIndex;
    canvasPanning = grabbedStrokeIndex < 0;
    previousGrabPoint = filteredPoint ? { ...filteredPoint } : null;
  }

  if (next === "DELETE") {
    stopStroke();
    grabbedStrokeIndex = -1;
    canvasPanning = false;
    deleteTarget = findHoveredStroke(activePoints, SETTINGS.collisionRadius * 1.15);
    deleteStartedAt = deleteTarget >= 0 ? performance.now() : 0;
    deleteConsumed = false;
  }

  if (next !== "DELETE") {
    deleteTarget = -1;
    deleteStartedAt = 0;
    deleteConsumed = false;
  }
}

function estimateKinematics(point, now) {
  if (!rawPoint) {
    rawPoint = { ...point };
    lastSampleAt = now;
    motionSamples = [{ ...point, t: now }];
    return { speed: 0, curvature: 0 };
  }

  const dt = Math.max(.008, Math.min(.08, (now - lastSampleAt) / 1000));
  const step = dist(point, rawPoint);
  const speed = step / dt;

  motionSamples.push({ x: point.x, y: point.y, t: now });
  while (motionSamples.length > 9) motionSamples.shift();

  let curvature = 0;
  if (motionSamples.length >= 4) {
    const a = motionSamples[motionSamples.length - 3];
    const b = motionSamples[motionSamples.length - 2];
    const c = motionSamples[motionSamples.length - 1];

    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const bcx = c.x - b.x;
    const bcy = c.y - b.y;

    const ab = Math.hypot(abx, aby);
    const bc = Math.hypot(bcx, bcy);
    const cross = Math.abs(abx * bcy - aby * bcx);

    if (ab > .5 && bc > .5) {
      curvature = cross / (ab * bc * Math.max(1, (ab + bc) * .5));
    }
  }

  rawPoint = { ...point };
  lastSampleAt = now;
  return { speed, curvature };
}

function naturalFilter(point, now, mode) {
  if (!filteredPoint) {
    filteredPoint = { ...point };
    velocity = { x: 0, y: 0 };
    estimateKinematics(point, now);
    return { ...point };
  }

  const dt = Math.max(.008, Math.min(.06, (now - lastSampleAt) / 1000 || .016));
  const motion = estimateKinematics(point, now);
  const speedN = Math.min(1, motion.speed / 1150);

  // Adaptive one-euro-like response: human micro-tremor gets damped,
  // but intentional fast motion stays close to the measured fingertip.
  let alpha = .16 + .70 * Math.pow(speedN, .58);
  if (mode === "GRAB") alpha += .16;

  // Natural drawing often slows in tighter curves. Use curvature only
  // as a gentle brake, never as a hard geometric constraint.
  const curvatureBrake = 1 / (1 + 3.8 * Math.min(1, motion.curvature * 18));
  alpha *= .76 + .24 * curvatureBrake;

  alpha = Math.max(mode === "GRAB" ? .68 : .12, Math.min(mode === "GRAB" ? .98 : .86, alpha));

  const next = {
    x: filteredPoint.x + (point.x - filteredPoint.x) * alpha,
    y: filteredPoint.y + (point.y - filteredPoint.y) * alpha,
  };

  velocity = {
    x: (next.x - filteredPoint.x) / dt,
    y: (next.y - filteredPoint.y) / dt,
  };

  filteredPoint = next;
  return { ...next };
}

function predictPoint(now) {
  if (!filteredPoint || !lastSampleAt) return null;
  const age = now - lastSampleAt;
  if (age > SETTINGS.lostGraceMs) return null;

  const t = Math.min(.11, age / 1000);
  const curvatureBrake = 1 / (1 + lastCurvature * 18);
  const fade = Math.max(0, 1 - age / SETTINGS.lostGraceMs);

  return {
    x: filteredPoint.x + velocity.x * t * fade * (.22 + .18 * curvatureBrake),
    y: filteredPoint.y + velocity.y * t * fade * (.22 + .18 * curvatureBrake),
  };
}

function updateStrokeBounds(index, point) {
  const b = strokeBounds[index];
  b.minX = Math.min(b.minX, point.x);
  b.minY = Math.min(b.minY, point.y);
  b.maxX = Math.max(b.maxX, point.x);
  b.maxY = Math.max(b.maxY, point.y);
}

function startStroke(point) {
  currentStroke = [{ ...point }];
  strokes.push(currentStroke);
  strokeOffsets.push({ x: 0, y: 0 });
  strokeBounds.push({ minX: point.x, minY: point.y, maxX: point.x, maxY: point.y });
}

function addStrokePoint(point) {
  if (!currentStroke) {
    startStroke(point);
    return true;
  }

  const last = currentStroke[currentStroke.length - 1];
  const d = dist(last, point);
  if (d < SETTINGS.drawMinMove) return false;

  const steps = Math.max(1, Math.ceil(d / SETTINGS.resampleStep));
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const p = {
      x: last.x + (point.x - last.x) * t,
      y: last.y + (point.y - last.y) * t,
    };
    currentStroke.push(p);
    updateStrokeBounds(strokes.length - 1, p);
  }

  return true;
}

function stopStroke() {
  if (currentStroke) redraw();
  currentStroke = null;
  drawAnchor = null;
}

function strokePoint(index, i) {
  const o = strokeOffsets[index] || { x: 0, y: 0 };
  const p = strokes[index][i];
  return { x: p.x + o.x + panX, y: p.y + o.y + panY };
}

function distanceToSegment(p, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const denom = abx * abx + aby * aby;
  if (!denom) return dist(p, a);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / denom));
  return Math.hypot(p.x - (a.x + abx * t), p.y - (a.y + aby * t));
}

function findHoveredStroke(points, radius) {
  if (!points.length || !strokes.length) return -1;

  let best = -1;
  let bestDistance = radius;

  for (let i = 0; i < strokes.length; i++) {
    const b = strokeBounds[i];
    const o = strokeOffsets[i] || { x: 0, y: 0 };

    const minX = b.minX + o.x + panX - radius;
    const maxX = b.maxX + o.x + panX + radius;
    const minY = b.minY + o.y + panY - radius;
    const maxY = b.maxY + o.y + panY + radius;

    for (const p of points) {
      if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY) continue;

      for (let j = 1; j < strokes[i].length; j++) {
        const d = distanceToSegment(p, strokePoint(i, j - 1), strokePoint(i, j));
        if (d < bestDistance) {
          bestDistance = d;
          best = i;
        }
      }
    }
  }

  return best;
}

function deleteStroke(index) {
  if (index < 0 || index >= strokes.length) return;
  strokes.splice(index, 1);
  strokeOffsets.splice(index, 1);
  strokeBounds.splice(index, 1);
  hoveredStrokeIndex = -1;
  deleteTarget = -1;
  redraw();
}

function drawStroke(ctx, stroke, index, highlight) {
  if (!stroke.length) return;

  const offset = strokeOffsets[index] || { x: 0, y: 0 };
  ctx.save();
  ctx.translate(offset.x + panX, offset.y + panY);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (stroke.length === 1) {
    ctx.beginPath();
    ctx.fillStyle = highlight ? COLORS.hover : COLORS.draw;
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = highlight ? 18 : 12;
    ctx.arc(stroke[0].x, stroke[0].y, SETTINGS.brushWidth * .55, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(stroke[0].x, stroke[0].y);

  for (let i = 0; i < stroke.length - 1; i++) {
    const p0 = stroke[Math.max(0, i - 1)];
    const p1 = stroke[i];
    const p2 = stroke[i + 1];
    const p3 = stroke[Math.min(stroke.length - 1, i + 2)];

    const c1 = {
      x: p1.x + (p2.x - p0.x) / 6,
      y: p1.y + (p2.y - p0.y) / 6,
    };
    const c2 = {
      x: p2.x - (p3.x - p1.x) / 6,
      y: p2.y - (p3.y - p1.y) / 6,
    };

    ctx.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, p2.x, p2.y);
  }

  ctx.lineWidth = highlight ? SETTINGS.brushWidth + 1.15 : SETTINGS.brushWidth;
  ctx.strokeStyle = highlight ? COLORS.hover : COLORS.draw;
  ctx.shadowColor = ctx.strokeStyle;

  ctx.globalAlpha = .28;
  ctx.shadowBlur = highlight ? 24 : 22;
  ctx.stroke();

  ctx.globalAlpha = .86;
  ctx.shadowBlur = 8;
  ctx.stroke();

  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
  ctx.stroke();

  ctx.restore();
}

function redraw() {
  const { width, height } = size();
  drawCtx.clearRect(0, 0, width, height);

  for (let i = 0; i < strokes.length; i++) {
    drawStroke(drawCtx, strokes[i], i, i === hoveredStrokeIndex);
  }
}

function drawHand(points) {
  if (points.length !== 21) return;

  fxCtx.save();
  fxCtx.strokeStyle = COLORS.hand;
  fxCtx.shadowColor = COLORS.hand;
  fxCtx.shadowBlur = 9;
  fxCtx.lineWidth = 1.8;

  for (const [a, b] of CONNECTIONS) {
    fxCtx.beginPath();
    fxCtx.moveTo(points[a].x, points[a].y);
    fxCtx.lineTo(points[b].x, points[b].y);
    fxCtx.stroke();
  }

  fxCtx.shadowBlur = 0;

  for (const p of points) {
    fxCtx.beginPath();
    fxCtx.fillStyle = COLORS.handPoint;
    fxCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    fxCtx.fill();
  }

  fxCtx.restore();
}

function emitParticles(point, speed, curvature) {
  const turnBrake = 1 / (1 + curvature * 20);
  const count = Math.min(5, Math.max(1, Math.round(1 + speed / 620) * (turnBrake > .55 ? 1 : .55)));

  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const speedOut = 28 + Math.random() * (70 + Math.min(210, speed * .13));

    particles.push({
      x: point.x,
      y: point.y,
      vx: Math.cos(a) * speedOut,
      vy: Math.sin(a) * speedOut,
      life: 0,
      maxLife: .22 + Math.random() * .34,
      size: .7 + Math.random() * 1.8,
    });
  }

  while (particles.length > SETTINGS.maxParticles) particles.shift();
}

function drawEffects(dt) {
  const { width, height } = size();
  fxCtx.clearRect(0, 0, width, height);

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life += dt;

    if (p.life >= p.maxLife) {
      particles.splice(i, 1);
      continue;
    }

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vx *= .965;
    p.vy *= .965;

    const alpha = 1 - p.life / p.maxLife;

    fxCtx.save();
    fxCtx.globalAlpha = alpha;
    fxCtx.fillStyle = COLORS.white;
    fxCtx.shadowColor = COLORS.white;
    fxCtx.shadowBlur = 8;
    fxCtx.beginPath();
    fxCtx.arc(p.x, p.y, p.size * alpha, 0, Math.PI * 2);
    fxCtx.fill();
    fxCtx.restore();
  }

  for (let i = flashes.length - 1; i >= 0; i--) {
    const f = flashes[i];
    f.life += dt;

    if (f.life >= f.maxLife) {
      flashes.splice(i, 1);
      continue;
    }

    const t = f.life / f.maxLife;

    fxCtx.save();
    fxCtx.globalAlpha = 1 - t;
    fxCtx.strokeStyle = COLORS.white;
    fxCtx.shadowColor = COLORS.white;
    fxCtx.shadowBlur = 12;
    fxCtx.lineWidth = 1.3;
    fxCtx.beginPath();
    fxCtx.arc(f.x, f.y, 3 + t * 10, 0, Math.PI * 2);
    fxCtx.stroke();
    fxCtx.restore();
  }

  if (activePoints.length === 21) drawHand(activePoints);

  if (stableGesture === "DELETE" && deleteTarget >= 0 && deleteStartedAt && !deleteConsumed) {
    const progress = Math.max(0, Math.min(1, (performance.now() - deleteStartedAt) / SETTINGS.deleteHoldMs));
    const b = strokeBounds[deleteTarget];

    if (b) {
      const cx = (b.minX + b.maxX) / 2 + (strokeOffsets[deleteTarget]?.x || 0) + panX;
      const cy = (b.minY + b.maxY) / 2 + (strokeOffsets[deleteTarget]?.y || 0) + panY;
      const radius = Math.max(26, Math.hypot(b.maxX - b.minX, b.maxY - b.minY) * .18);

      fxCtx.save();
      fxCtx.strokeStyle = "rgba(255,105,135,.72)";
      fxCtx.shadowColor = "rgba(255,80,120,.9)";
      fxCtx.shadowBlur = 12;
      fxCtx.lineWidth = 2;
      fxCtx.beginPath();
      fxCtx.arc(cx, cy, radius + 8, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      fxCtx.stroke();
      fxCtx.restore();
    }
  }
}

function handleDrawing(point) {
  if (!point) return;

  if (!currentStroke) {
    if (!drawAnchor) drawAnchor = { ...point };

    if (dist(point, drawAnchor) >= SETTINGS.drawStartMove) {
      startStroke(drawAnchor);
      addStrokePoint(point);
      emitParticles(point, lastSpeed, lastCurvature);
      flashes.push({ x: point.x, y: point.y, life: 0, maxLife: .16 });
      redraw();
    }
    return;
  }

  if (addStrokePoint(point)) {
    emitParticles(point, lastSpeed, lastCurvature);
    if (Math.random() < .36) flashes.push({ x: point.x, y: point.y, life: 0, maxLife: .17 });
    redraw();
  }
}

function handleGrab(point) {
  if (!point) return;

  if (!previousGrabPoint) {
    previousGrabPoint = { ...point };
    return;
  }

  const dx = point.x - previousGrabPoint.x;
  const dy = point.y - previousGrabPoint.y;

  if (Math.hypot(dx, dy) >= 1.1) {
    if (grabbedStrokeIndex >= 0) {
      strokeOffsets[grabbedStrokeIndex].x += dx;
      strokeOffsets[grabbedStrokeIndex].y += dy;
    } else if (canvasPanning) {
      panX += dx;
      panY += dy;
    }

    previousGrabPoint = { ...point };
    redraw();
  }
}

function handleDelete(now) {
  if (deleteTarget < 0) return;

  const stillOver = findHoveredStroke(activePoints, SETTINGS.collisionRadius * 1.2);

  if (stillOver !== deleteTarget) {
    deleteTarget = stillOver;
    deleteStartedAt = stillOver >= 0 ? now : 0;
    return;
  }

  if (!deleteConsumed && now - deleteStartedAt >= SETTINGS.deleteHoldMs) {
    deleteStroke(deleteTarget);
    deleteConsumed = true;
  }
}

function updateInteraction(now) {
  if (latestHand) {
    activePoints = latestHand.map(p => displayPoint(p.x, p.y));

    const tip = latestHand[8];
    const point = naturalFilter(displayPoint(tip.x, tip.y), now, stableGesture === "GRAB" ? "GRAB" : "DRAW");

    if (stableGesture === "DRAW") handleDrawing(point);
    else if (stableGesture === "GRAB") handleGrab(point);
  } else {
    activePoints = [];
    predictPoint(now);
  }

  if (latestHand) {
    commitGesture(classifyGesture(latestHand), now);
  } else if (now - lastSampleAt > SETTINGS.lostGraceMs) {
    stableGesture = "IDLE";
    gestureCandidate = "IDLE";
  }

  const radius = hoveredStrokeIndex >= 0 ? SETTINGS.hoverExitRadius : SETTINGS.collisionRadius;
  hoveredStrokeIndex = findHoveredStroke(activePoints, radius);

  if (stableGesture === "DELETE") handleDelete(now);

  const labels = {
    IDLE: "READY",
    TRACK: "TRACKING",
    DRAW: "DRAWING",
    GRAB: grabbedStrokeIndex >= 0 ? "GRAB · OBJECT" : "GRAB · CANVAS",
    DELETE: "DELETE TARGET",
  };

  status.textContent = running ? labels[stableGesture] : "CAMERA OFF";
  motionLabel.textContent = running
    ? "MOTION MODEL · speed " + Math.round(lastSpeed) + " · curvature " + lastCurvature.toFixed(2)
    : "MOTION MODEL · READY";
}

function processVideo(now) {
  if (!running || !landmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime || now - lastDetectAt < SETTINGS.detectIntervalMs) return;

  lastVideoTime = video.currentTime;
  lastDetectAt = now;

  try {
    const result = landmarker.detectForVideo(video, now);
    latestHand = result.landmarks?.[0] || null;
  } catch (error) {
    console.warn("Hand detection failed", error);
    latestHand = null;
  }
}

function frame(now) {
  const dt = Math.min(.05, Math.max(.001, (now - lastFrameAt) / 1000));
  lastFrameAt = now;
  processVideo(now);
  updateInteraction(now);
  drawEffects(dt);
  requestAnimationFrame(frame);
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  try {
    return await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: .58,
      minHandPresenceConfidence: .58,
      minTrackingConfidence: .55,
    });
  } catch (gpuError) {
    console.warn("GPU delegate unavailable; using CPU.", gpuError);
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: .58,
      minHandPresenceConfidence: .55,
      minTrackingConfidence: .55,
    });
  }
}

async function start() {
  startButton.disabled = true;
  startError.textContent = "";

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access requires HTTPS or localhost.");
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 60 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    landmarker = await createLandmarker();
    running = true;
    startScreen.classList.add("hidden");
    resizeCanvas();
  } catch (error) {
    console.error(error);
    startError.textContent = error?.message || "Unable to access the camera or hand model.";
    startButton.disabled = false;

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
  }
}

function clearAll() {
  strokes = [];
  strokeOffsets = [];
  strokeBounds = [];
  currentStroke = null;
  hoveredStrokeIndex = -1;
  grabbedStrokeIndex = -1;
  deleteTarget = -1;
  panX = 0;
  panY = 0;
  redraw();
}

function savePng() {
  const { width, height } = size();
  const out = document.createElement("canvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  out.width = Math.round(width * dpr);
  out.height = Math.round(height * dpr);

  const ctx = out.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < strokes.length; i++) {
    drawStroke(ctx, strokes[i], i, false);
  }

  const link = document.createElement("a");
  link.download = "tuff-gesture-canvas.png";
  link.href = out.toDataURL("image/png");
  link.click();
}

startButton.addEventListener("click", start);
window.addEventListener("resize", resizeCanvas);
window.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();
  if (key === "c") clearAll();
  if (key === "s") savePng();
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});

resizeCanvas();
requestAnimationFrame(frame);
