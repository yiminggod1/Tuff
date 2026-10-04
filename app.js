import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const COLORS = {
  draw: "#FF4F91",
  hover: "#D8FF5A",
  hand: "#29E8D4",
  handPoint: "#85FFF3",
  white: "#FFFFFF",
  delete: "#FF6F91",
};

const SETTINGS = {
  detectIntervalMs: 26,
  lostGraceMs: 110,
  gestureHoldMs: 62,
  pinchStart: 0.47,
  pinchRelease: 0.62,
  collisionRadius: 26,
  hoverExitRadius: 42,
  deleteHoldMs: 520,
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

function midpoint(a, b) {
  return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
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

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function fingerGeometry(hand, mcp, pip, dip, tip) {
  const palm = Math.max(.0001, dist(hand[0], hand[9]));
  const direct = dist(hand[mcp], hand[tip]);
  const chain =
    dist(hand[mcp], hand[pip]) +
    dist(hand[pip], hand[dip]) +
    dist(hand[dip], hand[tip]);

  const straightness = chain ? direct / chain : 0;
  const wristReach = dist(hand[0], hand[tip]) / palm;
  const pipA = jointAngle(hand[mcp], hand[pip], hand[dip]);
  const dipA = jointAngle(hand[pip], hand[dip], hand[tip]);

  const pipExtended = clamp01((pipA - 138) / 34);
  const dipExtended = clamp01((dipA - 140) / 32);
  const reachExtended = clamp01((wristReach - .95) / .78);
  const straightExtended = clamp01((straightness - .76) / .18);

  const extended = (
    pipExtended * .34 +
    dipExtended * .28 +
    reachExtended * .22 +
    straightExtended * .16
  );

  const pipCurled = clamp01((146 - pipA) / 48);
  const dipCurled = clamp01((150 - dipA) / 45);
  const reachCurled = clamp01((1.55 - wristReach) / .70);

  const curled = (
    pipCurled * .43 +
    dipCurled * .32 +
    reachCurled * .25
  );

  return {
    extended: clamp01(extended),
    curled: clamp01(curled),
    pip: pipA,
    dip: dipA,
    reach: wristReach,
  };
}

function fingerFlags(hand) {
  const fingers = {
    index: fingerGeometry(hand, 5, 6, 7, 8),
    middle: fingerGeometry(hand, 9, 10, 11, 12),
    ring: fingerGeometry(hand, 13, 14, 15, 16),
    pinky: fingerGeometry(hand, 17, 18, 19, 20),
  };

  const palm = Math.max(.0001, dist(hand[0], hand[9]));
  const thumbReach = dist(hand[0], hand[4]) / palm;
  const thumbOpen = clamp01((thumbReach - .72) / .62);

  return {
    thumb: thumbOpen,
    ...fingers,
  };
}

function pinchRatio(hand) {
  return dist(hand[4], hand[8]) / Math.max(.0001, dist(hand[0], hand[9]));
}

function pinchConfidence(hand) {
  const ratio = pinchRatio(hand);
  return clamp01((.62 - ratio) / .24);
}

function classifyGesture(hand) {
  const f = fingerFlags(hand);
  const pinch = pinchConfidence(hand);

  const drawScore =
    f.index.extended *
    f.middle.curled *
    f.ring.curled *
    f.pinky.curled *
    (1 - pinch * .9);

  const deleteScore =
    f.pinky.extended *
    f.index.curled *
    f.middle.curled *
    f.ring.curled *
    (1 - pinch * .9);

  const trackScore =
    Math.min(
      f.thumb,
      f.index.extended,
      f.middle.extended,
      f.ring.extended,
      f.pinky.extended
    ) *
    (1 - pinch);

  const grabScore = pinch;

  const ranked = [
    { name: "GRAB", score: grabScore },
    { name: "DRAW", score: drawScore },
    { name: "DELETE", score: deleteScore },
    { name: "TRACK", score: trackScore },
  ].sort((a, b) => b.score - a.score);

  const best = ranked[0];
  const second = ranked[1];

  if (best.score < .58) {
    return { name: "IDLE", score: best.score, margin: best.score };
  }

  if (best.score - second.score < .11) {
    return { name: "IDLE", score: best.score, margin: best.score - second.score };
  }

  return {
    name: best.name,
    score: best.score,
    margin: best.score - second.score,
  };
}

function commitGesture(result, now) {
  const next = result.name;
  const confidence = result.score;

  if (next === stableGesture) {
    gestureCandidate = next;
    gestureCandidateAt = now;
    return;
  }

  if (next !== gestureCandidate) {
    gestureCandidate = next;
    gestureCandidateAt = now;
    return;
  }

  const requiredHold =
    next === "GRAB" && confidence > .82 ? 34 :
    confidence > .80 ? 42 :
    confidence > .70 ? 62 :
    95;

  if (now - gestureCandidateAt >= requiredHold) {
    const previous = stableGesture;
    stableGesture = next;
    onGestureChanged(previous, next);
  }
}

function resetMotionFilter() {
  rawPoint = null;
  filteredPoint = null;
  velocity = { x: 0, y: 0 };
  lastSampleAt = 0;
  motionSamples = [];
  lastCurvature = 0;
  lastSpeed = 0;
}

function onGestureChanged(prev, next) {
  if (prev === "DRAW" && next !== "DRAW") stopStroke();

  if (prev === "GRAB" && next !== "GRAB") {
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;
  }

  if (next === "DRAW") {
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;
    deleteTarget = -1;
    deleteStartedAt = 0;
    deleteConsumed = false;
    drawAnchor = filteredPoint ? { ...filteredPoint } : null;
    currentStroke = null;
  }

  if (next === "GRAB") {
    const pinchPoint = latestHand
      ? midpoint(activePoints[4], activePoints[8])
      : filteredPoint;

    const grabPoints = latestHand
      ? [pinchPoint, activePoints[4], activePoints[8]].filter(Boolean)
      : [];

    grabbedStrokeIndex = findHoveredStroke(grabPoints, SETTINGS.collisionRadius * 1.55);
    canvasPanning = grabbedStrokeIndex < 0;
    previousGrabPoint = pinchPoint ? { ...pinchPoint } : null;

    deleteTarget = -1;
    deleteStartedAt = 0;
    deleteConsumed = false;
    resetMotionFilter();
    if (pinchPoint) filteredPoint = { ...pinchPoint };
  }

  if (next === "DELETE") {
    stopStroke();
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;

    // Only one target can ever be selected. Nothing is deleted when the
    // pinky is away from an existing stroke.
    deleteTarget = findHoveredStroke(activePoints, SETTINGS.collisionRadius * 1.2);
    deleteStartedAt = deleteTarget >= 0 ? performance.now() : 0;
    deleteConsumed = false;
    resetMotionFilter();
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
  lastSpeed = speed;
  lastCurvature = curvature;

  return { speed, curvature };
}

function naturalFilter(point, now, mode) {
  if (!filteredPoint) {
    filteredPoint = { ...point };
    velocity = { x: 0, y: 0 };
    estimateKinematics(point, now);
    return { ...point };
  }

  const beforeSample = { ...filteredPoint };
  const dt = Math.max(.008, Math.min(.06, (now - lastSampleAt) / 1000 || .016));
  const motion = estimateKinematics(point, now);
  const speedN = Math.min(1, motion.speed / 1150);

  let alpha = .16 + .70 * Math.pow(speedN, .58);
  if (mode === "GRAB") alpha += .16;

  const curvatureBrake = 1 / (1 + 3.8 * Math.min(1, motion.curvature * 18));
  alpha *= .76 + .24 * curvatureBrake;
  alpha = Math.max(mode === "GRAB" ? .70 : .12, Math.min(mode === "GRAB" ? .98 : .86, alpha));

  const next = {
    x: filteredPoint.x + (point.x - filteredPoint.x) * alpha,
    y: filteredPoint.y + (point.y - filteredPoint.y) * alpha,
  };

  velocity = {
    x: (next.x - beforeSample.x) / dt,
    y: (next.y - beforeSample.y) / dt,
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

  const t = Math.max(
    0,
    Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / denom)
  );

  return Math.hypot(
    p.x - (a.x + abx * t),
    p.y - (a.y + aby * t)
  );
}

function findHoveredStroke(points, radius = SETTINGS.collisionRadius) {
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
        const d = distanceToSegment(
          p,
          strokePoint(i, j - 1),
          strokePoint(i, j)
        );

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

  // Delete one and only one indexed stroke. No global clear occurs here.
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
    ctx.arc(
      stroke[0].x,
      stroke[0].y,
      SETTINGS.brushWidth * .55,
      0,
      Math.PI * 2
    );
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

  ctx.lineWidth = highlight
    ? SETTINGS.brushWidth + 1.15
    : SETTINGS.brushWidth;

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
    drawStroke(drawCtx, strokes[i], i, i === hoveredStrokeIndex || i === deleteTarget);
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

function drawDeletePreview() {
  if (stableGesture !== "DELETE" || deleteTarget < 0 || !deleteStartedAt || deleteConsumed) return;

  const b = strokeBounds[deleteTarget];
  if (!b) return;

  const progress = Math.max(
    0,
    Math.min(1, (performance.now() - deleteStartedAt) / SETTINGS.deleteHoldMs)
  );

  const offset = strokeOffsets[deleteTarget] || { x: 0, y: 0 };
  const cx = (b.minX + b.maxX) / 2 + offset.x + panX;
  const cy = (b.minY + b.maxY) / 2 + offset.y + panY;
  const radius = Math.max(28, Math.hypot(b.maxX - b.minX, b.maxY - b.minY) * .18);

  fxCtx.save();
  fxCtx.strokeStyle = COLORS.delete;
  fxCtx.shadowColor = COLORS.delete;
  fxCtx.shadowBlur = 14;
  fxCtx.lineWidth = 2.2;
  fxCtx.beginPath();
  fxCtx.arc(
    cx,
    cy,
    radius + 9,
    -Math.PI / 2,
    -Math.PI / 2 + progress * Math.PI * 2
  );
  fxCtx.stroke();
  fxCtx.restore();
}

function emitParticles(point, speed, curvature) {
  const turnBrake = 1 / (1 + curvature * 20);
  const count = Math.min(
    5,
    Math.max(1, Math.round(1 + speed / 620) * (turnBrake > .55 ? 1 : .55))
  );

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
  drawDeletePreview();
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

    if (Math.random() < .36) {
      flashes.push({ x: point.x, y: point.y, life: 0, maxLife: .17 });
    }

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

  if (Math.hypot(dx, dy) >= 1.0) {
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

  const stillOver = findHoveredStroke(activePoints, SETTINGS.collisionRadius * 1.22);

  if (stillOver !== deleteTarget) {
    deleteTarget = stillOver;
    deleteStartedAt = stillOver >= 0 ? now : 0;
    deleteConsumed = false;
    return;
  }

  if (!deleteConsumed && now - deleteStartedAt >= SETTINGS.deleteHoldMs) {
    deleteStroke(deleteTarget);
    deleteConsumed = true;
  }
}

function updateInteraction(now) {
  let point = null;

  if (latestHand) {
    activePoints = latestHand.map(p => displayPoint(p.x, p.y));

    const indexPoint = activePoints[8];
    const pinchPoint = midpoint(activePoints[4], activePoints[8]);

    if (stableGesture === "GRAB") {
      point = naturalFilter(pinchPoint, now, "GRAB");
      handleGrab(point);
    } else if (stableGesture === "DRAW") {
      point = naturalFilter(indexPoint, now, "DRAW");
      handleDrawing(point);
    }
  } else {
    activePoints = [];
    predictPoint(now);
  }

  if (latestHand) {
    commitGesture(classifyGesture(latestHand), now);
  } else if (now - lastSampleAt > SETTINGS.lostGraceMs) {
    stableGesture = "IDLE";
    gestureCandidate = "IDLE";
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    deleteTarget = -1;
  }

  const hoverPoints = stableGesture === "GRAB"
    ? [activePoints[4], activePoints[8], midpoint(activePoints[4], activePoints[8])].filter(Boolean)
    : activePoints;

  const radius = hoveredStrokeIndex >= 0
    ? SETTINGS.hoverExitRadius
    : SETTINGS.collisionRadius;

  hoveredStrokeIndex = findHoveredStroke(hoverPoints, radius);

  if (stableGesture === "DELETE") {
    // Pinky mode locks to one hovered stroke. Never call clearAll here.
    if (deleteTarget < 0 && hoveredStrokeIndex >= 0) {
      deleteTarget = hoveredStrokeIndex;
      deleteStartedAt = now;
    }
    handleDelete(now);
  }

  const labels = {
    IDLE: "READY",
    TRACK: "TRACKING",
    DRAW: "DRAWING",
    GRAB: grabbedStrokeIndex >= 0 ? "PINCH · MOVE OBJECT" : "PINCH · MOVE CANVAS",
    DELETE: deleteTarget >= 0 ? "PINKY · SELECT / DELETE" : "PINKY · TARGET A STROKE",
  };

  status.textContent = running ? labels[stableGesture] : "CAMERA OFF";
  motionLabel.textContent = running
    ? "HAND MOTION · speed " + Math.round(lastSpeed) + " · curvature " + lastCurvature.toFixed(2)
    : "HAND MOTION · READY";
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
