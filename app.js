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

  // === 15-frame hysteresis state machine ===
  gestureWindowSize: 15,
  gestureEnterRatio: .85,
  gestureMaintainRatio: .30,
  gestureMinFrames: 15,
  sideOnEnterRatio: .92,

  // === World Landmark 1€ filter ===
  // minCutoff 越低越穩、延遲越高；beta 越高高速時越靈敏。
  worldFilterMinCutoff: .30,
  worldFilterBeta: .007,
  worldFilterDerivativeCutoff: 1.0,

  // Image landmarks only drive the visible cursor/skeleton.
  imageFilterMinCutoff: 1.10,
  imageFilterBeta: .025,
  imageFilterDerivativeCutoff: 1.0,

  // === Dynamic scale-invariant geometry ===
  pinchMaxPalmRatio: .25,
  pinchSideOnPalmRatio: .18,
  pinchMinOrientation: .28,

  // Normalized by PalmScale per second. Only blocks DRAW entry.
  drawMaxStartPalmSpeeds: 3.2,

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
let lastTrackerFrameAt = 0;
let lastHandSeenAt = 0;
let videoFrameCallbackId = 0;
let latestHand = null;
let latestWorldHand = null;
let activePoints = [];

class OneEuroFilter {
  constructor(minCutoff = .30, beta = .007, derivativeCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.prevRaw = null;
    this.prevFiltered = null;
    this.prevDerivative = 0;
    this.prevTime = 0;
  }

  reset() {
    this.prevRaw = null;
    this.prevFiltered = null;
    this.prevDerivative = 0;
    this.prevTime = 0;
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * Math.max(.001, cutoff));
    return 1 / (1 + tau / Math.max(.001, dt));
  }

  filter(value, nowMs) {
    if (this.prevRaw == null || this.prevFiltered == null) {
      this.prevRaw = value;
      this.prevFiltered = value;
      this.prevTime = nowMs;
      return value;
    }

    const dt = Math.max(.008, Math.min(.12, (nowMs - this.prevTime) / 1000));
    const rawDerivative = (value - this.prevRaw) / dt;
    const dAlpha = this.alpha(this.derivativeCutoff, dt);
    this.prevDerivative =
      dAlpha * rawDerivative + (1 - dAlpha) * this.prevDerivative;

    const cutoff = this.minCutoff + this.beta * Math.abs(this.prevDerivative);
    const a = this.alpha(cutoff, dt);
    const filtered = a * value + (1 - a) * this.prevFiltered;

    this.prevRaw = value;
    this.prevFiltered = filtered;
    this.prevTime = nowMs;
    return filtered;
  }
}

function createLandmarkFilterBank(minCutoff, beta, derivativeCutoff) {
  return Array.from({ length: 21 }, () => ({
    x: new OneEuroFilter(minCutoff, beta, derivativeCutoff),
    y: new OneEuroFilter(minCutoff, beta, derivativeCutoff),
    z: new OneEuroFilter(minCutoff, beta, derivativeCutoff),
  }));
}

const imageLandmarkFilters = createLandmarkFilterBank(
  SETTINGS.imageFilterMinCutoff,
  SETTINGS.imageFilterBeta,
  SETTINGS.imageFilterDerivativeCutoff
);

const worldLandmarkFilters = createLandmarkFilterBank(
  SETTINGS.worldFilterMinCutoff,
  SETTINGS.worldFilterBeta,
  SETTINGS.worldFilterDerivativeCutoff
);

function resetLandmarkFilters() {
  for (const bank of [imageLandmarkFilters, worldLandmarkFilters]) {
    for (const filter of bank) {
      filter.x.reset();
      filter.y.reset();
      filter.z.reset();
    }
  }
}

function smoothLandmarks(points, nowMs, filterBank) {
  if (!points || points.length !== 21) return null;

  return points.map((point, index) => ({
    x: filterBank[index].x.filter(point.x, nowMs),
    y: filterBank[index].y.filter(point.y, nowMs),
    z: point.z == null
      ? 0
      : filterBank[index].z.filter(point.z, nowMs),
  }));
}

let currentState = "IDLE";

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
  const dz = (a.z != null && b.z != null) ? a.z - b.z : 0;
  return Math.hypot(a.x - b.x, a.y - b.y, dz);
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


function sub3(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
}

function add3(a, b) {
  return { x: a.x + b.x, y: a.y + b.y, z: (a.z || 0) + (b.z || 0) };
}

function mul3(v, scalar) {
  return { x: v.x * scalar, y: v.y * scalar, z: (v.z || 0) * scalar };
}

function dot3(a, b) {
  return a.x * b.x + a.y * b.y + (a.z || 0) * (b.z || 0);
}

function cross3(a, b) {
  return {
    x: a.y * (b.z || 0) - (a.z || 0) * b.y,
    y: a.z * b.x - a.x * (b.z || 0),
    z: a.x * b.y - a.y * b.x,
  };
}

function norm3(v) {
  return Math.hypot(v.x, v.y, v.z || 0);
}

function normalize3(v) {
  const length = norm3(v);
  return length > 1e-6 ? mul3(v, 1 / length) : { x: 0, y: 0, z: 0 };
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

function palmScale(hand) {
  return Math.max(.0001, dist(hand[0], hand[9]));
}

function palmNormal(hand) {
  const a = sub3(hand[5], hand[0]);
  const b = sub3(hand[17], hand[0]);
  return normalize3(cross3(a, b));
}

function palmOrientationScore(hand) {
  if (!hand || hand.length !== 21) return 0;
  const normal = palmNormal(hand);
  if (!norm3(normal)) return 0;
  return clamp01(Math.abs(dot3(normal, { x: 0, y: 0, z: 1 })));
}

function isSideOnHand(hand) {
  return palmOrientationScore(hand) < .45;
}

function angleBendScore(angle, openAngle = 170, bendAngle = 105) {
  return clamp01((openAngle - angle) / (openAngle - bendAngle));
}

function fingerFoldbackScore(hand, mcp, pip, dip, tip) {
  const scale = palmScale(hand);
  const axis = normalize3(sub3(hand[pip], hand[mcp]));
  const tipFromPip = sub3(hand[tip], hand[pip]);

  // 用手指自己的軸，而不是固定 camera-Y，判斷指尖是否折回 PIP。
  const foldback = dot3(tipFromPip, axis);
  const foldbackScore = clamp01(
    (-foldback) / (scale * .20)
  );

  const pipAngle = jointAngle(hand[mcp], hand[pip], hand[dip]);
  const dipAngle = jointAngle(hand[pip], hand[dip], hand[tip]);

  const angleScore =
    angleBendScore(pipAngle, 170, 118) * .55 +
    angleBendScore(dipAngle, 174, 120) * .45;

  return clamp01(foldbackScore * .42 + angleScore * .58);
}

function pinchConfidence(hand) {
  const scale = palmScale(hand);
  const ratio = dist(hand[4], hand[8]) / scale;
  const sideOn = isSideOnHand(hand);
  const ratioLimit = sideOn
    ? SETTINGS.pinchSideOnPalmRatio
    : SETTINGS.pinchMaxPalmRatio;

  // 完全動態比例：4↔8 距離只相對於 0↔9 PalmScale。
  const tipScore = clamp01(
    (ratioLimit - ratio) / Math.max(.03, ratioLimit * .58)
  );

  const thumbBend = angleBendScore(
    jointAngle(hand[2], hand[3], hand[4]), 165, 112
  );
  const indexBend = angleBendScore(
    jointAngle(hand[5], hand[6], hand[7]), 175, 112
  );
  const indexDipBend = angleBendScore(
    jointAngle(hand[6], hand[7], hand[8]), 174, 118
  );

  const middleFolded = fingerFoldbackScore(hand, 9, 10, 11, 12);
  const ringFolded = fingerFoldbackScore(hand, 13, 14, 15, 16);
  const pinkyFolded = fingerFoldbackScore(hand, 17, 18, 19, 20);

  const otherFingers = Math.min(middleFolded, ringFolded, pinkyFolded);
  const orientation = palmOrientationScore(hand);

  const orientationFactor =
    orientation < SETTINGS.pinchMinOrientation
      ? .44
      : .70 + .30 * clamp01(
          (orientation - SETTINGS.pinchMinOrientation) /
          (.45 - SETTINGS.pinchMinOrientation)
        );

  if (
    tipScore <= 0 ||
    thumbBend < .08 ||
    indexBend < .10 ||
    otherFingers < .50
  ) {
    return 0;
  }

  const chainScore =
    thumbBend * .25 +
    indexBend * .24 +
    indexDipBend * .16 +
    otherFingers * .20 +
    orientationFactor * .15;

  return clamp01(tipScore * chainScore * orientationFactor);
}

function classifyGesture(hand, motionSpeed = indexWorldSpeed) {
  const f = fingerFlags(hand);
  const pinch = pinchConfidence(hand);
  const sideOn = isSideOnHand(hand);

  const otherExtendedForIndex = Math.max(
    f.middle.extended,
    f.ring.extended,
    f.pinky.extended
  );

  const otherExtendedForPinky = Math.max(
    f.index.extended,
    f.middle.extended,
    f.ring.extended
  );

  const indexFolded = fingerFoldbackScore(hand, 5, 6, 7, 8);
  const middleFolded = fingerFoldbackScore(hand, 9, 10, 11, 12);
  const ringFolded = fingerFoldbackScore(hand, 13, 14, 15, 16);
  const pinkyFolded = fingerFoldbackScore(hand, 17, 18, 19, 20);

  const drawQualified =
    f.index.extended >= .72 &&
    f.index.extended - otherExtendedForIndex >= .12 &&
    middleFolded >= .58 &&
    ringFolded >= .58 &&
    pinkyFolded >= .58 &&
    pinch < .34;

  const deleteQualified =
    f.pinky.extended >= .72 &&
    f.pinky.extended - otherExtendedForPinky >= .14 &&
    indexFolded >= .62 &&
    middleFolded >= .62 &&
    ringFolded >= .62 &&
    f.thumb <= .52 &&
    pinch < .20;

  const trackScore = Math.min(
    f.thumb,
    f.index.extended,
    f.middle.extended,
    f.ring.extended,
    f.pinky.extended
  ) * (1 - pinch);

  // 高速移動時不允許「剛進入」DRAW；已經在 DRAW 中則保持可正常畫線。
  const drawMotionGate =
    motionSpeed <= SETTINGS.drawMaxStartPalmSpeeds ||
    currentState === "DRAW";

  const drawScore = drawQualified && drawMotionGate ? 1 : 0;
  const deleteScore = deleteQualified ? 1 : 0;

  const ranked = [
    { name: "GRAB", score: pinch },
    { name: "DRAW", score: drawScore },
    { name: "DELETE", score: deleteScore },
    { name: "TRACK", score: trackScore },
  ].sort((a, b) => b.score - a.score);

  const best = ranked[0];
  const second = ranked[1];
  const base = {
    score: best.score,
    margin: best.score - second.score,
    sideOn,
  };

  if (
    best.name === "GRAB" &&
    best.score >= .62 &&
    best.margin >= .13
  ) {
    return { name: "GRAB", ...base };
  }

  if (best.score < .56) {
    return { name: "IDLE", ...base };
  }

  if (best.margin < .13) {
    return { name: "IDLE", ...base };
  }

  if (best.name === "DRAW" && !drawQualified) {
    return { name: "IDLE", ...base };
  }

  if (best.name === "DELETE" && !deleteQualified) {
    return { name: "IDLE", ...base };
  }

  return { name: best.name, ...base };
}
let gestureWindow = [];

function windowRatio(target, requireFullWindow = false) {
  if (!gestureWindow.length) return 0;
  if (requireFullWindow && gestureWindow.length < SETTINGS.gestureMinFrames) return 0;

  let matched = 0;
  for (const sample of gestureWindow) {
    if (
      sample.name === target &&
      sample.score >= .60 &&
      sample.margin >= .10
    ) {
      matched++;
    }
  }

  return matched / gestureWindow.length;
}

function onStart(next) {
  if (next === "DRAW") {
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;
    deleteTarget = -1;
    deleteStartedAt = 0;
    deleteConsumed = false;
    currentStroke = null;

    const indexPoint = activePoints[8];
    resetMotionFilter();
    filteredPoint = indexPoint ? { ...indexPoint } : null;
    drawAnchor = indexPoint ? { ...indexPoint } : null;
  }

  if (next === "GRAB") {
    const pinchPoint = latestHand
      ? midpoint(activePoints[4], activePoints[8])
      : filteredPoint;

    const grabPoints = latestHand
      ? [pinchPoint, activePoints[4], activePoints[8]].filter(Boolean)
      : [];

    grabbedStrokeIndex = findHoveredStroke(
      grabPoints,
      SETTINGS.collisionRadius * 1.55
    );

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

    const pinkyPoint = activePoints[20];
    deleteTarget = pinkyPoint
      ? findHoveredStroke(
          [pinkyPoint],
          SETTINGS.collisionRadius * 1.2
        )
      : -1;

    deleteStartedAt = deleteTarget >= 0 ? performance.now() : 0;
    deleteConsumed = false;
    resetMotionFilter();
  }
}

function onEnd(prev) {
  if (prev === "DRAW") stopStroke();

  if (prev === "GRAB") {
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    canvasPanning = false;
  }

  if (prev === "DELETE") {
    deleteTarget = -1;
    deleteStartedAt = 0;
    deleteConsumed = false;
  }
}

function updateGestureState(result, now) {
  gestureWindow.push({
    name: result.name,
    score: result.score,
    margin: result.margin,
    sideOn: Boolean(result.sideOn),
    t: now,
  });

  while (gestureWindow.length > SETTINGS.gestureWindowSize) {
    gestureWindow.shift();
  }

  if (currentState === "IDLE") {
    if (gestureWindow.length < SETTINGS.gestureMinFrames) return;
    if (result.name === "IDLE") return;

    const entryRatio = result.sideOn
      ? SETTINGS.sideOnEnterRatio
      : SETTINGS.gestureEnterRatio;

    if (windowRatio(result.name, true) >= entryRatio) {
      currentState = result.name;
      onStart(currentState);
      // 保留已通過的 15-frame 證據，讓下一幀不會因窗口重置而誤釋放。
    }

    return;
  }

  const maintainRatio = windowRatio(currentState, false);

  // Seamless handoff: if a competing gesture has already satisfied the full
  // entry window and the current gesture is below its 30% maintain threshold,
  // end the old state and start the new one in the same frame.
  if (
    result.name !== currentState &&
    result.name !== "IDLE" &&
    maintainRatio < SETTINGS.gestureMaintainRatio
  ) {
    const entryRatio = result.sideOn
      ? SETTINGS.sideOnEnterRatio
      : SETTINGS.gestureEnterRatio;

    if (windowRatio(result.name, true) >= entryRatio) {
      const ended = currentState;
      onEnd(ended);
      currentState = result.name;
      onStart(currentState);
      gestureWindow = [{
        name: result.name,
        score: result.score,
        margin: result.margin,
        sideOn: Boolean(result.sideOn),
        t: now,
      }];
      return;
    }
  }

  // Schmitt-style maintain gate: once active, only <30% releases it.
  if (maintainRatio < SETTINGS.gestureMaintainRatio) {
    const ended = currentState;
    onEnd(ended);
    currentState = "IDLE";
    gestureWindow = [];
  }
}

let indexWorldSpeed = 0;
let indexWorldPoint = null;
let indexWorldAt = 0;

function resetMotionFilter() {
  rawPoint = null;
  filteredPoint = null;
  velocity = { x: 0, y: 0 };
  lastSampleAt = 0;
  motionSamples = [];
  lastCurvature = 0;
  lastSpeed = 0;
  indexWorldSpeed = 0;
  indexWorldPoint = null;
  indexWorldAt = 0;
}

function updateIndexWorldSpeed(hand, now) {
  const point = hand?.[8];
  if (!point) {
    indexWorldSpeed = 0;
    indexWorldPoint = null;
    indexWorldAt = 0;
    return 0;
  }

  if (!indexWorldPoint) {
    indexWorldPoint = { ...point };
    indexWorldAt = now;
    indexWorldSpeed = 0;
    return 0;
  }

  const dt = Math.max(.01, Math.min(.12, (now - indexWorldAt) / 1000));
  const scale = palmScale(hand);
  const raw = dist(point, indexWorldPoint) / Math.max(.0001, scale * dt);

  indexWorldSpeed = indexWorldSpeed * .72 + raw * .28;
  indexWorldPoint = { ...point };
  indexWorldAt = now;
  return indexWorldSpeed;
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
  if (currentState !== "DELETE" || deleteTarget < 0 || !deleteStartedAt || deleteConsumed) return;

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

  const pinkyPoint = activePoints[20];
  const stillOver = pinkyPoint
    ? findHoveredStroke([pinkyPoint], SETTINGS.collisionRadius * 1.22)
    : -1;

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
    updateIndexSpeed(indexPoint, now);
    const pinchPoint = midpoint(activePoints[4], activePoints[8]);

    if (currentState === "GRAB") {
      point = naturalFilter(pinchPoint, now, "GRAB");
      handleGrab(point);
    } else if (currentState === "DRAW") {
      point = naturalFilter(indexPoint, now, "DRAW");
      handleDrawing(point);
    }
  } else {
    activePoints = [];
    predictPoint(now);
  }

  if (latestHand) {
    const gestureResult = latestWorldHand
      ? classifyGesture(latestWorldHand, indexWorldSpeed)
      : { name: "IDLE", score: 0, margin: 0, sideOn: true };

    updateGestureState(gestureResult, now);
  } else if (now - lastSampleAt > SETTINGS.lostGraceMs) {
    const ended = currentState;
    currentState = "IDLE";
    gestureWindow = [];
    if (ended !== "IDLE") onEnd(ended);
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    deleteTarget = -1;
    resetLandmarkFilters();
    resetMotionFilter();
  }

  const hoverPoints =
    currentState === "GRAB"
      ? [activePoints[4], activePoints[8], midpoint(activePoints[4], activePoints[8])].filter(Boolean)
      : currentState === "DELETE"
        ? [activePoints[20]].filter(Boolean)
        : activePoints;

  const radius = hoveredStrokeIndex >= 0
    ? SETTINGS.hoverExitRadius
    : SETTINGS.collisionRadius;

  hoveredStrokeIndex = findHoveredStroke(hoverPoints, radius);

  if (currentState === "DELETE") {
    // Pinky mode locks to one hovered stroke. Never call clearAll here.
    if (deleteTarget < 0 && hoveredStrokeIndex >= 0) {
      deleteTarget = hoveredStrokeIndex;
      deleteStartedAt = now;
    }
    handleDelete(now);
  }

  const labels = {
    IDLE: latestHand ? "READY" : "NO HAND · CAMERA LIVE",
    TRACK: "TRACKING",
    DRAW: "DRAWING",
    GRAB: grabbedStrokeIndex >= 0 ? "PINCH · MOVE OBJECT" : "PINCH · MOVE CANVAS",
    DELETE: deleteTarget >= 0 ? "PINKY · SELECT / DELETE" : "PINKY · TARGET A STROKE",
  };

  if (!running) {
    status.textContent = "CAMERA OFF";
  } else if (currentState !== "IDLE") {
    status.textContent = labels[currentState];
  } else if (lastTrackerFrameAt && now - lastTrackerFrameAt > 700) {
    status.textContent = "TRACKER WAITING";
  } else if (!lastHandSeenAt || now - lastHandSeenAt > 450) {
    status.textContent = "NO HAND · TRACKER LIVE";
  } else {
    status.textContent = labels[currentState];
  }
  motionLabel.textContent = running
    ? "HAND MOTION · 3D speed " + indexWorldSpeed.toFixed(2) + " · curvature " + lastCurvature.toFixed(2)
    : "HAND MOTION · READY";
}

function processVideo(now, mediaTimeMs = null) {
  if (!running || !landmarker || video.readyState < 2) return;

  const videoTime = video.currentTime;
  if (videoTime === lastVideoTime && now - lastDetectAt < SETTINGS.detectIntervalMs) return;

  lastVideoTime = videoTime;
  lastDetectAt = now;
  lastTrackerFrameAt = now;

  try {
    // requestVideoFrameCallback 的 mediaTime 是媒體時間；用它做 timestamp
    // 可避免 RAF / 實際 camera frame 不同步造成的推論節奏問題。
    const timestamp = Number.isFinite(mediaTimeMs)
      ? mediaTimeMs
      : now;

    const result = landmarker.detectForVideo(video, timestamp);
    const rawImageHand = result.landmarks?.[0] || null;
    const rawWorldHand = result.worldLandmarks?.[0] || null;

    if (rawImageHand) {
      lastHandSeenAt = now;
      latestHand = smoothLandmarks(rawImageHand, now, imageLandmarkFilters);
      latestWorldHand = rawWorldHand
        ? smoothLandmarks(rawWorldHand, now, worldLandmarkFilters)
        : null;

      if (latestWorldHand) {
        updateIndexWorldSpeed(latestWorldHand, now);
      } else {
        updateIndexWorldSpeed(null, now);
      }
    } else {
      latestHand = null;
      latestWorldHand = null;
      updateIndexWorldSpeed(null, now);
      resetLandmarkFilters();
    }
  } catch (error) {
    console.warn("Hand detection failed", error);
    latestHand = null;
    latestWorldHand = null;
    resetLandmarkFilters();
  }
}


let fallbackDetectionTimer = 0;

function startVideoDetection() {
  if (video.requestVideoFrameCallback) {
    scheduleVideoDetection();
    return;
  }

  const tick = () => {
    if (!running) return;
    const now = performance.now();
    processVideo(now, now);
    fallbackDetectionTimer = window.setTimeout(tick, SETTINGS.detectIntervalMs);
  };

  tick();
}

function scheduleVideoDetection() {
  if (!running || !videoFrameCallbackId && !video.requestVideoFrameCallback) return;

  if (video.requestVideoFrameCallback) {
    videoFrameCallbackId = video.requestVideoFrameCallback((now, metadata) => {
      if (running) {
        processVideo(now, metadata.mediaTime * 1000);
        scheduleVideoDetection();
      }
    });
  }
}

function frame(now) {
  const dt = Math.min(.05, Math.max(.001, (now - lastFrameAt) / 1000));
  lastFrameAt = now;

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
    resetLandmarkFilters();
    startVideoDetection();
    resizeCanvas();
  } catch (error) {
    console.error(error);
    running = false;
    startScreen.classList.remove("hidden");
    startError.textContent = error?.message || "Unable to access the camera or hand model.";
    startButton.disabled = false;

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
  }
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
  if (key === "s") savePng();
});
window.addEventListener("beforeunload", () => {
  if (videoFrameCallbackId && video.cancelVideoFrameCallback) {
    video.cancelVideoFrameCallback(videoFrameCallbackId);
  }
  if (fallbackDetectionTimer) {
    clearTimeout(fallbackDetectionTimer);
  }
  if (stream) stream.getTracks().forEach(track => track.stop());
});

resizeCanvas();
requestAnimationFrame(frame);
