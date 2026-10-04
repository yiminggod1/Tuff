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
let latestWorldHand = null;
let activePoints = [];

const calibrationCanvas = document.querySelector("#calibrationCanvas");
const calibrationCtx = calibrationCanvas?.getContext("2d");
const calibrationOverlayCanvas = document.querySelector("#calibrationOverlayCanvas");
const calibrationOverlayCtx = calibrationOverlayCanvas?.getContext("2d");
const calibrationScreen = document.querySelector("#calibrationScreen");
const calibrationTitle = document.querySelector("#calibrationTitle");
const calibrationText = document.querySelector("#calibrationText");
const calibrationMeterFill = document.querySelector("#calibrationMeterFill");
const calibrationStep = document.querySelector("#calibrationStep");
const calibrationConfirm = document.querySelector("#calibrationConfirm");
const calibrationPoseTitle = document.querySelector("#calibrationPoseTitle");
const calibrationPose = document.querySelector("#calibrationPose");
const calibrationFeedback = document.querySelector("#calibrationFeedback");
const checkHand = document.querySelector("#checkHand");
const checkShape = document.querySelector("#checkShape");
const checkDepth = document.querySelector("#checkDepth");
const checkStability = document.querySelector("#checkStability");
const checkPersonal = document.querySelector("#checkPersonal");

let calibrationActive = false;
let calibrationIndex = 0;
let calibrationHoldStarted = 0;
let calibrationConfirmedUntil = 0;
let calibrationSamples = [];
let calibrationRecentPoses = [];
let calibrationRecentScales = [];
const calibrationHoldMs = 900;

const calibrationSteps = [
  {
    gesture: "TRACK",
    title: "Step 1 · Open your hand",
    text: "Show your whole hand to the live camera. Spread all five fingers naturally and keep the palm facing the camera.",
    guide: {
      thumb: "OPEN",
      index: "OPEN",
      middle: "OPEN",
      ring: "OPEN",
      pinky: "OPEN",
    },
  },
  {
    gesture: "GRAB",
    title: "Step 2 · Pinch to move",
    text: "Touch thumb tip to index tip. Keep the other three fingers folded. Hold the pinch still.",
    guide: {
      thumb: "PINCH",
      index: "PINCH",
      middle: "BENT",
      ring: "BENT",
      pinky: "BENT",
    },
  },
  {
    gesture: "DELETE",
    title: "Step 3 · Pinky selects",
    text: "Raise only your pinky. Fold index, middle and ring. Let the thumb stay relaxed. Hold without waving.",
    guide: {
      thumb: "RELAX",
      index: "BENT",
      middle: "BENT",
      ring: "BENT",
      pinky: "OPEN",
    },
  },
  {
    gesture: "DRAW",
    title: "Step 4 · Index draws",
    text: "Raise only your index. Fold middle, ring and pinky. Let the thumb stay relaxed. Hold without waving.",
    guide: {
      thumb: "RELAX",
      index: "OPEN",
      middle: "BENT",
      ring: "BENT",
      pinky: "BENT",
    },
  },
];

const calibrationProfile = Object.create(null);
let calibrationLastExpected = null;
let calibrationLastConfidence = 0;

function setupCalibrationPoseGuide(step) {
  if (!calibrationPose) return;

  const labels = [
    ["thumb", "THUMB"],
    ["index", "INDEX"],
    ["middle", "MIDDLE"],
    ["ring", "RING"],
    ["pinky", "PINKY"],
  ];

  calibrationPose.innerHTML = labels.map(([key, label]) => {
    const state = step.guide[key];
    const cls = state === "BENT" ? "bend" : state === "PINCH" ? "pinch" : "";
    return '<div class="finger-guide ' + cls + '">' +
      '<div class="finger-name">' + label + '</div>' +
      '<div class="finger-state">' + state + '</div>' +
    '</div>';
  }).join("");

  if (calibrationPoseTitle) {
    calibrationPoseTitle.textContent = "DO THIS · THEN HOLD STILL";
  }
}

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
  resizeCalibrationOverlay();
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

// Canonical 3D pose: wrist-centered, palm-axis aligned, scale-normalized.
// This makes the learned template substantially less sensitive to camera
// position, hand size, and in-plane rotation while keeping finger geometry.
function canonicalHand(hand) {
  if (!hand || hand.length !== 21 || hand.some(p => p.z == null)) return null;

  const origin = hand[0];
  const palmAcross = normalize3(sub3(hand[17], hand[5]));
  let palmForward = sub3(hand[9], origin);
  palmForward = sub3(palmForward, mul3(palmAcross, dot3(palmForward, palmAcross)));
  palmForward = normalize3(palmForward);

  let palmNormal = normalize3(cross3(palmAcross, palmForward));
  const scale = Math.max(.0001, dist(hand[5], hand[17]));

  if (!norm3(palmForward) || !norm3(palmNormal)) return null;

  const flat = [];
  for (const point of hand) {
    const rel = sub3(point, origin);
    flat.push(
      dot3(rel, palmAcross) / scale,
      dot3(rel, palmForward) / scale,
      dot3(rel, palmNormal) / scale
    );
  }
  return flat;
}

function poseRms(a, b) {
  if (!a || !b || a.length !== b.length) return 1;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return Math.sqrt(sum / a.length);
}

function poseSimilarity(hand, template) {
  const vector = canonicalHand(hand);
  if (!vector || !template) return 0;
  return clamp01(1 - poseRms(vector, template) / .24);
}

function averageVectors(vectors) {
  if (!vectors.length) return null;
  const out = new Array(vectors[0].length).fill(0);
  for (const vector of vectors) {
    for (let i = 0; i < out.length; i++) out[i] += vector[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= vectors.length;
  return out;
}

function handImageQuality(hand) {
  if (!hand || hand.length !== 21) return 0;

  let inside = 0;
  for (const point of hand) {
    if (point.x > .02 && point.x < .98 && point.y > .02 && point.y < .98) inside++;
  }

  const palmScale = dist(hand[5], hand[17]);
  const scaleScore =
    clamp01((palmScale - .05) / .045) *
    clamp01((.22 - palmScale) / .10);

  return clamp01((inside / 21) * .72 + scaleScore * .28);
}

function calibrationStabilityScore(hand) {
  const pose = canonicalHand(hand);
  if (!pose || !calibrationRecentPoses.length) return 0.58;

  const previous = calibrationRecentPoses[calibrationRecentPoses.length - 1];
  const jitter = poseRms(pose, previous);
  return clamp01(1 - jitter / .055);
}

function calibrationDepthScore(hand) {
  if (!hand || hand.length !== 21 || hand.some(p => p.z == null)) return 0;

  const zValues = hand.map(p => p.z);
  const zRange = Math.max(...zValues) - Math.min(...zValues);
  const palmScale = dist(hand[5], hand[17]);
  const depthStructure = clamp01(zRange / Math.max(.025, palmScale * .7));
  const scaleHealth =
    clamp01((palmScale - .045) / .04) *
    clamp01((.22 - palmScale) / .10);

  return clamp01(depthStructure * .72 + scaleHealth * .28);
}

function captureCalibrationSample(gesture, hand) {
  const pose = canonicalHand(hand);
  if (!pose) return;

  const f = fingerFlags(hand);
  calibrationSamples.push({
    pose,
    palmScale: dist(hand[5], hand[17]),
    pinchRatio: pinchRatio(hand),
    extended: {
      index: f.index.extended,
      middle: f.middle.extended,
      ring: f.ring.extended,
      pinky: f.pinky.extended,
      thumb: f.thumb,
    },
    curled: {
      index: f.index.curled,
      middle: f.middle.curled,
      ring: f.ring.curled,
      pinky: f.pinky.curled,
    },
    gesture,
  });

  if (calibrationSamples.length > 36) calibrationSamples.shift();
}

function learnCalibrationProfile(gesture) {
  if (!calibrationSamples.length) return;

  const pose = averageVectors(calibrationSamples.map(sample => sample.pose));
  const average = key => calibrationSamples.reduce((sum, sample) => sum + sample[key], 0) / calibrationSamples.length;

  const meanExtended = {};
  const meanCurled = {};
  for (const finger of ["thumb", "index", "middle", "ring", "pinky"]) {
    meanExtended[finger] = calibrationSamples.reduce(
      (sum, sample) => sum + sample.extended[finger], 0
    ) / calibrationSamples.length;
  }
  for (const finger of ["index", "middle", "ring", "pinky"]) {
    meanCurled[finger] = calibrationSamples.reduce(
      (sum, sample) => sum + sample.curled[finger], 0
    ) / calibrationSamples.length;
  }

  calibrationProfile[gesture] = {
    template: pose,
    palmScale: average("palmScale"),
    pinchRatio: average("pinchRatio"),
    extended: meanExtended,
    curled: meanCurled,
    sampleCount: calibrationSamples.length,
  };
}

function personalFeatureSimilarity(hand, profile) {
  if (!profile) return 0;

  const f = fingerFlags(hand);
  const values = {
    thumb: f.thumb,
    index: f.index.extended,
    middle: f.middle.extended,
    ring: f.ring.extended,
    pinky: f.pinky.extended,
  };

  const curled = {
    index: f.index.curled,
    middle: f.middle.curled,
    ring: f.ring.curled,
    pinky: f.pinky.curled,
  };

  let error = 0;
  let count = 0;

  for (const finger of Object.keys(profile.extended)) {
    const delta = values[finger] - profile.extended[finger];
    error += Math.min(1, Math.abs(delta) / .28);
    count++;
  }

  for (const finger of Object.keys(profile.curled)) {
    const delta = curled[finger] - profile.curled[finger];
    error += Math.min(1, Math.abs(delta) / .28);
    count++;
  }

  error += Math.min(
    1,
    Math.abs(pinchRatio(hand) - profile.pinchRatio) / .18
  );

  return clamp01(1 - error / (count + 1));
}

function personalizedScore(hand, gesture, genericScore) {
  const profile = calibrationProfile[gesture];
  if (!profile) return genericScore;

  const shape = poseSimilarity(hand, profile.template);
  const features = personalFeatureSimilarity(hand, profile);

  return clamp01(
    genericScore * .42 +
    shape * .38 +
    features * .20
  );
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

  const drawScore = Math.min(
    f.index.extended,
    f.middle.curled,
    f.ring.curled,
    f.pinky.curled
  ) * (1 - pinch * .95);

  const deleteScore = Math.min(
    f.pinky.extended,
    f.index.curled,
    f.middle.curled,
    f.ring.curled
  ) * (1 - pinch * .95);

  const trackScore = Math.min(
    f.thumb,
    f.index.extended,
    f.middle.extended,
    f.ring.extended,
    f.pinky.extended
  ) * (1 - pinch);

  const drawQualified =
    f.index.extended >= .62 &&
    f.index.extended - otherExtendedForIndex >= .10 &&
    f.middle.curled >= .46 &&
    f.ring.curled >= .46 &&
    f.pinky.curled >= .46;

  const deleteQualified =
    f.pinky.extended >= .62 &&
    f.pinky.extended - otherExtendedForPinky >= .10 &&
    f.index.curled >= .46 &&
    f.middle.curled >= .46 &&
    f.ring.curled >= .46;

  const generic = [
    { name: "GRAB", score: pinch },
    { name: "DRAW", score: drawQualified ? Math.max(drawScore, .64) : drawScore },
    { name: "DELETE", score: deleteQualified ? Math.max(deleteScore, .64) : deleteScore },
    { name: "TRACK", score: trackScore },
  ];

  const ranked = generic.map(item => ({
    name: item.name,
    score: personalizedScore(hand, item.name, item.score),
  })).sort((a, b) => b.score - a.score);

  const best = ranked[0];
  const second = ranked[1];

  if (best.name === "GRAB" && best.score >= .62) {
    return { name: "GRAB", score: best.score, margin: best.score - second.score };
  }

  if (best.score < .56) {
    return { name: "IDLE", score: best.score, margin: best.score };
  }

  if (best.score - second.score < .13) {
    return { name: "IDLE", score: best.score, margin: best.score - second.score };
  }

  const bestProfile = calibrationProfile[best.name];
  const bestProfileShape = bestProfile ? poseSimilarity(hand, bestProfile.template) : 0;

  if (
    best.name === "DRAW" &&
    !drawQualified &&
    !(bestProfile && bestProfileShape >= .78 && generic.find(item => item.name === "DRAW")?.score >= .42)
  ) {
    return { name: "IDLE", score: best.score, margin: best.score - second.score };
  }

  if (
    best.name === "DELETE" &&
    !deleteQualified &&
    !(bestProfile && bestProfileShape >= .78 && generic.find(item => item.name === "DELETE")?.score >= .42)
  ) {
    return { name: "IDLE", score: best.score, margin: best.score - second.score };
  }

  return { name: best.name, score: best.score, margin: best.score - second.score };
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

    // Only the pinky fingertip can select the target.
    // No other part of the hand participates in delete targeting.
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


function resizeCalibrationOverlay() {
  if (!calibrationOverlayCanvas) return;
  const { width, height } = size();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  calibrationOverlayCanvas.width = Math.max(1, Math.round(width * dpr));
  calibrationOverlayCanvas.height = Math.max(1, Math.round(height * dpr));
  calibrationOverlayCanvas.style.width = width + "px";
  calibrationOverlayCanvas.style.height = height + "px";
  calibrationOverlayCtx?.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawCalibrationCameraGuide(hand) {
  if (!calibrationOverlayCtx || !calibrationOverlayCanvas) return;

  const { width, height } = size();
  calibrationOverlayCtx.clearRect(0, 0, width, height);
  calibrationOverlayCtx.save();

  calibrationOverlayCtx.fillStyle = "rgba(0,0,0,.12)";
  calibrationOverlayCtx.fillRect(0, 0, width, height);

  calibrationOverlayCtx.strokeStyle = "rgba(41,232,212,.35)";
  calibrationOverlayCtx.lineWidth = 1.3;
  const frameW = Math.min(width * .84, 1040);
  const frameH = Math.min(height * .72, 710);
  const fx = (width - frameW) * .5;
  const fy = (height - frameH) * .5 + 14;
  calibrationOverlayCtx.setLineDash([8, 10]);
  calibrationOverlayCtx.strokeRect(fx, fy, frameW, frameH);
  calibrationOverlayCtx.setLineDash([]);

  calibrationOverlayCtx.fillStyle = "rgba(41,232,212,.72)";
  calibrationOverlayCtx.font = "700 10px Inter, system-ui, sans-serif";
  calibrationOverlayCtx.letterSpacing = "2px";
  calibrationOverlayCtx.fillText("LIVE CAMERA · KEEP WHOLE HAND INSIDE FRAME", fx, fy - 10);

  if (!hand || hand.length !== 21) {
    calibrationOverlayCtx.restore();
    return;
  }

  const points = hand.map(p => ({
    x: width - p.x * width,
    y: p.y * height,
  }));

  const bounds = points.reduce((acc, p) => ({
    minX: Math.min(acc.minX, p.x),
    minY: Math.min(acc.minY, p.y),
    maxX: Math.max(acc.maxX, p.x),
    maxY: Math.max(acc.maxY, p.y),
  }), { minX: width, minY: height, maxX: 0, maxY: 0 });

  const center = {
    x: (bounds.minX + bounds.maxX) * .5,
    y: (bounds.minY + bounds.maxY) * .5,
  };

  calibrationOverlayCtx.strokeStyle = "rgba(41,232,212,.88)";
  calibrationOverlayCtx.shadowColor = "#29E8D4";
  calibrationOverlayCtx.shadowBlur = 10;
  calibrationOverlayCtx.lineWidth = 2;

  for (const [a, b] of CONNECTIONS) {
    calibrationOverlayCtx.beginPath();
    calibrationOverlayCtx.moveTo(points[a].x, points[a].y);
    calibrationOverlayCtx.lineTo(points[b].x, points[b].y);
    calibrationOverlayCtx.stroke();
  }

  calibrationOverlayCtx.shadowBlur = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    calibrationOverlayCtx.beginPath();
    calibrationOverlayCtx.fillStyle =
      i === 8 ? COLORS.draw :
      i === 20 ? COLORS.hover :
      COLORS.handPoint;
    calibrationOverlayCtx.arc(p.x, p.y, i === 8 || i === 20 ? 5 : 3, 0, Math.PI * 2);
    calibrationOverlayCtx.fill();
  }

  calibrationOverlayCtx.strokeStyle = "rgba(216,255,90,.42)";
  calibrationOverlayCtx.beginPath();
  calibrationOverlayCtx.arc(center.x, center.y, 42 + Math.min(38, (bounds.maxX - bounds.minX) * .12), 0, Math.PI * 2);
  calibrationOverlayCtx.stroke();

  calibrationOverlayCtx.fillStyle = "rgba(0,0,0,.55)";
  calibrationOverlayCtx.fillRect(center.x - 47, center.y - 18, 94, 36);
  calibrationOverlayCtx.fillStyle = "#D8FF5A";
  calibrationOverlayCtx.textAlign = "center";
  calibrationOverlayCtx.font = "700 10px Inter, system-ui, sans-serif";
  calibrationOverlayCtx.fillText("HAND LOCKED", center.x, center.y - 2);
  calibrationOverlayCtx.font = "600 8px Inter, system-ui, sans-serif";
  calibrationOverlayCtx.fillStyle = "rgba(238,254,250,.72)";
  calibrationOverlayCtx.fillText("3D + TIME + SHAPE", center.x, center.y + 11);

  calibrationOverlayCtx.restore();
}

function projectWorldPoint(point, center) {
  let x = (point.x - center.x) * 860;
  let y = (point.y - center.y) * 860;
  let z = (point.z - center.z) * 860;

  const yaw = -0.58;
  const pitch = 0.22;

  const x1 = x * Math.cos(yaw) - z * Math.sin(yaw);
  const z1 = x * Math.sin(yaw) + z * Math.cos(yaw);

  const y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);

  const camera = 760;
  const depth = Math.max(300, camera + z2);
  const perspective = camera / depth;

  return {
    x: calibrationCanvas.width * .5 + x1 * perspective,
    y: calibrationCanvas.height * .52 + y2 * perspective,
    depth: perspective,
  };
}

function draw3DPreview(hand) {
  if (!calibrationCtx || !calibrationCanvas) return;

  const w = calibrationCanvas.width;
  const h = calibrationCanvas.height;
  calibrationCtx.clearRect(0, 0, w, h);

  calibrationCtx.save();
  calibrationCtx.fillStyle = "#000";
  calibrationCtx.fillRect(0, 0, w, h);

  calibrationCtx.strokeStyle = "rgba(41,232,212,.09)";
  calibrationCtx.lineWidth = 1;

  for (let i = 1; i < 7; i++) {
    const x = w * i / 7;
    const y = h * i / 7;
    calibrationCtx.beginPath();
    calibrationCtx.moveTo(x, 0);
    calibrationCtx.lineTo(x, h);
    calibrationCtx.stroke();
    calibrationCtx.beginPath();
    calibrationCtx.moveTo(0, y);
    calibrationCtx.lineTo(w, y);
    calibrationCtx.stroke();
  }

  if (!hand || hand.length !== 21) {
    calibrationCtx.restore();
    return;
  }

  const center = {
    x: (hand[0].x + hand[9].x + hand[13].x + hand[17].x) / 4,
    y: (hand[0].y + hand[9].y + hand[13].y + hand[17].y) / 4,
    z: (hand[0].z + hand[9].z + hand[13].z + hand[17].z) / 4,
  };

  const projected = hand.map(point => projectWorldPoint(point, center));

  calibrationCtx.lineCap = "round";
  calibrationCtx.lineJoin = "round";
  calibrationCtx.shadowColor = "#29E8D4";
  calibrationCtx.shadowBlur = 12;
  calibrationCtx.lineWidth = 2.4;
  calibrationCtx.strokeStyle = "rgba(41,232,212,.82)";

  for (const [a, b] of CONNECTIONS) {
    calibrationCtx.beginPath();
    calibrationCtx.moveTo(projected[a].x, projected[a].y);
    calibrationCtx.lineTo(projected[b].x, projected[b].y);
    calibrationCtx.stroke();
  }

  calibrationCtx.shadowBlur = 0;

  for (let i = 0; i < projected.length; i++) {
    const p = projected[i];
    const pointSize = 3.2 + p.depth * 1.7;

    calibrationCtx.beginPath();
    calibrationCtx.fillStyle = i === 8
      ? COLORS.draw
      : i === 20
        ? COLORS.hover
        : COLORS.handPoint;
    calibrationCtx.arc(p.x, p.y, Math.max(2.2, pointSize), 0, Math.PI * 2);
    calibrationCtx.fill();
  }

  calibrationCtx.fillStyle = "rgba(238,254,250,.54)";
  calibrationCtx.font = "700 9px Inter, system-ui, sans-serif";
  calibrationCtx.fillText("TRUE 3D WORLD LANDMARKS", 12, 18);

  calibrationCtx.restore();
}

function updateCalibration(now) {
  if (!calibrationActive) return;

  const step = calibrationSteps[calibrationIndex];
  if (!step) return;

  if (calibrationConfirmedUntil) {
    if (now < calibrationConfirmedUntil) {
      drawCalibrationCameraGuide(latestHand);
      draw3DPreview(latestWorldHand);
      calibrationConfirm.textContent = "✓ CONFIRMED";
      calibrationConfirm.classList.add("good");
      calibrationConfirm.classList.remove("bad");
      calibrationFeedback.textContent =
        calibrationIndex === calibrationSteps.length - 1
          ? "Final hand profile captured. Starting the canvas next."
          : "Pose captured. Next step is ready.";
      calibrationStep.textContent =
        "CONFIRMED · STEP " + (calibrationIndex + 1) + " / " + calibrationSteps.length;
      calibrationMeterFill.style.width = "100%";
      return;
    }

    calibrationConfirmedUntil = 0;
    calibrationIndex += 1;
    calibrationHoldStarted = 0;
    calibrationConfirmedUntil = 0;
    calibrationSamples = [];
    calibrationRecentPoses = [];
    calibrationRecentScales = [];

    if (calibrationIndex >= calibrationSteps.length) {
      calibrationActive = false;
      stage.classList.remove("calibrating");
      calibrationScreen?.classList.add("hidden");
      stableGesture = "IDLE";
      gestureCandidate = "IDLE";
      gestureCandidateAt = now;
      resetMotionFilter();
      return;
    }
  }

  setupCalibrationPoseGuide(calibrationSteps[calibrationIndex]);
  drawCalibrationCameraGuide(latestHand);
  draw3DPreview(latestWorldHand);

  calibrationStep.textContent = "STEP " + (calibrationIndex + 1) + " / " + calibrationSteps.length;
  calibrationTitle.textContent = step.title;
  calibrationText.textContent = step.text;

  const handScore = handImageQuality(latestHand);
  const shapeResult = latestWorldHand ? classifyGestureBeforeCalibration(latestWorldHand) : { name: "IDLE", score: 0 };
  const shapeScore = shapeResult.name === step.gesture ? shapeResult.score : 0;
  const depthScore = calibrationDepthScore(latestWorldHand);
  const stabilityScore = calibrationStabilityScore(latestWorldHand);

  const centered = latestHand?.length === 21
    ? clamp01(1 - Math.hypot(
        ((latestHand.reduce((s, p) => s + p.x, 0) / 21) - .5) / .42,
        ((latestHand.reduce((s, p) => s + p.y, 0) / 21) - .5) / .48
      ))
    : 0;

  const personalScore = calibrationProfile[step.gesture] ? 1 : calibrationSamples.length ? .62 : .45;
  const overall = handScore * .30 + shapeScore * .30 + depthScore * .15 + stabilityScore * .15 + centered * .10;

  const ready =
    handScore >= .58 &&
    shapeScore >= .68 &&
    depthScore >= .42 &&
    stabilityScore >= .64 &&
    centered >= .42;

  if (latestWorldHand && canonicalHand(latestWorldHand)) {
    calibrationRecentPoses.push(canonicalHand(latestWorldHand));
    if (calibrationRecentPoses.length > 12) calibrationRecentPoses.shift();

    calibrationRecentScales.push(dist(latestWorldHand[5], latestWorldHand[17]));
    if (calibrationRecentScales.length > 18) calibrationRecentScales.shift();
  }

  if (ready) {
    if (!calibrationHoldStarted) {
      calibrationHoldStarted = now;
      calibrationSamples = [];
    }
    captureCalibrationSample(step.gesture, latestWorldHand);
  } else {
    calibrationHoldStarted = 0;
    calibrationSamples = [];
  }

  const progress = calibrationHoldStarted
    ? Math.max(0, Math.min(1, (now - calibrationHoldStarted) / calibrationHoldMs))
    : 0;

  calibrationMeterFill.style.width = Math.round(progress * 100) + "%";

  const setCheck = (node, value, good) => {
    if (!node) return;
    node.textContent = value;
    node.parentElement?.classList.toggle("good", good);
    node.parentElement?.classList.toggle("bad", !good && value !== "—");
  };

  setCheck(checkHand,
    handScore >= .58 ? "VISIBLE" : "SEARCHING",
    handScore >= .58
  );
  setCheck(checkShape,
    shapeScore >= .68 ? Math.round(shapeScore * 100) + "% MATCH" : "MATCH THE GUIDE",
    shapeScore >= .68
  );
  setCheck(checkDepth,
    depthScore >= .42 ? "3D OK" : "SHOW PALM",
    depthScore >= .42
  );
  setCheck(checkStability,
    stabilityScore >= .64 ? "STEADY" : "HOLD STILL",
    stabilityScore >= .64
  );
  setCheck(checkPersonal,
    calibrationProfile[step.gesture] ? "LEARNED" : "LEARNING",
    Boolean(calibrationProfile[step.gesture])
  );

  calibrationConfirm.classList.toggle("good", ready);
  calibrationConfirm.classList.toggle("bad", !ready);

  if (!latestHand) {
    calibrationConfirm.textContent = "WAITING FOR HAND";
    calibrationFeedback.textContent = "Move your whole hand into the live camera frame.";
  } else if (shapeScore < .68) {
    calibrationConfirm.textContent = "NOT MATCHED YET";
    calibrationFeedback.textContent = "Follow the finger guide exactly. TUFF needs the correct pose, not just a similar pose.";
  } else if (depthScore < .42) {
    calibrationConfirm.textContent = "3D VIEW NEEDS MORE DATA";
    calibrationFeedback.textContent = "Keep the whole hand visible and avoid covering fingertips.";
  } else if (stabilityScore < .64) {
    calibrationConfirm.textContent = "HOLD STEADY";
    calibrationFeedback.textContent = "Good pose. Stop moving for the confirmation countdown.";
  } else {
    calibrationConfirm.textContent = Math.round(overall * 100) + "% · READY";
    calibrationFeedback.textContent = "Confirmed when the bar reaches 100%. Keep the same pose.";
  }

  if (progress >= 1 && !calibrationConfirmedUntil) {
    learnCalibrationProfile(step.gesture);
    calibrationConfirmedUntil = now + 650;
    calibrationConfirm.textContent = "✓ CONFIRMED";
    calibrationConfirm.classList.add("good");
    calibrationConfirm.classList.remove("bad");
    calibrationFeedback.textContent =
      calibrationIndex === calibrationSteps.length - 1
        ? "Final pose saved. Your personal 3D profile is ready."
        : "This gesture is saved. The next instruction will appear immediately.";
  }
}

function classifyGestureBeforeCalibration(hand) {
  // Calibration must never use the user's learned profile while that profile
  // is still being collected. Recreate the anatomical-only ranking here.
  const f = fingerFlags(hand);
  const pinch = pinchConfidence(hand);

  const drawScore = Math.min(
    f.index.extended,
    f.middle.curled,
    f.ring.curled,
    f.pinky.curled
  ) * (1 - pinch * .95);

  const deleteScore = Math.min(
    f.pinky.extended,
    f.index.curled,
    f.middle.curled,
    f.ring.curled
  ) * (1 - pinch * .95);

  const trackScore = Math.min(
    f.thumb,
    f.index.extended,
    f.middle.extended,
    f.ring.extended,
    f.pinky.extended
  ) * (1 - pinch);

  const ranked = [
    { name: "GRAB", score: pinch },
    { name: "DRAW", score: drawScore },
    { name: "DELETE", score: deleteScore },
    { name: "TRACK", score: trackScore },
  ].sort((a, b) => b.score - a.score);

  return ranked[0];
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
  if (calibrationActive) return;

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
    commitGesture(classifyGesture(latestWorldHand || latestHand), now);
  } else if (now - lastSampleAt > SETTINGS.lostGraceMs) {
    stableGesture = "IDLE";
    gestureCandidate = "IDLE";
    grabbedStrokeIndex = -1;
    previousGrabPoint = null;
    deleteTarget = -1;
  }

  const hoverPoints =
    stableGesture === "GRAB"
      ? [activePoints[4], activePoints[8], midpoint(activePoints[4], activePoints[8])].filter(Boolean)
      : stableGesture === "DELETE"
        ? [activePoints[20]].filter(Boolean)
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
    latestWorldHand = result.worldLandmarks?.[0] || latestHand;
  } catch (error) {
    console.warn("Hand detection failed", error);
    latestHand = null;
    latestWorldHand = null;
  }
}

function frame(now) {
  const dt = Math.min(.05, Math.max(.001, (now - lastFrameAt) / 1000));
  lastFrameAt = now;

  processVideo(now);
  if (calibrationActive) {
    updateCalibration(now);
  } else {
    updateInteraction(now);
  }
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
    stage.classList.add("calibrating");

    calibrationActive = true;
    calibrationIndex = 0;
    calibrationHoldStarted = 0;
    calibrationSamples = [];
    calibrationRecentPoses = [];
    calibrationRecentScales = [];
    calibrationLastExpected = null;
    calibrationLastConfidence = 0;
    for (const key of Object.keys(calibrationProfile)) delete calibrationProfile[key];
    calibrationScreen?.classList.remove("hidden");
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
  if (stream) stream.getTracks().forEach(track => track.stop());
});

resizeCanvas();
requestAnimationFrame(frame);
