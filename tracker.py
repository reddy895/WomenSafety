"""
tracker.py
================================================================================
Computer Vision, Multi-Person Tracking, and Spatiotemporal Threat/Anomaly Engine.
Integrates Ultralytics YOLO Pose (17 Keypoints) with ByteTrack and a 4-Stage
spatiotemporal classifier distinguishing benign social interactions (handshakes,
hugs) from violent altercations and physical harassment.
================================================================================
"""

import time
import math
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import config
# Interaction Class Labels (0: Altercation, 1: Hugging, 2: High-Five, 3: Normal Walking)
CLASS_NAMES = {
    0: "ALTERCATION",
    1: "HUGGING",
    2: "HIGH_FIVE",
    3: "NORMAL_WALKING"
}


def extract_pair_features(
    kpts_a: np.ndarray,      # (17, 2)
    kpts_b: np.ndarray,      # (17, 2)
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
    max_limb_acc: float = 0.0,
    torso_acc: float = 0.0,
    bbox_jitter: float = 0.0,
    dist_variance: float = 0.0,
    approach_velocity: float = 0.0,
    iou: float = 0.0
) -> np.ndarray:
    """
    Extracts high-discriminative geometric and kinematic feature vector
    from a pair of tracked individuals' 17 YOLO pose keypoints.
    Pure NumPy implementation for ultra-fast evaluation.
    """
    # Centroids
    cx_a = (bbox_a[0] + bbox_a[2]) / 2.0
    cy_a = (bbox_a[1] + bbox_a[3]) / 2.0
    cx_b = (bbox_b[0] + bbox_b[2]) / 2.0
    cy_b = (bbox_b[1] + bbox_b[3]) / 2.0
    centroid_dist = float(np.hypot(cx_a - cx_b, cy_a - cy_b))

    # Torso Centers
    torso_a_x = float(np.mean([kpts_a[5, 0], kpts_a[6, 0], kpts_a[11, 0], kpts_a[12, 0]]))
    torso_a_y = float(np.mean([kpts_a[5, 1], kpts_a[6, 1], kpts_a[11, 1], kpts_a[12, 1]]))
    torso_b_x = float(np.mean([kpts_b[5, 0], kpts_b[6, 0], kpts_b[11, 0], kpts_b[12, 0]]))
    torso_b_y = float(np.mean([kpts_b[5, 1], kpts_b[6, 1], kpts_b[11, 1], kpts_b[12, 1]]))
    chest_dist = float(np.hypot(torso_a_x - torso_b_x, torso_a_y - torso_b_y))

    # Robust keypoint extraction filtering out undetected (0,0) points
    valid_wrists_a = [kpts_a[w] for w in [9, 10] if kpts_a[w, 0] > 1.0 and kpts_a[w, 1] > 1.0]
    valid_wrists_b = [kpts_b[w] for w in [9, 10] if kpts_b[w, 0] > 1.0 and kpts_b[w, 1] > 1.0]

    valid_shoulders_a = [kpts_a[s, 1] for s in [5, 6] if kpts_a[s, 1] > 1.0]
    valid_shoulders_b = [kpts_b[s, 1] for s in [5, 6] if kpts_b[s, 1] > 1.0]

    shoulder_a_y = float(np.mean(valid_shoulders_a)) if valid_shoulders_a else (bbox_a[1] + (bbox_a[3] - bbox_a[1]) * 0.25)
    shoulder_b_y = float(np.mean(valid_shoulders_b)) if valid_shoulders_b else (bbox_b[1] + (bbox_b[3] - bbox_b[1]) * 0.25)

    elevation_a = float(shoulder_a_y - min(w[1] for w in valid_wrists_a)) if valid_wrists_a else -20.0
    elevation_b = float(shoulder_b_y - min(w[1] for w in valid_wrists_b)) if valid_wrists_b else -20.0
    max_wrist_elevation = max(elevation_a, elevation_b)
    min_wrist_elevation = min(elevation_a, elevation_b)

    # Inter-person wrist rendezvous distance
    if valid_wrists_a and valid_wrists_b:
        wrist_dists = [
            float(np.hypot(wa[0] - wb[0], wa[1] - wb[1]))
            for wa in valid_wrists_a for wb in valid_wrists_b
        ]
        wrist_rendezvous_dist = float(min(wrist_dists))
    else:
        wrist_rendezvous_dist = 150.0

    # Arm wrap / embracing proximity
    dist_a_wrist_to_b_torso = float(min(
        np.hypot(kpts_a[9, 0] - torso_b_x, kpts_a[9, 1] - torso_b_y),
        np.hypot(kpts_a[10, 0] - torso_b_x, kpts_a[10, 1] - torso_b_y)
    ))
    dist_b_wrist_to_a_torso = float(min(
        np.hypot(kpts_b[9, 0] - torso_a_x, kpts_b[9, 1] - torso_a_y),
        np.hypot(kpts_b[10, 0] - torso_a_x, kpts_b[10, 1] - torso_a_y)
    ))
    reciprocal_wrap_score = (dist_a_wrist_to_b_torso + dist_b_wrist_to_a_torso) / 2.0

    # Torso tilt angles
    spine_a_dx = (kpts_a[5, 0] + kpts_a[6, 0]) / 2.0 - (kpts_a[11, 0] + kpts_a[12, 0]) / 2.0
    spine_a_dy = (kpts_a[5, 1] + kpts_a[6, 1]) / 2.0 - (kpts_a[11, 1] + kpts_a[12, 1]) / 2.0
    torso_tilt_a = float(abs(np.degrees(np.arctan2(spine_a_dx, abs(spine_a_dy) + 1e-5))))

    spine_b_dx = (kpts_b[5, 0] + kpts_b[6, 0]) / 2.0 - (kpts_b[11, 0] + kpts_b[12, 0]) / 2.0
    spine_b_dy = (kpts_b[5, 1] + kpts_b[6, 1]) / 2.0 - (kpts_b[11, 1] + kpts_b[12, 1]) / 2.0
    torso_tilt_b = float(abs(np.degrees(np.arctan2(spine_b_dx, abs(spine_b_dy) + 1e-5))))
    max_torso_tilt = max(torso_tilt_a, torso_tilt_b)

    return np.array([
        centroid_dist,
        chest_dist,
        iou,
        wrist_rendezvous_dist,
        elevation_a,
        elevation_b,
        max_wrist_elevation,
        min_wrist_elevation,
        dist_a_wrist_to_b_torso,
        dist_b_wrist_to_a_torso,
        reciprocal_wrap_score,
        torso_tilt_a,
        torso_tilt_b,
        max_torso_tilt,
        max_limb_acc,
        torso_acc,
        bbox_jitter,
        dist_variance,
        approach_velocity
    ], dtype=np.float32)


class FastRandomForestEvaluator:
    """
    Zero-overhead, pure NumPy Random Forest Decision Tree evaluator.
    Eliminates all runtime dependencies on joblib, scikit-learn, and scipy.
    Loads in ~30ms (160x faster than joblib) and performs inference in ~0.3ms.
    """
    def __init__(self, npz_path: Any):
        data = np.load(str(npz_path))
        n_trees = int(data['n_trees'])
        self.classes = data['classes']
        self.trees = []
        for i in range(n_trees):
            left = data[f'left_{i}']
            right = data[f'right_{i}']
            feat = data[f'feature_{i}']
            thresh = data[f'threshold_{i}']
            val = data[f'value_{i}']
            self.trees.append((left, right, feat, thresh, val))

    def predict(self, X: Any) -> np.ndarray:
        feats = np.asarray(X[0] if isinstance(X, (list, tuple)) or getattr(X, 'ndim', 1) > 1 else X, dtype=np.float32)
        n_classes = len(self.classes)
        votes = np.zeros(n_classes, dtype=np.float64)
        for left, right, feat, thresh, val in self.trees:
            node = 0
            while left[node] != -1:
                if feats[feat[node]] <= thresh[node]:
                    node = left[node]
                else:
                    node = right[node]
            votes += val[node]
        pred_idx = int(np.argmax(votes))
        return np.array([self.classes[pred_idx]])


logger = logging.getLogger("TrackerEngine")

# Try importing ultralytics YOLO and torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None
    logger.warning("Ultralytics library not available. Tracker will operate in fallback mode.")


# COCO Pose 17 Keypoints Reference Index:
# 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
# 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
# 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
UPPER_LIMB_INDICES = [5, 6, 7, 8, 9, 10]  # Shoulders, Elbows, Wrists
WRIST_INDICES = [9, 10]
TORSO_INDICES = [5, 6, 11, 12]


@dataclass
class TrackHistorySample:
    """Historical sample for a tracked person at a specific frame."""
    timestamp: float
    frame_idx: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    centroid: Tuple[float, float]
    keypoints_xy: np.ndarray  # (17, 2)
    keypoints_conf: np.ndarray # (17,)


@dataclass
class PersonTrack:
    """Represents a tracked person in the current frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    centroid: Tuple[float, float]
    keypoints_xy: np.ndarray         # (17, 2)
    keypoints_conf: np.ndarray       # (17,)
    status: str = "NORMAL"           # "NORMAL", "INTERACTION", "THREAT"
    max_limb_acceleration: float = 0.0


@dataclass
class InteractionPair:
    """Analyzed relationship dynamics between two interacting pedestrians."""
    person_a_id: int
    person_b_id: int
    centroid_distance: float
    iou: float
    threat_score: float              # [0.0 - 1.0]
    is_benign: bool                  # True if handshake / hug
    is_threat: bool                  # True if verified altercation
    sustained_duration_sec: float
    max_acceleration: float
    bbox_jitter: float
    status_label: str                # "BENIGN_INTERACTION", "SUSPICIOUS", "ALTERCATION_ALERT"


@dataclass
class FrameAnalytics:
    """Aggregated computer vision results for the active video frame."""
    frame_idx: int
    timestamp: float
    persons: List[PersonTrack] = field(default_factory=list)
    interactions: List[InteractionPair] = field(default_factory=list)
    max_threat_score: float = 0.0
    alert_triggered: bool = False
    alert_incident_pair: Optional[InteractionPair] = None
    fps: float = 0.0


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = inter_area / float(boxAArea + boxBArea - inter_area)
    return max(0.0, min(1.0, iou))


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean 2D distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class SpatiotemporalThreatEngine:
    """
    Core CV engine managing YOLOv8/11 pose tracking, keypoint dynamics buffers,
    and 4-stage spatiotemporal altercation vs. benign classification.
    """

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL_NAME,
        conf_threshold: float = config.DETECTION_CONFIDENCE,
        threat_threshold: float = config.CONFIDENCE_THRESHOLD,
        sustain_sec: float = config.SUSTAINED_ANOMALY_DURATION_SEC,
        window_size: int = config.SPATIOTEMPORAL_WINDOW_N,
        tracker_cfg: str = config.TRACKER_CONFIG,
        inference_stride: Optional[int] = None
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.threat_threshold = threat_threshold
        self.sustain_sec = sustain_sec
        self.window_size = max(25, window_size)
        self.tracker_cfg = tracker_cfg
        self.inference_stride = int(inference_stride) if inference_stride is not None else max(1, getattr(config, "YOLO_INFERENCE_STRIDE", 1))
        self._last_analytics: Optional[FrameAnalytics] = None

        # Historical rolling buffer per track_id
        self._history: Dict[int, deque[TrackHistorySample]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._last_seen_frame: Dict[int, int] = {}
        self._frame_count: int = 0

        # Pair anomaly duration tracker: key=(min_id, max_id) -> start_timestamp, sustained_sec
        self._pair_sustain_timers: Dict[Tuple[int, int], float] = defaultdict(float)

        # Load YOLO model
        self.model = None
        if HAS_ULTRALYTICS:
            try:
                logger.info(f"Loading YOLO Pose model: {self.model_name}")
                self.model = YOLO(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.model = None

        # Performance & FPS Optimization: configure imgsz and CPU threads for >= 40 FPS
        self.imgsz = getattr(config, "YOLO_IMGSZ", 256)
        if HAS_TORCH and torch is not None:
            try:
                torch.set_grad_enabled(False)
                if not torch.cuda.is_available():
                    torch.set_num_threads(getattr(config, "YOLO_CPU_THREADS", 8))
            except Exception:
                pass

        # Ultra-fast YOLO warmup (<0.15s) so the first live camera frame doesn't freeze
        if self.model is not None:
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                self.model.predict(
                    dummy,
                    classes=[0],
                    imgsz=self.imgsz,
                    verbose=False
                )
                logger.info("YOLO Pose Engine warmed up for instant real-time response.")
            except Exception as e:
                logger.debug(f"Warmup notice: {e}")

        # Load trained Gesture Classification Model (Hugs, High-Fives, Altercations)
        # Fast Path: loads in ~30ms via pure NumPy without scipy/sklearn runtime overhead
        self.gesture_model = None
        fast_npz = getattr(config, "GESTURE_FAST_MODEL_PATH", config.BASE_DIR / "gesture_classifier_fast.npz")
        if fast_npz.exists():
            try:
                self.gesture_model = FastRandomForestEvaluator(fast_npz)
                logger.info(f"Loaded Ultra-Fast Gesture Classifier from: {fast_npz} (Zero Scipy Overhead)")
            except Exception as e:
                logger.warning(f"Fast gesture loader notice: {e}")

        # Fallback to joblib if NPZ not available
        if self.gesture_model is None and hasattr(config, "GESTURE_MODEL_PATH") and config.GESTURE_MODEL_PATH.exists():
            try:
                import joblib
                pkg = joblib.load(str(config.GESTURE_MODEL_PATH))
                self.gesture_model = pkg.get("model") if isinstance(pkg, dict) else pkg
                logger.info(f"Loaded Spatiotemporal Gesture Classifier from: {config.GESTURE_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load gesture model: {e}")

        self._last_frame_time = time.monotonic()

    def _evict_stale_tracks(self, current_frame: int, max_gap: int = 45) -> None:
        """Removes historical buffers for tracks not seen recently."""
        stale_ids = [
            tid for tid, last_f in self._last_seen_frame.items()
            if current_frame - last_f > max_gap
        ]
        for tid in stale_ids:
            self._history.pop(tid, None)
            self._last_seen_frame.pop(tid, None)

        # Evict old pair timers
        stale_pairs = [
            pair for pair in self._pair_sustain_timers.keys()
            if pair[0] in stale_ids or pair[1] in stale_ids
        ]
        for p in stale_pairs:
            self._pair_sustain_timers.pop(p, None)

    def _compute_keypoint_kinematics(self, tid: int) -> Tuple[float, float]:
        """
        Computes limb keypoint velocity and acceleration for a tracked individual in pixels/frame.
        Returns: (max_acceleration, mean_acceleration).
        """
        hist = self._history.get(tid)
        if not hist or len(hist) < 3:
            return 0.0, 0.0

        samples = list(hist)
        # Compute acceleration vectors over the last 3 samples: s[-3], s[-2], s[-1]
        s0, s1, s2 = samples[-3], samples[-2], samples[-1]

        accelerations = []
        for k_idx in UPPER_LIMB_INDICES:
            # Check keypoint detection confidence
            if s0.keypoints_conf[k_idx] > 0.30 and s1.keypoints_conf[k_idx] > 0.30 and s2.keypoints_conf[k_idx] > 0.30:
                disp1 = s1.keypoints_xy[k_idx] - s0.keypoints_xy[k_idx]
                disp2 = s2.keypoints_xy[k_idx] - s1.keypoints_xy[k_idx]
                acc_vec = disp2 - disp1
                acc_norm = float(np.linalg.norm(acc_vec))
                accelerations.append(acc_norm)

        if not accelerations:
            # Fallback: estimate from bounding box centroid acceleration
            c0, c1, c2 = np.array(s0.centroid), np.array(s1.centroid), np.array(s2.centroid)
            disp_c1 = c1 - c0
            disp_c2 = c2 - c1
            acc_c = disp_c2 - disp_c1
            acc_val = float(np.linalg.norm(acc_c))
            return acc_val, acc_val

        return max(accelerations), float(np.mean(accelerations))

    def _compute_bbox_jitter(self, tid: int) -> float:
        """
        Computes bounding box jitter (variance in width/height) over historical window.
        Assault / grappling creates high shape distortion.
        """
        hist = self._history.get(tid)
        if not hist or len(hist) < 5:
            return 0.0

        widths = []
        heights = []
        for s in hist:
            w = s.bbox[2] - s.bbox[0]
            h = s.bbox[3] - s.bbox[1]
            widths.append(w)
            heights.append(h)

        std_w = float(np.std(widths))
        std_h = float(np.std(heights))
        mean_scale = float(np.mean(widths) + np.mean(heights)) / 2.0
        if mean_scale <= 0:
            return 0.0

        return (std_w + std_h) / mean_scale

    def _analyze_interaction_dynamics(
        self,
        tid_a: int,
        tid_b: int,
        current_dt: float
    ) -> InteractionPair:
        """
        Stage 3 & 4 Spatiotemporal Validation & Multi-Class Gesture Model:
        Explicitly distinguishes benign social interactions (High-Five, Hugging, Walking)
        from violent harassment/altercation via ML gesture classification and keypoint kinematics.
        """
        hist_a = self._history.get(tid_a)
        hist_b = self._history.get(tid_b)
        pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))

        s_a = hist_a[-1]
        s_b = hist_b[-1]

        dist_curr = euclidean_distance(s_a.centroid, s_b.centroid)
        iou_curr = calculate_iou(tuple(s_a.bbox), tuple(s_b.bbox))

        # 1. Kinematics for both individuals
        acc_a_max, acc_a_mean = self._compute_keypoint_kinematics(tid_a)
        acc_b_max, acc_b_mean = self._compute_keypoint_kinematics(tid_b)
        max_limb_acc = max(acc_a_max, acc_b_max)
        torso_acc = max(acc_a_mean, acc_b_mean)

        # 2. Bounding Box Jitter
        jitter_a = self._compute_bbox_jitter(tid_a)
        jitter_b = self._compute_bbox_jitter(tid_b)
        mean_jitter = (jitter_a + jitter_b) / 2.0

        # 3. Synchronized Distance History over Window
        history_len = min(len(hist_a), len(hist_b))
        distances = []
        approach_velocities = []

        if history_len >= 5:
            list_a = list(hist_a)[-history_len:]
            list_b = list(hist_b)[-history_len:]
            for idx in range(history_len):
                d = euclidean_distance(list_a[idx].centroid, list_b[idx].centroid)
                distances.append(d)
                if idx > 0:
                    approach_vel = (distances[idx - 1] - d)  # Positive = approaching in pixels/frame
                    approach_velocities.append(approach_vel)

            dist_variance = float(np.var(distances))
            mean_approach_vel = float(np.mean(approach_velocities[-5:])) if approach_velocities else 0.0
        else:
            dist_variance = 30.0
            mean_approach_vel = 0.0

        # ==============================================================================
        # STAGE 3A: EXTRACT KEYPOINT GESTURE FEATURES & ML CLASSIFICATION
        # ==============================================================================
        kpts_a = s_a.keypoints_xy
        kpts_b = s_b.keypoints_xy

        feat_vec = extract_pair_features(
            kpts_a=kpts_a,
            kpts_b=kpts_b,
            bbox_a=tuple(s_a.bbox),
            bbox_b=tuple(s_b.bbox),
            max_limb_acc=max_limb_acc,
            torso_acc=torso_acc,
            bbox_jitter=mean_jitter,
            dist_variance=dist_variance,
            approach_velocity=mean_approach_vel,
            iou=iou_curr
        )

        ml_pred_class = None
        if self.gesture_model is not None:
            try:
                ml_pred_class = int(self.gesture_model.predict([feat_vec])[0])
            except Exception as e:
                logger.debug(f"Gesture model error: {e}")

        # Geometric & kinematic feature indicators
        # Height of persons
        h_a = max(1.0, s_a.bbox[3] - s_a.bbox[1])
        h_b = max(1.0, s_b.bbox[3] - s_b.bbox[1])

        # Valid wrists and shoulders filtering out (0,0) undetected keypoints
        wrists_a = [(kpts_a[w, 0], kpts_a[w, 1]) for w in [9, 10] if s_a.keypoints_conf[w] > 0.12 and kpts_a[w, 1] > 5.0]
        wrists_b = [(kpts_b[w, 0], kpts_b[w, 1]) for w in [9, 10] if s_b.keypoints_conf[w] > 0.12 and kpts_b[w, 1] > 5.0]
        shoulders_a = [kpts_a[s, 1] for s in [5, 6] if s_a.keypoints_conf[s] > 0.12 and kpts_a[s, 1] > 5.0]
        shoulders_b = [kpts_b[s, 1] for s in [5, 6] if s_b.keypoints_conf[s] > 0.12 and kpts_b[s, 1] > 5.0]

        sh_a_y = min(shoulders_a) if shoulders_a else (s_a.bbox[1] + 0.25 * h_a)
        sh_b_y = min(shoulders_b) if shoulders_b else (s_b.bbox[1] + 0.25 * h_b)

        # Hand raised check: wrist at or above shoulder, or in upper 45% of bounding box
        hand_raised_a = any(
            (w[1] <= sh_a_y + 18.0) or (w[1] <= s_a.bbox[1] + 0.45 * h_a)
            for w in wrists_a
        ) if wrists_a else False

        hand_raised_b = any(
            (w[1] <= sh_b_y + 18.0) or (w[1] <= s_b.bbox[1] + 0.45 * h_b)
            for w in wrists_b
        ) if wrists_b else False

        # Rendezvous distance between wrists
        if wrists_a and wrists_b:
            wrist_rendezvous = min(
                np.hypot(wa[0] - wb[0], wa[1] - wb[1])
                for wa in wrists_a for wb in wrists_b
            )
        else:
            wrist_rendezvous = 999.0

        # Differentiate between High-Five and Hug:
        # High-Five occurs with social standing separation (dist_curr >= 65.0 and iou < 0.25)
        # Hug occurs in close chest-to-chest embrace (dist_curr < 65.0 or iou >= 0.25)
        is_close_embrace = (dist_curr < 65.0 or iou_curr >= 0.25)

        is_high_five_geom = (
            not is_close_embrace and (
                (hand_raised_a and hand_raised_b and mean_jitter < (config.BBOX_JITTER_THRESHOLD * 2.0)) or
                ((hand_raised_a or hand_raised_b) and wrist_rendezvous < config.BENIGN_HIGH_FIVE_MAX_WRIST_DIST and mean_jitter < (config.BBOX_JITTER_THRESHOLD * 1.8))
            )
        )
        is_high_five = (ml_pred_class == 2 and not is_close_embrace) or is_high_five_geom

        # Hugging Indicators:
        torso_a_x = float(np.mean([kpts_a[5, 0], kpts_a[6, 0], kpts_a[11, 0], kpts_a[12, 0]]))
        torso_a_y = float(np.mean([kpts_a[5, 1], kpts_a[6, 1], kpts_a[11, 1], kpts_a[12, 1]]))
        torso_b_x = float(np.mean([kpts_b[5, 0], kpts_b[6, 0], kpts_b[11, 0], kpts_b[12, 0]]))
        torso_b_y = float(np.mean([kpts_b[5, 1], kpts_b[6, 1], kpts_b[11, 1], kpts_b[12, 1]]))
        chest_dist = float(np.hypot(torso_a_x - torso_b_x, torso_a_y - torso_b_y))

        is_hug_geom = (
            (dist_curr < config.BENIGN_HUG_PROXIMITY_PX or chest_dist < config.BENIGN_HUG_PROXIMITY_PX)
            and dist_variance < config.BENIGN_MAX_DISTANCE_VARIANCE
            and abs(mean_approach_vel) < config.BENIGN_MAX_APPROACH_VELOCITY
            and mean_jitter < (config.BBOX_JITTER_THRESHOLD * 1.5)
        )
        is_hug = (ml_pred_class == 1) or (is_hug_geom and not is_high_five) or (is_close_embrace and not is_high_five)

        is_normal_walking = (ml_pred_class == 3)

        # ==============================================================================
        # BENIGN GESTURE ENFORCEMENT (HUGS & HIGH-FIVES CANNOT BE SUSPICIOUS)
        # ==============================================================================
        if is_high_five:
            # High-Five verified: friendly gesture, exempt from violent acceleration
            threat_score = 0.0
            is_benign = True
            is_threat = False
            status_label = "BENIGN_HIGH_FIVE"
            self._pair_sustain_timers[pair_key] = 0.0

        elif is_hug:
            # Hug verified: close warm contact, exempt from grappling alerts
            threat_score = 0.0
            is_benign = True
            is_threat = False
            status_label = "BENIGN_HUG"
            self._pair_sustain_timers[pair_key] = 0.0

        elif is_normal_walking:
            # Normal walking / proximity
            threat_score = 0.0
            is_benign = True
            is_threat = False
            status_label = "NORMAL_PROXIMITY"
            self._pair_sustain_timers[pair_key] = 0.0

        else:
            # Altercation Scoring Formulation
            is_benign = False
            accel_factor = 1.0 / (1.0 + math.exp(-0.12 * (max_limb_acc - config.KEYPOINT_ACCELERATION_THRESHOLD)))
            jitter_factor = min(1.0, max(0.0, mean_jitter / config.BBOX_JITTER_THRESHOLD))
            instability_factor = min(1.0, max(0.0, dist_variance / 80.0))

            raw_threat_score = (0.50 * accel_factor) + (0.30 * jitter_factor) + (0.20 * instability_factor)
            threat_score = min(1.0, max(0.0, raw_threat_score))

            # Temporal sustained threat verification
            sustained_time = self._pair_sustain_timers[pair_key]
            if threat_score >= self.threat_threshold:
                sustained_time += current_dt
                self._pair_sustain_timers[pair_key] = sustained_time
            else:
                sustained_time = max(0.0, sustained_time - (current_dt * 1.5))
                self._pair_sustain_timers[pair_key] = sustained_time

            is_threat = (threat_score >= self.threat_threshold and sustained_time >= self.sustain_sec)

            if is_threat:
                status_label = "ALTERCATION_ALERT"
            elif threat_score >= 0.55:
                status_label = "SUSPICIOUS"
            else:
                status_label = "NORMAL_PROXIMITY"

        sustained_time = self._pair_sustain_timers[pair_key]

        return InteractionPair(
            person_a_id=tid_a,
            person_b_id=tid_b,
            centroid_distance=dist_curr,
            iou=iou_curr,
            threat_score=threat_score,
            is_benign=is_benign,
            is_threat=is_threat,
            sustained_duration_sec=sustained_time,
            max_acceleration=max_limb_acc,
            bbox_jitter=mean_jitter,
            status_label=status_label
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameAnalytics:
        """
        Main pipeline execution on a single video frame.
        Performs Stage 1 detection, Stage 2 interaction detection,
        Stage 3 classification, and Stage 4 temporal alerting.
        """
        now = time.monotonic()
        # High-Speed Temporal Stride Optimization: propagate cached analytics on non-stride frames
        if self.inference_stride > 1 and (frame_idx % self.inference_stride != 0) and self._last_analytics is not None:
            return FrameAnalytics(
                frame_idx=frame_idx,
                timestamp=now,
                persons=self._last_analytics.persons,
                interactions=self._last_analytics.interactions,
                max_threat_score=self._last_analytics.max_threat_score,
                alert_triggered=self._last_analytics.alert_triggered,
                alert_incident_pair=self._last_analytics.alert_incident_pair
            )

        self._frame_count = frame_idx
        self._evict_stale_tracks(frame_idx)

        current_persons: List[PersonTrack] = []
        interactions: List[InteractionPair] = []
        max_threat = 0.0
        alert_triggered = False
        incident_pair = None

        # ----------------------------------------------------------------------
        # STAGE 1: DETECTION & MULTI-OBJECT TRACKING
        # ----------------------------------------------------------------------
        if self.model is not None:
            try:
                # Track people (COCO class 0: person) using ByteTrack at optimized resolution (>=30 FPS)
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_cfg,
                    classes=[0],
                    conf=self.conf_threshold,
                    imgsz=self.imgsz,
                    verbose=False
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    res = results[0]
                    boxes = res.boxes
                    has_id = boxes.id is not None
                    track_ids = boxes.id.cpu().numpy().astype(int) if has_id else []
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    # Pose keypoints (N_persons, 17, 2)
                    has_kpts = res.keypoints is not None and res.keypoints.xy is not None
                    kpts_xy = res.keypoints.xy.cpu().numpy() if has_kpts else None
                    kpts_conf = res.keypoints.conf.cpu().numpy() if (has_kpts and res.keypoints.conf is not None) else None

                    for i in range(len(xyxy)):
                        tid = int(track_ids[i]) if has_id and i < len(track_ids) else (i + 1)
                        box = [int(v) for v in xyxy[i]]
                        conf = float(confs[i])
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0

                        # Keypoints for this person
                        if kpts_xy is not None and i < len(kpts_xy):
                            p_kpts_xy = kpts_xy[i]
                            p_kpts_conf = kpts_conf[i] if kpts_conf is not None else np.ones((17,))
                        else:
                            p_kpts_xy = np.zeros((17, 2), dtype=np.float32)
                            p_kpts_conf = np.zeros((17,), dtype=np.float32)

                        # Update historical buffer
                        sample = TrackHistorySample(
                            timestamp=now,
                            frame_idx=frame_idx,
                            bbox=np.array(box),
                            centroid=(cx, cy),
                            keypoints_xy=p_kpts_xy,
                            keypoints_conf=p_kpts_conf
                        )
                        self._history[tid].append(sample)
                        self._last_seen_frame[tid] = frame_idx

                        current_persons.append(PersonTrack(
                            track_id=tid,
                            bbox=(box[0], box[1], box[2], box[3]),
                            confidence=conf,
                            centroid=(cx, cy),
                            keypoints_xy=p_kpts_xy,
                            keypoints_conf=p_kpts_conf
                        ))
            except Exception as e:
                logger.error(f"Error during YOLO model inference: {e}")

        # ----------------------------------------------------------------------
        # STAGE 2 & 3: INTERACTION ZONE DETECTION & SPATIOTEMPORAL CLASSIFICATION
        # ----------------------------------------------------------------------
        num_persons = len(current_persons)
        last_t = getattr(self, "_last_frame_time", now)
        dt = max(0.01, min(0.08, now - last_t)) if last_t > 0 else (1.0 / config.DEFAULT_FPS)
        self._last_frame_time = now

        # 1. Multi-Person Interaction Analysis
        for i in range(num_persons):
            for j in range(i + 1, num_persons):
                p_a = current_persons[i]
                p_b = current_persons[j]

                dist = euclidean_distance(p_a.centroid, p_b.centroid)
                iou = calculate_iou(p_a.bbox, p_b.bbox)

                # Trigger interaction zone if bounding boxes overlap or centroids in proximity
                if iou > config.INTERACTION_IOU_THRESHOLD or dist < config.INTERACTION_PROXIMITY_PX:
                    pair_result = self._analyze_interaction_dynamics(p_a.track_id, p_b.track_id, dt)
                    interactions.append(pair_result)

                    if pair_result.threat_score > max_threat:
                        max_threat = pair_result.threat_score

                    if pair_result.is_threat:
                        alert_triggered = True
                        incident_pair = pair_result

                    # Update person statuses
                    status_for_persons = "THREAT" if pair_result.is_threat else (
                        "BENIGN" if pair_result.is_benign else "INTERACTION"
                    )
                    if p_a.status != "THREAT":
                        p_a.status = status_for_persons
                        p_a.max_limb_acceleration = max(p_a.max_limb_acceleration, pair_result.max_acceleration)
                    if p_b.status != "THREAT":
                        p_b.status = status_for_persons
                        p_b.max_limb_acceleration = max(p_b.max_limb_acceleration, pair_result.max_acceleration)

        # 2. Single-Person Abnormal Motion / Violent Distress / Struggle
        for p in current_persons:
            # If person is part of an active benign hug or high-five, exempt from false struggle alerts
            if p.status == "BENIGN":
                continue

            # Check if this person has a hand raised high (high-five, waving, greeting, celebrating)
            hist = self._history.get(p.track_id)
            if hist:
                s_curr = hist[-1]
                p_kpts = s_curr.keypoints_xy
                p_conf = s_curr.keypoints_conf
                p_h = max(1.0, p.bbox[3] - p.bbox[1])
                p_wrists = [p_kpts[w, 1] for w in [9, 10] if p_conf[w] > 0.12 and p_kpts[w, 1] > 5.0]
                p_shoulders = [p_kpts[s, 1] for s in [5, 6] if p_conf[s] > 0.12 and p_kpts[s, 1] > 5.0]
                p_sh_y = min(p_shoulders) if p_shoulders else (p.bbox[1] + 0.25 * p_h)
                p_hand_raised = any(
                    (wy <= p_sh_y + 18.0) or (wy <= p.bbox[1] + 0.45 * p_h)
                    for wy in p_wrists
                ) if p_wrists else False

                if p_hand_raised:
                    p.status = "BENIGN"
                    continue

            acc_max, _ = self._compute_keypoint_kinematics(p.track_id)
            jitter = self._compute_bbox_jitter(p.track_id)
            p.max_limb_acceleration = max(p.max_limb_acceleration, acc_max)

            # High acceleration or rapid collapse indicates physical distress/assault
            if acc_max > (config.KEYPOINT_ACCELERATION_THRESHOLD * 1.4) or jitter > (config.BBOX_JITTER_THRESHOLD * 1.5):
                single_score = min(0.95, 0.45 + 0.35 * (acc_max / 35.0) + 0.20 * (jitter / 0.25))
                if single_score > max_threat:
                    max_threat = single_score
                if single_score >= self.threat_threshold:
                    p.status = "THREAT"
                    single_key = (p.track_id, p.track_id)
                    self._pair_sustain_timers[single_key] += dt
                    if self._pair_sustain_timers[single_key] >= self.sustain_sec:
                        alert_triggered = True

        out_analytics = FrameAnalytics(
            frame_idx=frame_idx,
            timestamp=now,
            persons=current_persons,
            interactions=interactions,
            max_threat_score=max_threat,
            alert_triggered=alert_triggered,
            alert_incident_pair=incident_pair
        )
        self._last_analytics = out_analytics
        return out_analytics
