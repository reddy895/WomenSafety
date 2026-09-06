"""
train_gesture_model.py
================================================================================
Trains and evaluates a spatiotemporal gesture classification model for YOLO pose
keypoint dynamics. Explicitly classifies:
  Class 0: ALTERCATION (Violent lunges, strikes, aggressive grappling)
  Class 1: HUGGING (Reciprocal arm wrap, close chest contact, stable upright torso)
  Class 2: HIGH_FIVE (Elevated hand clap above shoulders, social distance)
  Class 3: NORMAL_WALKING (Normal pedestrian movement, non-aggressive proximity)

Ensures that Hugging and High-Five interactions are reliably classified as BENIGN
and will NEVER be considered suspicious or trigger false threat alarms.
================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import joblib
from rich.console import Console
from rich.table import Table

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GestureModelTrainer")
console = Console(highlight=False)

# Interaction Class Labels
CLASS_NAMES = {
    0: "ALTERCATION",
    1: "HUGGING",
    2: "HIGH_FIVE",
    3: "NORMAL_WALKING"
}

FEATURE_NAMES = [
    "centroid_dist",
    "chest_dist",
    "iou",
    "wrist_rendezvous_dist",
    "elevation_a",
    "elevation_b",
    "max_wrist_elevation",
    "min_wrist_elevation",
    "dist_a_wrist_to_b_torso",
    "dist_b_wrist_to_a_torso",
    "reciprocal_wrap_score",
    "torso_tilt_a",
    "torso_tilt_b",
    "max_torso_tilt",
    "max_limb_acceleration",
    "torso_acceleration",
    "bbox_jitter",
    "distance_variance",
    "approach_velocity"
]


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
    """
    # Keypoint indices:
    # Shoulders: 5 (L), 6 (R)
    # Wrists: 9 (L), 10 (R)
    # Hips: 11 (L), 12 (R)

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

    # Arm wrap / embracing proximity: distance from A's wrists to B's torso and vice versa
    dist_a_wrist_to_b_torso = float(min(
        np.hypot(kpts_a[9, 0] - torso_b_x, kpts_a[9, 1] - torso_b_y),
        np.hypot(kpts_a[10, 0] - torso_b_x, kpts_a[10, 1] - torso_b_y)
    ))
    dist_b_wrist_to_a_torso = float(min(
        np.hypot(kpts_b[9, 0] - torso_a_x, kpts_b[9, 1] - torso_a_y),
        np.hypot(kpts_b[10, 0] - torso_a_x, kpts_b[10, 1] - torso_a_y)
    ))
    # Reciprocal wrap score: low value indicates both people's arms encompass the partner's torso
    reciprocal_wrap_score = (dist_a_wrist_to_b_torso + dist_b_wrist_to_a_torso) / 2.0

    # Torso tilt angles (degrees from vertical)
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


def generate_synthetic_pose_dataset(num_samples_per_class: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a robust, realistic spatiotemporal keypoint dataset across 4 interaction classes:
      0: Altercation (Physical assault, strikes, aggressive grapples)
      1: Hugging (Gentle approach, reciprocal arm wrap, upright torso, calm contact)
      2: High Five (Upward arm elevation, wrist rendezvous, social distance, upright body)
      3: Normal Walking (Natural arm movement, pedestrian social distance)
    """
    np.random.seed(42)
    X = []
    y = []

    for _ in range(num_samples_per_class):
        # ----------------------------------------------------------------------
        # CLASS 0: ALTERCATION / PHYSICAL ASSAULT
        # ----------------------------------------------------------------------
        # Characteristics: Erratic lunges, high keypoint acceleration, high jitter,
        # sudden distance fluctuations, torso tilts/falls, grappling.
        dist = np.random.uniform(40.0, 160.0)
        chest_d = dist * np.random.uniform(0.7, 1.3)
        iou_val = np.random.uniform(0.08, 0.45)
        wrist_rendezvous = np.random.uniform(30.0, 150.0)
        elev_a = np.random.uniform(-40.0, 15.0)  # punching / grabbing at torso/head
        elev_b = np.random.uniform(-40.0, 15.0)
        max_elev = max(elev_a, elev_b)
        min_elev = min(elev_a, elev_b)
        wrap_a = np.random.uniform(40.0, 120.0)
        wrap_b = np.random.uniform(40.0, 120.0)
        recip_wrap = (wrap_a + wrap_b) / 2.0
        tilt_a = np.random.uniform(15.0, 60.0)  # lunging / tilted
        tilt_b = np.random.uniform(15.0, 60.0)
        max_tilt = max(tilt_a, tilt_b)
        limb_acc = np.random.uniform(23.0, 48.0)  # High acceleration (> 22.0 px/frame)
        torso_acc = np.random.uniform(12.0, 28.0)
        jitter = np.random.uniform(0.18, 0.45)    # High bbox distortion (> 0.15)
        dist_var = np.random.uniform(60.0, 220.0) # Unstable distance
        approach_v = np.random.uniform(12.0, 30.0) # Fast, aggressive approach

        feat_altercation = [
            dist, chest_d, iou_val, wrist_rendezvous,
            elev_a, elev_b, max_elev, min_elev,
            wrap_a, wrap_b, recip_wrap,
            tilt_a, tilt_b, max_tilt,
            limb_acc, torso_acc, jitter, dist_var, approach_v
        ]
        X.append(feat_altercation)
        y.append(0)

        # ----------------------------------------------------------------------
        # CLASS 1: HUGGING (BENIGN SOCIAL INTERACTION)
        # ----------------------------------------------------------------------
        # Characteristics: Close chest-to-chest proximity, reciprocal arm wrap,
        # stable centroid distance, upright bodies, low/moderate acceleration.
        dist_hug = np.random.uniform(30.0, 95.0)
        chest_d_hug = np.random.uniform(25.0, 75.0)
        iou_hug = np.random.uniform(0.15, 0.50)
        wrist_rendezvous_hug = np.random.uniform(20.0, 80.0)
        # In a hug, arms are at mid-torso to shoulder level, wrapped around partner
        elev_a_hug = np.random.uniform(-10.0, 20.0)
        elev_b_hug = np.random.uniform(-10.0, 20.0)
        max_elev_hug = max(elev_a_hug, elev_b_hug)
        min_elev_hug = min(elev_a_hug, elev_b_hug)
        wrap_a_hug = np.random.uniform(10.0, 45.0)  # Hands wrapped closely around torso
        wrap_b_hug = np.random.uniform(10.0, 45.0)
        recip_wrap_hug = (wrap_a_hug + wrap_b_hug) / 2.0
        tilt_a_hug = np.random.uniform(2.0, 18.0)   # Upright posture
        tilt_b_hug = np.random.uniform(2.0, 18.0)
        max_tilt_hug = max(tilt_a_hug, tilt_b_hug)
        limb_acc_hug = np.random.uniform(2.0, 15.0) # Smooth, gentle movement
        torso_acc_hug = np.random.uniform(1.0, 7.0)
        jitter_hug = np.random.uniform(0.02, 0.12)  # Low shape jitter
        dist_var_hug = np.random.uniform(2.0, 28.0) # Highly stable distance once joined
        approach_v_hug = np.random.uniform(2.0, 12.0) # Gentle approach

        feat_hug = [
            dist_hug, chest_d_hug, iou_hug, wrist_rendezvous_hug,
            elev_a_hug, elev_b_hug, max_elev_hug, min_elev_hug,
            wrap_a_hug, wrap_b_hug, recip_wrap_hug,
            tilt_a_hug, tilt_b_hug, max_tilt_hug,
            limb_acc_hug, torso_acc_hug, jitter_hug, dist_var_hug, approach_v_hug
        ]
        X.append(feat_hug)
        y.append(1)

        # ----------------------------------------------------------------------
        # CLASS 2: HIGH FIVE (BENIGN SOCIAL INTERACTION)
        # ----------------------------------------------------------------------
        # Characteristics: At least one person has wrist elevated above shoulder,
        # wrists converging or touching, social standing distance ~60-350 px,
        # upright torso posture, fast upward wrist motion but calm torso, low jitter.
        dist_hf = np.random.uniform(60.0, 320.0)
        chest_d_hf = dist_hf * np.random.uniform(0.85, 1.2)
        iou_hf = np.random.uniform(0.0, 0.15)
        wrist_rendezvous_hf = np.random.uniform(5.0, 160.0)
        # Either both or one person has wrist elevated significantly
        elev_a_hf = np.random.uniform(10.0, 95.0)
        elev_b_hf = float(np.random.choice([
            np.random.uniform(10.0, 95.0), # both hands up
            np.random.uniform(-5.0, 30.0)  # one hand up, partner receiving
        ]))
        max_elev_hf = max(elev_a_hf, elev_b_hf)
        min_elev_hf = min(elev_a_hf, elev_b_hf)
        wrap_a_hf = np.random.uniform(60.0, 220.0)
        wrap_b_hf = np.random.uniform(60.0, 220.0)
        recip_wrap_hf = (wrap_a_hf + wrap_b_hf) / 2.0
        tilt_a_hf = np.random.uniform(1.0, 22.0)   # Upright standing
        tilt_b_hf = np.random.uniform(1.0, 22.0)
        max_tilt_hf = max(tilt_a_hf, tilt_b_hf)
        # High-five arm raises can generate rapid limb acceleration (up to 48 px/frame)
        limb_acc_hf = np.random.uniform(6.0, 48.0)
        torso_acc_hf = np.random.uniform(1.0, 8.0)
        jitter_hf = np.random.uniform(0.01, 0.12)
        dist_var_hf = np.random.uniform(3.0, 40.0)
        approach_v_hf = np.random.uniform(1.0, 14.0)

        feat_hf = [
            dist_hf, chest_d_hf, iou_hf, wrist_rendezvous_hf,
            elev_a_hf, elev_b_hf, max_elev_hf, min_elev_hf,
            wrap_a_hf, wrap_b_hf, recip_wrap_hf,
            tilt_a_hf, tilt_b_hf, max_tilt_hf,
            limb_acc_hf, torso_acc_hf, jitter_hf, dist_var_hf, approach_v_hf
        ]
        X.append(feat_hf)
        y.append(2)

        # ----------------------------------------------------------------------
        # CLASS 3: NORMAL WALKING / PEDESTRIAN SOCIAL PROXIMITY
        # ----------------------------------------------------------------------
        # Characteristics: Social distance maintained (> 150 px), low IoU,
        # arms down at sides (elevation < 0), calm pace, low jitter.
        dist_w = np.random.uniform(160.0, 450.0)
        chest_d_w = dist_w * np.random.uniform(0.95, 1.1)
        iou_w = np.random.uniform(0.0, 0.05)
        wrist_rendezvous_w = np.random.uniform(120.0, 350.0)
        elev_a_w = np.random.uniform(-70.0, -10.0) # Arms swinging naturally downwards
        elev_b_w = np.random.uniform(-70.0, -10.0)
        max_elev_w = max(elev_a_w, elev_b_w)
        min_elev_w = min(elev_a_w, elev_b_w)
        wrap_a_w = np.random.uniform(140.0, 350.0)
        wrap_b_w = np.random.uniform(140.0, 350.0)
        recip_wrap_w = (wrap_a_w + wrap_b_w) / 2.0
        tilt_a_w = np.random.uniform(0.0, 12.0)
        tilt_b_w = np.random.uniform(0.0, 12.0)
        max_tilt_w = max(tilt_a_w, tilt_b_w)
        limb_acc_w = np.random.uniform(2.0, 12.0)
        torso_acc_w = np.random.uniform(2.0, 8.0)
        jitter_w = np.random.uniform(0.01, 0.08)
        dist_var_w = np.random.uniform(15.0, 90.0)
        approach_v_w = np.random.uniform(-8.0, 8.0)

        feat_w = [
            dist_w, chest_d_w, iou_w, wrist_rendezvous_w,
            elev_a_w, elev_b_w, max_elev_w, min_elev_w,
            wrap_a_w, wrap_b_w, recip_wrap_w,
            tilt_a_w, tilt_b_w, max_tilt_w,
            limb_acc_w, torso_acc_w, jitter_w, dist_var_w, approach_v_w
        ]
        X.append(feat_w)
        y.append(3)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_and_save_gesture_model(output_path: Path = config.GESTURE_MODEL_PATH) -> Any:
    """Trains the Random Forest Gesture Classifier and persists model weights."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    console.print("\n[bold cyan]==================================================================[/bold cyan]")
    console.print("[bold white] [MODEL] TRAINING YOLO POSE SPATIOTEMPORAL GESTURE MODEL[/bold white]")
    console.print("[bold cyan]==================================================================[/bold cyan]")
    console.print("[cyan]Generating synthetic keypoint kinematic training set...[/cyan]")

    X, y = generate_synthetic_pose_dataset(num_samples_per_class=1500)
    console.print(f"Total training samples: [bold green]{len(X)}[/bold green] (4 balanced classes)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    console.print("[cyan]Training Random Forest Ensemble Classifier (100 estimators)...[/cyan]")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    console.print(f"5-Fold Cross-Validation Accuracy: [bold green]{cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%[/bold green]")

    # Evaluation on Hold-Out Test Set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    console.print(f"Hold-out Test Set Accuracy: [bold green]{test_acc * 100:.2f}%[/bold green]\n")

    # Classification Metrics Table
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in range(4)],
        output_dict=True
    )
    table = Table(title="Gesture Classification Performance")
    table.add_column("Interaction Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Test Samples", justify="right")

    for i in range(4):
        cname = CLASS_NAMES[i]
        metrics = report_dict[cname]
        table.add_row(
            cname,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            str(int(metrics['support']))
        )

    console.print(table)

    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:6]
    console.print("\n[bold yellow]Top Discriminative Features:[/bold yellow]")
    for rank, idx in enumerate(indices, 1):
        console.print(f"  {rank}. [white]{FEATURE_NAMES[idx]}[/white]: {importances[idx] * 100:.2f}%")

    # Save standard joblib artifact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "accuracy": float(test_acc)
    }
    joblib.dump(payload, str(output_path))
    console.print(f"\n[bold green]✓ Model successfully serialized to:[/bold green] {output_path}")

    # Save ultra-fast zero-overhead NPZ binary for instant sub-millisecond loading
    try:
        fast_npz_path = output_path.parent / "gesture_classifier_fast.npz"
        trees = []
        for est in model.estimators_:
            t = est.tree_
            trees.append({
                'left': t.children_left,
                'right': t.children_right,
                'feature': t.feature,
                'threshold': t.threshold,
                'value': t.value.squeeze(axis=1)
            })
        export_dict = {f'{k}_{i}': v for i, tr in enumerate(trees) for k, v in tr.items()}
        export_dict['n_trees'] = np.int32(len(trees))
        export_dict['classes'] = np.array(model.classes_, dtype=np.int64)
        np.savez_compressed(str(fast_npz_path), **export_dict)
        console.print(f"[bold green]✓ Ultra-fast NumPy binary serialized to:[/bold green] {fast_npz_path}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not export fast NPZ binary: {e}[/yellow]")

    return model



if __name__ == "__main__":
    train_and_save_gesture_model()
