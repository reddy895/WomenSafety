"""
test_system.py
================================================================================
Comprehensive test suite verifying all functional modules of the CCTV Threat
& Anomaly Surveillance System:
- CooldownTimer thread safety & timeout logic
- GPSLocation geocoding & Google Maps URL generation
- SnapshotManager incident frame creation & watermarking
- WhatsAppDispatcher parallel dispatch & dry-run simulation
- Tracker spatiotemporal kinematics & benign vs. altercation modeling
================================================================================
"""

import time
import pytest
import numpy as np
from pathlib import Path

import config
from utils import CooldownTimer, GPSLocation, SnapshotManager, get_current_utc_timestamp
from whatsapp_bot import WhatsAppDispatcher, AlertPayload, DispatchResult
from tracker import (
    SpatiotemporalThreatEngine,
    calculate_iou,
    euclidean_distance,
    FrameAnalytics,
    PersonTrack,
    TrackHistorySample
)


def test_cooldown_timer_logic():
    """Validates strict cooldown timer enforcement."""
    timer = CooldownTimer(cooldown_seconds=0.5)
    # Initially ready
    assert timer.can_alert() is True
    assert timer.remaining_cooldown() == 0.0

    # First trigger should succeed
    assert timer.trigger() is True
    # Immediate second trigger must be blocked
    assert timer.trigger() is False
    assert timer.can_alert() is False
    assert timer.remaining_cooldown() > 0.0

    # Wait for cooldown expiration
    time.sleep(0.55)
    assert timer.can_alert() is True
    assert timer.remaining_cooldown() == 0.0
    # Next trigger allowed
    assert timer.trigger() is True


def test_gps_location_formatting():
    """Validates GPS coordinates format and Google Maps deep link."""
    gps = GPSLocation(camera_id="Camera Node 01", coords=(12.9716, 77.5946))
    assert gps.latitude == 12.9716
    assert gps.longitude == 77.5946
    assert gps.maps_url == "https://maps.google.com/?q=12.9716,77.5946"
    assert "Camera Node 01" in gps.formatted_location
    assert "12.9716" in gps.formatted_location
    assert "77.5946" in gps.formatted_location


def test_snapshot_manager(tmp_path: Path):
    """Validates saving incident snapshot with red emergency banner overlay."""
    mgr = SnapshotManager(output_dir=tmp_path)
    gps = GPSLocation(camera_id="Node 01", coords=(12.9716, 77.5946))
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    snapshot_path = mgr.save_incident_snapshot(
        frame=dummy_frame,
        threat_score=0.92,
        location=gps,
        timestamp_str="2026-09-03 13:00:00 UTC"
    )

    assert snapshot_path.exists()
    assert snapshot_path.is_file()
    assert snapshot_path.stat().st_size > 0
    assert snapshot_path.suffix.lower() in (".jpg", ".jpeg")


def test_whatsapp_dispatcher_simulation():
    """Validates asynchronous parallel dispatch to target 7760489756 via UAV WhatsApp engine."""
    recipients = ["7760489756", "+917760489756"]
    dispatcher = WhatsAppDispatcher(
        recipients=recipients,
        use_free_engine=True
    )
    # Check automatic phone number normalization
    assert "+917760489756" in dispatcher.recipients

    gps = GPSLocation(camera_id="Camera Node 01", coords=(12.9716, 77.5946))
    payload = AlertPayload(
        threat_score=0.95,
        location=gps,
        timestamp_str=get_current_utc_timestamp()
    )

    # Verify message body formatting
    body = payload.format_message()
    assert "⚠️ RED ALERT: Physical Altercation/Harassment Anomaly Detected!" in body
    assert "Camera Node 01" in body
    assert "Google Maps: https://maps.google.com/?q=12.9716,77.5946" in body

    # Mock direct message send in unit test to prevent network blocking
    import whatsapp_bot.sender
    orig_send = whatsapp_bot.sender.send_direct_message
    whatsapp_bot.sender.send_direct_message = lambda to_number, message, image_path=None, server_url=None: DispatchResult(
        recipient=to_number, success=True, engine="UAV_HEADLESS_BOT_TEST"
    )

    try:
        # Test dispatch async
        dispatcher.dispatch_alert_async(payload)
        time.sleep(0.4)  # Allow worker thread to finish

        assert dispatcher.total_dispatched == len(dispatcher.recipients)
        assert "SENT" in dispatcher.last_alert_status
    finally:
        whatsapp_bot.sender.send_direct_message = orig_send
        dispatcher.shutdown(wait=False)


def test_iou_and_distance_functions():
    """Validates IoU and Euclidean geometry calculations."""
    boxA = (10, 10, 100, 100)
    boxB = (10, 10, 100, 100)
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = (200, 200, 300, 300)
    assert calculate_iou(boxA, boxC) == 0.0

    boxD = (50, 50, 150, 150)
    iou_d = calculate_iou(boxA, boxD)
    assert 0.0 < iou_d < 1.0

    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    assert euclidean_distance(p1, p2) == 5.0


def test_tracker_spatiotemporal_benign_vs_anomaly():
    """
    Validates Stage 3 spatiotemporal classification:
    - Verifies benign interaction suppression for stable distance + smooth velocity (hug/handshake)
    - Verifies high threat score for high acceleration + erratic jitter (altercation)
    """
    engine = SpatiotemporalThreatEngine(
        model_name="yolov8n-pose.pt",
        threat_threshold=0.85,
        sustain_sec=0.2,
        window_size=30
    )

    # 1. Simulate Benign Hug:
    # 25 consecutive frames of close proximity with stable contact
    t_start = time.monotonic()
    for f in range(25):
        t = t_start + (f * 0.04)
        kpts_a = np.zeros((17, 2), dtype=np.float32)
        kpts_b = np.zeros((17, 2), dtype=np.float32)
        conf_ones = np.ones((17,), dtype=np.float32)

        # Set shoulders and hips
        kpts_a[5] = [200.0, 120.0]
        kpts_a[6] = [240.0, 120.0]
        kpts_a[11] = [205.0, 220.0]
        kpts_a[12] = [235.0, 220.0]

        kpts_b[5] = [230.0, 120.0]
        kpts_b[6] = [270.0, 120.0]
        kpts_b[11] = [235.0, 220.0]
        kpts_b[12] = [265.0, 220.0]

        # Arms wrapped around partner's torso
        kpts_a[9] = [250.0, 160.0]
        kpts_a[10] = [255.0, 160.0]
        kpts_b[9] = [215.0, 160.0]
        kpts_b[10] = [220.0, 160.0]

        s_a = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([180, 100, 260, 350]),
            centroid=(220.0, 225.0),
            keypoints_xy=kpts_a, keypoints_conf=conf_ones
        )
        s_b = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([210, 100, 290, 350]),
            centroid=(250.0, 225.0),
            keypoints_xy=kpts_b, keypoints_conf=conf_ones
        )
        engine._history[1].append(s_a)
        engine._history[2].append(s_b)

    benign_pair = engine._analyze_interaction_dynamics(1, 2, 0.04)
    assert benign_pair.is_benign is True
    assert benign_pair.threat_score == 0.0
    assert benign_pair.is_threat is False
    assert benign_pair.status_label in ("BENIGN_HUG", "BENIGN_INTERACTION")

    # 2. Simulate Violent Altercation / Physical Harassment:
    # Sudden high acceleration spikes in wrists and erratic bbox dimensions
    for f in range(25, 40):
        t = t_start + (f * 0.04)
        kpts_a = np.zeros((17, 2), dtype=np.float32)
        kpts_b = np.zeros((17, 2), dtype=np.float32)
        conf_ones = np.ones((17,), dtype=np.float32)

        osc = 140.0 if (f % 2 == 0) else -140.0
        kpts_a[9] = [200.0 + osc, 200.0]
        kpts_a[10] = [205.0 - osc, 205.0]
        kpts_b[9] = [208.0 - osc, 202.0]
        kpts_b[10] = [204.0 + osc, 206.0]

        jitter_w = 45.0 if (f % 2 == 0) else -45.0
        s_a = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([150 - jitter_w, 100, 250 + jitter_w, 350]),
            centroid=(200.0 + osc * 0.4, 225.0),
            keypoints_xy=kpts_a, keypoints_conf=conf_ones
        )
        s_b = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([180 + jitter_w, 100, 280 - jitter_w, 350]),
            centroid=(230.0 - osc * 0.4, 225.0),
            keypoints_xy=kpts_b, keypoints_conf=conf_ones
        )
        engine._history[1].append(s_a)
        engine._history[2].append(s_b)

        altercation_pair = engine._analyze_interaction_dynamics(1, 2, 0.04)

    # Threat score must be significantly elevated
    assert altercation_pair.threat_score >= 0.85
    assert altercation_pair.is_benign is False
    assert altercation_pair.is_threat is True
    assert altercation_pair.status_label == "ALTERCATION_ALERT"


def test_high_five_benign_gesture():
    """
    Validates that High-Five gestures (hands elevated above shoulders meeting in clap)
    are strictly classified as BENIGN_HIGH_FIVE with threat score 0.0.
    """
    engine = SpatiotemporalThreatEngine(threat_threshold=0.78, sustain_sec=0.2)
    t_start = time.monotonic()

    # Simulate 10 frames of High-Five
    for f in range(10):
        t = t_start + (f * 0.04)
        kpts_a = np.zeros((17, 2), dtype=np.float32)
        kpts_b = np.zeros((17, 2), dtype=np.float32)
        conf_ones = np.ones((17,), dtype=np.float32)

        # Shoulders at y=160
        kpts_a[5] = [200.0, 160.0]
        kpts_a[6] = [240.0, 160.0]
        kpts_b[5] = [320.0, 160.0]
        kpts_b[6] = [360.0, 160.0]

        # Wrists elevated above shoulders at y=80 (80 < 160 means elevated)
        kpts_a[10] = [275.0, 80.0] # Right hand up
        kpts_b[9] = [285.0, 80.0]  # Left hand up (rendezvous dist = 10 px!)

        s_a = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([180, 120, 260, 360]),
            centroid=(220.0, 240.0),
            keypoints_xy=kpts_a, keypoints_conf=conf_ones
        )
        s_b = TrackHistorySample(
            timestamp=t, frame_idx=f,
            bbox=np.array([300, 120, 380, 360]),
            centroid=(340.0, 240.0),
            keypoints_xy=kpts_b, keypoints_conf=conf_ones
        )
        engine._history[10].append(s_a)
        engine._history[20].append(s_b)

    hf_pair = engine._analyze_interaction_dynamics(10, 20, 0.04)
    assert hf_pair.is_benign is True
    assert hf_pair.threat_score == 0.0
    assert hf_pair.is_threat is False
    assert hf_pair.status_label == "BENIGN_HIGH_FIVE"

