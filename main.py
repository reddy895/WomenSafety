"""
main.py
================================================================================
Production-grade CLI entrypoint for CCTV Threat & Anomaly Surveillance System
specifically designed for Women's Safety.
Supports Live Webcam, RTSP IP Streams, and Prerecorded Media Files.
================================================================================
"""

import os
import sys
import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from rich.console import Console
from rich.live import Live

import config
from utils import (
    CooldownTimer,
    GPSLocation,
    SnapshotManager,
    VideoStreamReader,
    get_current_utc_timestamp
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainSurveillance")


def generate_synthetic_demo_video(output_path: Path, duration_sec: int = 13, fps: int = 25) -> Path:
    """
    Generates a synthetic MP4 test clip simulating:
    1. Seconds 0-3: Normal pedestrians walking towards each other.
    2. Seconds 3-6: High-Five gesture (arms raised above shoulders in friendly clap).
    3. Seconds 6-9: Warm Hugging gesture (close embrace, arms wrapped, stable upright bodies).
    4. Seconds 9-13: Aggressive struggle / physical altercation simulation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 1280, 720
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

    total_frames = duration_sec * fps
    console = Console(highlight=False)
    console.print(f"[bold cyan]Generating synthetic test video clip:[/bold cyan] {output_path}")

    # Simulated pedestrian positions
    p1_x, p1_y = 300.0, 360.0
    p2_x, p2_y = 980.0, 360.0

    for f_idx in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: subtle surveillance grid
        for gy in range(0, h, 60):
            cv2.line(frame, (0, gy), (w, gy), (28, 28, 32), 1)
        for gx in range(0, w, 80):
            cv2.line(frame, (gx, 0), (gx, h), (28, 28, 32), 1)

        t_sec = f_idx / float(fps)

        arm_p1_up = False
        arm_p2_up = False
        is_hug_pose = False

        if t_sec < 3.0:
            # Phase 1: Normal walking towards center
            p1_x += 2.5
            p2_x -= 2.5
            phase_label = "PHASE 1: Normal Pedestrian Walking (Benign Social Distance)"
        elif t_sec < 6.0:
            # Phase 2: High-Five! Both people raise their hands high and clap in the air
            p1_x = 570.0 + 0.2 * np.sin(f_idx * 0.1)
            p2_x = 710.0 - 0.2 * np.cos(f_idx * 0.1)
            arm_p1_up = True
            arm_p2_up = True
            phase_label = "PHASE 2: HIGH-FIVE GESTURE (Hands Raised & Clapped Above Shoulders - BENIGN)"
        elif t_sec < 9.0:
            # Phase 3: Warm Hug! People step in close, embrace with arms wrapped, stable posture
            p1_x = 615.0 + 0.15 * np.sin(f_idx * 0.1)
            p2_x = 665.0 - 0.15 * np.cos(f_idx * 0.1)
            is_hug_pose = True
            phase_label = "PHASE 3: HUGGING GESTURE (Close Warm Embrace, Stable Bodies - BENIGN)"
        else:
            # Phase 4: Aggressive altercation / struggle simulation
            jitter1 = np.random.uniform(-20.0, 20.0)
            jitter2 = np.random.uniform(-20.0, 20.0)
            p1_x = 600.0 + np.sin(f_idx * 1.6) * 12.0 + jitter1
            p1_y = 360.0 + np.cos(f_idx * 1.6) * 14.0
            p2_x = 680.0 + np.cos(f_idx * 1.6) * 12.0 + jitter2
            p2_y = 360.0 + np.sin(f_idx * 1.6) * 14.0
            phase_label = "PHASE 4: HIGH-ACCELERATION PHYSICAL ALTERCATION / ATTACK!"

        # Draw Person 1 (Blue/Cyan silhouette)
        p1_pt = (int(p1_x), int(p1_y))
        cv2.circle(frame, (p1_pt[0], p1_pt[1] - 80), 28, (220, 180, 100), -1)  # Head
        cv2.rectangle(frame, (p1_pt[0] - 30, p1_pt[1] - 50), (p1_pt[0] + 30, p1_pt[1] + 60), (180, 120, 50), -1) # Body
        cv2.line(frame, (p1_pt[0] - 20, p1_pt[1] + 60), (p1_pt[0] - 25, p1_pt[1] + 150), (150, 90, 30), 10) # L-Leg
        cv2.line(frame, (p1_pt[0] + 20, p1_pt[1] + 60), (p1_pt[0] + 25, p1_pt[1] + 150), (150, 90, 30), 10) # R-Leg

        # Draw Person 2 (Magenta/Pink silhouette)
        p2_pt = (int(p2_x), int(p2_y))
        cv2.circle(frame, (p2_pt[0], p2_pt[1] - 80), 28, (200, 120, 220), -1)
        cv2.rectangle(frame, (p2_pt[0] - 30, p2_pt[1] - 50), (p2_pt[0] + 30, p2_pt[1] + 60), (160, 60, 180), -1)
        cv2.line(frame, (p2_pt[0] - 20, p2_pt[1] + 60), (p2_pt[0] - 25, p2_pt[1] + 150), (120, 40, 140), 10)
        cv2.line(frame, (p2_pt[0] + 20, p2_pt[1] + 60), (p2_pt[0] + 25, p2_pt[1] + 150), (120, 40, 140), 10)

        # Draw Arms based on gesture
        if arm_p1_up and arm_p2_up:
            # High-Five arms: raised up reaching center clap
            cv2.line(frame, (p1_pt[0] + 25, p1_pt[1] - 35), (640, p1_pt[1] - 125), (220, 220, 60), 8) # P1 right arm raised high
            cv2.line(frame, (p2_pt[0] - 25, p2_pt[1] - 35), (640, p2_pt[1] - 125), (220, 220, 60), 8) # P2 left arm raised high
            cv2.circle(frame, (640, p1_pt[1] - 125), 14, (0, 255, 255), -1) # High-five contact spark!
        elif is_hug_pose:
            # Hug arms: wrapped around each other's waist/back
            cv2.line(frame, (p1_pt[0] + 15, p1_pt[1] - 20), (p2_pt[0] + 10, p2_pt[1] + 10), (100, 240, 100), 8)
            cv2.line(frame, (p2_pt[0] - 15, p2_pt[1] - 20), (p1_pt[0] - 10, p1_pt[1] + 10), (100, 240, 100), 8)
        else:
            # Default arms down
            cv2.line(frame, (p1_pt[0] - 30, p1_pt[1] - 30), (p1_pt[0] - 40, p1_pt[1] + 45), (150, 90, 30), 8)
            cv2.line(frame, (p1_pt[0] + 30, p1_pt[1] - 30), (p1_pt[0] + 40, p1_pt[1] + 45), (150, 90, 30), 8)
            cv2.line(frame, (p2_pt[0] - 30, p2_pt[1] - 30), (p2_pt[0] - 40, p2_pt[1] + 45), (120, 40, 140), 8)
            cv2.line(frame, (p2_pt[0] + 30, p2_pt[1] - 30), (p2_pt[0] + 40, p2_pt[1] + 45), (120, 40, 140), 8)

        # Draw scene labels
        cv2.putText(frame, f"CCTV SURVEILLANCE CLIP | {phase_label}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {f_idx + 1}/{total_frames} ({t_sec:.1f}s)", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    console.print(f"[bold green]Synthetic test clip generated successfully at:[/bold green] {output_path}")
    return output_path


def send_whatsapp_alert(threat_score: float = 1.0, frame=None, location=None) -> None:
    """Direct UAV-style WhatsApp notification sender with exact location and map link."""
    if not config.ENABLE_WHATSAPP:
        return
    try:
        import requests
        import cv2
        import base64

        clean_to = config.WHATSAPP_TO_NUMBER.replace("+", "").replace(" ", "").strip()
        maps_link = location.maps_url if location else f"https://maps.google.com/?q={config.CAMERA_LATITUDE:.4f},{config.CAMERA_LONGITUDE:.4f}"

        msg_lines = [
            "🚨 [WOMEN SAFETY ALERT] Physical Altercation/Distress Anomaly Detected!",
            f"Threat Score: {threat_score:.2f}"
        ]
        if getattr(config, "WHATSAPP_INCLUDE_EXACT_LOCATION_NAME", False):
            loc_str = location.formatted_location if location else f"{config.CAMERA_LOCATION_NAME} (Lat: {config.CAMERA_LATITUDE:.4f}, Lon: {config.CAMERA_LONGITUDE:.4f})"
            msg_lines.append(f"📍 Location: {loc_str}")
        msg_lines.append(f"🗺️ Google Maps: {maps_link}")
        msg_body = "\n".join(msg_lines)

        payload = {
            "to": clean_to,
            "message": msg_body
        }

        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                payload["image"] = f"data:image/jpeg;base64,{img_base64}"

        response = requests.post(config.WHATSAPP_SERVER_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"\n[WHATSAPP] Alert message sent successfully to {config.WHATSAPP_TO_NUMBER}!\n[LOCATION] {maps_link}")
        else:
            print(f"\n[WHATSAPP_ERROR] Server returned status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"\n[WHATSAPP_ERROR] Failed to send WhatsApp message: {e}")


def resolve_media_source(source_val: Optional[str]) -> str:
    """
    Resolves a source input string. If the user provided a folder/directory path,
    automatically detects and returns the first/newest video file within that folder.
    """
    if not source_val:
        return "0"
    cleaned = str(source_val).strip().strip('"').strip("'")
    if cleaned.isdigit() or cleaned.startswith(("rtsp://", "http://", "https://")):
        return cleaned

    p = Path(cleaned)
    if p.exists() and p.is_dir():
        # Search for video files inside directory
        video_extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.webm")
        found = []
        for ext in video_extensions:
            found.extend(list(p.glob(ext)))
            found.extend(list(p.glob(ext.upper())))
        if found:
            # Sort newest first
            found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            chosen = str(found[0])
            console = Console(highlight=False)
            console.print(f"[bold green]✓ Folder automatically resolved to video:[/bold green] {chosen}")
            return chosen
        else:
            console = Console(highlight=False)
            console.print(f"[bold yellow]⚠️ No video files found in folder:[/bold yellow] {p}")

    return cleaned


def prompt_interactive_source() -> str:
    """Presents a clean interactive menu for Live vs. Upload mode when no CLI source is provided."""
    console = Console()
    console.print("\n[bold cyan]┌───────────────────────────────────────────────────────────────────┐[/bold cyan]")
    console.print("[bold cyan]│[/bold cyan]                   [bold white]SELECT INPUT SOURCE MODE[/bold white]                        [bold cyan]│[/bold cyan]")
    console.print("[bold cyan]├───────────────────────────────────────────────────────────────────┤[/bold cyan]")
    console.print("[bold cyan]│[/bold cyan]  [bold cyan][1][/bold cyan] [bold white]Live Camera Stream (Webcam 0) [DEFAULT][/bold white]                       [bold cyan]│[/bold cyan]")
    console.print("[bold cyan]│[/bold cyan]  [bold cyan][2][/bold cyan] [bold white]Select Video File or Folder (Windows File Explorer)[/bold white]         [bold cyan]│[/bold cyan]")
    console.print("[bold cyan]└───────────────────────────────────────────────────────────────────┘[/bold cyan]")
    console.print(f"[dim]WhatsApp Alert Recipient:[/dim] [bold green]{config.WHATSAPP_TO_NUMBER}[/bold green] [dim](UAV WhatsApp Bot Microservice)[/dim]\n")

    try:
        choice = input("  Select option (1 or 2) [default 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "0"

    if choice == "2":
        # Mode 2: Upload Video File or Folder
        selected_file = ""
        # Try native Windows file picker dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected_file = filedialog.askopenfilename(
                title="Select CCTV/Video File (or Cancel to pick Folder)",
                filetypes=[
                    ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm"),
                    ("All Files", "*.*")
                ]
            )
            # If user canceled file picker, prompt for folder
            if not selected_file:
                selected_folder = filedialog.askdirectory(title="Or Select Folder Containing Video Files")
                if selected_folder:
                    selected_file = selected_folder
            root.update()
            root.destroy()
        except Exception:
            selected_file = ""

        if selected_file:
            resolved = resolve_media_source(selected_file)
            if os.path.exists(resolved):
                console.print(f"[bold green]✓ Media source loaded:[/bold green] {resolved}")
                console.print("[bold cyan]▶ Redirecting to live video surveillance window...[/bold cyan]\n")
                return resolved

        # Fallback to terminal input
        try:
            path_input = input("Enter video file or folder path (or press Enter for sample clip): ").strip()
        except (EOFError, KeyboardInterrupt):
            path_input = ""

        if path_input:
            resolved = resolve_media_source(path_input)
            if os.path.exists(resolved):
                console.print(f"[bold green]✓ Media source loaded:[/bold green] {resolved}")
                console.print("[bold cyan]▶ Redirecting to live video surveillance window...[/bold cyan]\n")
                return resolved

        # Default sample clip
        sample_clip = config.TEST_CLIPS_DIR / "sample_people.mp4"
        if not sample_clip.exists():
            sample_clip = config.TEST_CLIPS_DIR / "demo_altercation.mp4"
        console.print(f"[bold green]✓ Using sample video clip:[/bold green] {sample_clip}")
        console.print("[bold cyan]▶ Redirecting to live video surveillance window...[/bold cyan]\n")
        return str(sample_clip)
    else:
        # Mode 1: Live Webcam (instant default, no second prompt)
        console.print("[bold green]✓ Live Webcam selected:[/bold green] Camera #0")
        console.print("[bold cyan]▶ Redirecting to live video surveillance window...[/bold cyan]\n")
        return "0"


def parse_cli_args() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production CCTV Threat & Anomaly Surveillance System for Women's Safety.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core source argument
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video source: webcam index ('0', '1'), RTSP URL ('rtsp://...'), or video file path."
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default=config.YOLO_MODEL_NAME,
        help="Ultralytics YOLO Pose model name or path (e.g. yolov8n-pose.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.DETECTION_CONFIDENCE,
        help="Detection confidence threshold for pedestrian detection."
    )
    parser.add_argument(
        "--threat-thresh",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Threshold confidence for anomaly verification (0.0 to 1.0)."
    )
    parser.add_argument(
        "--sustain-sec",
        type=float,
        default=config.SUSTAINED_ANOMALY_DURATION_SEC,
        help="Required sustained anomaly duration (seconds) before triggering alert."
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=config.COOLDOWN_SECONDS,
        help="Alert cooldown timer in seconds."
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=config.YOLO_INFERENCE_STRIDE,
        help="Inference frame stride (1 for full detection every frame, 2 for ultra-fast 60+ FPS tracking)."
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run in headless terminal mode without rendering the OpenCV GUI preview window."
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=getattr(config, "MAX_FPS", 40.0),
        help="Maximum processing and playback speed (strictly capped at 40 FPS max)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames and exit (useful for automated testing/benchmarking)."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously loop media files when end of file is reached."
    )
    parser.add_argument(
        "--mock-alert",
        action="store_true",
        help="Immediately test the WhatsApp notification pipeline on startup."
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate a synthetic altercation test clip and exit."
    )
    parser.add_argument(
        "--test-whatsapp",
        type=str,
        nargs="?",
        const="7760489756",
        default=None,
        help="Instantly send an automated test WhatsApp message to target recipient (default: 7760489756) and exit."
    )

    args = parser.parse_args()

    # Handle immediate direct WhatsApp test if requested
    if args.test_whatsapp:
        from whatsapp_bot.sender import send_direct_message, ensure_whatsapp_bot_running
        target_num = args.test_whatsapp
        console = Console(highlight=False)
        console.print(f"\n[bold cyan][WHATSAPP] Sending automated text from {config.SENDER_NUMBER} to {target_num}...[/bold cyan]")
        ensure_whatsapp_bot_running(auto_start=True)
        res = send_direct_message(
            to_number=target_num,
            message=f"[WOMEN SAFETY ALERT]: Automated surveillance message verified. Node: {config.CAMERA_ID}. High-Five & Hug gesture safety active."
        )
        if res.success:
            console.print(f"[bold green]✓ SUCCESS! Message delivered to {target_num} without interruption (Engine: {res.engine}).[/bold green]\n")
        else:
            console.print(f"[bold red]✗ WhatsApp dispatch returned: {res.error_message}[/bold red]")
            console.print(f"If unlinked, scan QR code at: [bold cyan]{config.WHATSAPP_QR_URL}[/bold cyan]\n")
        sys.exit(0)

    # If no source argument provided on CLI and not in utility mode, run interactive prompt
    if args.source is None and not args.generate_demo and not args.test_whatsapp:
        args.source = prompt_interactive_source()

    return args


def run_surveillance_pipeline(args: argparse.Namespace) -> None:
    """Main execution loop for CCTV surveillance system."""
    console = Console(highlight=False)
    console.print("\n[bold cyan]======================================================================[/bold cyan]")
    console.print("[bold white] [SURVEILLANCE] CCTV THREAT & ANOMALY SURVEILLANCE - WOMEN'S SAFETY[/bold white]")
    console.print("[bold cyan]======================================================================[/bold cyan]\n")

    # Handle demo generator option
    if args.generate_demo:
        demo_file = config.TEST_CLIPS_DIR / "demo_altercation.mp4"
        generate_synthetic_demo_video(demo_file)
        console.print(f"[bold green]Ready! Test with:[/bold green] python main.py --source {demo_file}")
        return

    # Check source file existence if not live webcam/rtsp
    source_val = resolve_media_source(args.source)
    if not source_val.isdigit() and not source_val.startswith(("rtsp://", "http://", "https://")):
        source_path = Path(source_val)
        if not source_path.exists():
            console.print(f"[bold red]Error:[/bold red] Specified media file '{source_val}' does not exist.")
            # If sample video exists, use it
            sample_fallback = config.TEST_CLIPS_DIR / "sample_people.mp4"
            if not sample_fallback.exists():
                sample_fallback = config.TEST_CLIPS_DIR / "demo_altercation.mp4"
            console.print(f"[green]Falling back to test clip:[/green] {sample_fallback}")
            source_val = str(sample_fallback)

    # Lazy-load heavy CV & notification subsystems on demand
    from tracker import SpatiotemporalThreatEngine, FrameAnalytics
    from whatsapp_bot import WhatsAppDispatcher, AlertPayload
    from ui import TerminalDashboard, VisualRenderer

    # Initialize Subsystems
    gps_location = GPSLocation(camera_id=config.CAMERA_ID, coords=config.CAMERA_LAT_LON)
    cooldown_timer = CooldownTimer(cooldown_seconds=args.cooldown)
    snapshot_manager = SnapshotManager(output_dir=config.CAPTURES_DIR)
    terminal_ui = TerminalDashboard(console=console)
    visual_renderer = VisualRenderer(window_name=config.DISPLAY_WINDOW_NAME)

    # Initialize WhatsApp Dispatcher (Free Engine - No Twilio Required)
    dispatcher = WhatsAppDispatcher(
        recipients=config.RECIPIENT_NUMBERS,
        use_free_engine=config.USE_FREE_WHATSAPP
    )

    console.print(f"[bold green]✓ WhatsApp Dispatch Ready:[/bold green] [white]{dispatcher.recipients}[/white] [dim](Free Engine - No Twilio Required)[/dim]")

    # Trigger mock alert test if requested
    if args.mock_alert:
        console.print("[magenta]🚀 Triggering test alert to WhatsApp pipeline...[/magenta]")
        test_payload = AlertPayload(
            threat_score=0.99,
            location=gps_location,
            timestamp_str=get_current_utc_timestamp(),
            snapshot_path=None
        )
        dispatcher.dispatch_alert_async(test_payload)
        console.print("[magenta]Test alert queued to background worker.[/magenta]\n")

    # Initialize Video Stream Reader
    console.print(f"[cyan]Connecting to video source:[/cyan] {source_val}")
    try:
        stream = VideoStreamReader(source=source_val)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize video source:[/bold red] {e}")
        return

    # Initialize Computer Vision Tracking Engine
    console.print(f"[cyan]Initializing YOLO Pose Tracking Engine:[/cyan] {args.model}")
    tracker_engine = SpatiotemporalThreatEngine(
        model_name=args.model,
        conf_threshold=args.conf,
        threat_threshold=args.threat_thresh,
        sustain_sec=args.sustain_sec,
        window_size=config.SPATIOTEMPORAL_WINDOW_N,
        inference_stride=args.stride
    )

    console.print(f"[bold green]✓ Surveillance Node Active.[/bold green] Node: [white]{config.CAMERA_ID}[/white] | Cooldown: [white]{args.cooldown}s[/white]\n")

    # Graceful exit handler
    running = True

    def sigint_handler(sig, frame_signal):
        nonlocal running
        running = False
        console.print("\n[yellow]Shutdown signal received. Exiting gracefully...[/yellow]")

    signal.signal(signal.SIGINT, sigint_handler)

    # Initialize OpenCV preview window
    if not args.no_preview:
        cv2.namedWindow(config.DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.DISPLAY_WINDOW_NAME, config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)

    is_media_file = not str(source_val).isdigit() and not str(source_val).startswith(("rtsp://", "http://", "https://"))
    # Default looping for media files unless explicit max-frames specified
    should_loop = args.loop or (is_media_file and args.max_frames is None)

    # Strict maximum FPS limit (strictly capped at max 40 FPS per user requirement)
    max_fps_limit = float(getattr(args, "max_fps", getattr(config, "MAX_FPS", 40.0)))
    min_frame_time = (1.0 / max_fps_limit) if max_fps_limit > 0 else 0.0

    frame_idx = 0
    fps_smooth = min(max_fps_limit, float(stream.fps))
    t_last = time.monotonic()
    last_ui_update = 0.0

    try:
        # Use Rich Live Display for flicker-free terminal dashboard (4 Hz to minimize CPU load)
        with Live(console=console, refresh_per_second=4, transient=False) as live_display:
            while running:
                t_frame_start = time.monotonic()
                ret, frame = stream.read()
                if not ret or frame is None:
                    if stream.is_live:
                        time.sleep(0.05)
                        continue
                    elif should_loop:
                        stream.release()
                        stream = VideoStreamReader(source=source_val)
                        continue
                    else:
                        console.print("[cyan]End of video stream reached. Processing complete.[/cyan]")
                        break

                frame_idx += 1
                if args.max_frames is not None and frame_idx > args.max_frames:
                    console.print(f"[cyan]Reached requested maximum frame count ({args.max_frames}). Halting.[/cyan]")
                    break

                # 1. Run Spatiotemporal Computer Vision Tracking Pipeline
                analytics: FrameAnalytics = tracker_engine.process_frame(frame, frame_idx)

                # 2. Check Alert Dispatch Conditions
                if analytics.alert_triggered:
                    if cooldown_timer.can_alert():
                        # Trigger cooldown timer immediately
                        cooldown_timer.trigger()
                        ts_str = get_current_utc_timestamp()

                        # Clone frame for background worker so video thread never pauses or drops frames
                        frame_snapshot = frame.copy()

                        # Dispatch WhatsApp Alert via background thread with instant raw frame
                        payload = AlertPayload(
                            threat_score=analytics.max_threat_score,
                            location=gps_location,
                            timestamp_str=ts_str,
                            raw_frame=frame_snapshot
                        )
                        dispatcher.dispatch_alert_async(payload)

                        # Asynchronously persist annotated snapshot to disk in background worker
                        snapshot_manager.save_incident_snapshot_async(
                            frame=frame_snapshot,
                            threat_score=analytics.max_threat_score,
                            location=gps_location,
                            timestamp_str=ts_str
                        )

                # 3. Update Terminal Dashboard UI (throttled to 4 Hz to maximize FPS for video processing)
                if t_frame_start - last_ui_update > 0.25:
                    last_ui_update = t_frame_start
                    panel = terminal_ui.generate_dashboard(
                        source_name=str(source_val),
                        fps=fps_smooth,
                        analytics=analytics,
                        cooldown_remaining=cooldown_timer.remaining_cooldown(),
                        last_alert_time=dispatcher.last_dispatched_at,
                        last_dispatch_status=dispatcher.last_alert_status,
                        is_simulated=dispatcher.is_simulated
                    )
                    live_display.update(panel)

                # 4. Render OpenCV Preview Window (unless --no-preview)
                if not args.no_preview:
                    annotated_frame = visual_renderer.render(
                        frame=frame,
                        analytics=analytics,
                        fps=fps_smooth,
                        cooldown_remaining=cooldown_timer.remaining_cooldown(),
                        camera_label=config.CAMERA_ID
                    )

                    cv2.imshow(config.DISPLAY_WINDOW_NAME, annotated_frame)

                    # Force window directly to the foreground so the user is immediately redirected
                    if frame_idx <= 2:
                        try:
                            cv2.setWindowProperty(config.DISPLAY_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                            import ctypes
                            hwnd = ctypes.windll.user32.FindWindowW(None, config.DISPLAY_WINDOW_NAME)
                            if hwnd:
                                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                                ctypes.windll.user32.SetForegroundWindow(hwnd)
                            cv2.setWindowProperty(config.DISPLAY_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
                        except Exception:
                            pass

                # 5. Precise Frame Rate Pacing (Strictly Capped at max 40 FPS)
                elapsed_frame = time.monotonic() - t_frame_start
                delay_needed = max(0.0, min_frame_time - elapsed_frame)

                if not args.no_preview:
                    wait_ms = max(1, int(round(delay_needed * 1000.0)))
                    key = cv2.waitKey(wait_ms)
                    if (key & 0xFF) in (ord('q'), 27):  # 'q' or ESC
                        running = False
                        break
                else:
                    if delay_needed > 0:
                        time.sleep(delay_needed)

                # Update smoothed FPS (strictly capped at max_fps_limit)
                t_frame_end = time.monotonic()
                dt_frame = max(0.001, t_frame_end - t_last)
                t_last = t_frame_end
                inst_fps = min(max_fps_limit, 1.0 / dt_frame)
                fps_smooth = min(max_fps_limit, 0.85 * fps_smooth + 0.15 * inst_fps)

    except KeyboardInterrupt:
        console.print("\n[yellow]Surveillance interrupted by user.[/yellow]")
    finally:
        console.print("[cyan]Cleaning up resources...[/cyan]")
        stream.release()
        cv2.destroyAllWindows()
        dispatcher.shutdown(wait=False)
        console.print("[bold green]System terminated cleanly.[/bold green]")


def main():
    args = parse_cli_args()
    run_surveillance_pipeline(args)


if __name__ == "__main__":
    main()
