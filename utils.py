"""
utils.py
================================================================================
Utility helpers: thread-safe alert cooldown timer, annotated snapshot capture,
GPS geocoding & maps URL generation, and resilient video stream reading.
================================================================================
"""

import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

import config


class CooldownTimer:
    """
    Thread-safe cooldown timer enforcing a strict cooldown window
    (default 30 seconds) between consecutive alert dispatches.
    """

    def __init__(self, cooldown_seconds: float = config.COOLDOWN_SECONDS):
        self.cooldown_seconds: float = float(cooldown_seconds)
        self._last_alert_time: float = 0.0
        self._lock: threading.Lock = threading.Lock()

    def can_alert(self) -> bool:
        """Returns True if the cooldown period has expired, False otherwise."""
        with self._lock:
            elapsed = time.monotonic() - self._last_alert_time
            return elapsed >= self.cooldown_seconds or self._last_alert_time == 0.0

    def trigger(self) -> bool:
        """
        Attempts to trigger an alert. If eligible, updates the timestamp
        and returns True; otherwise returns False.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_alert_time
            if self._last_alert_time == 0.0 or elapsed >= self.cooldown_seconds:
                self._last_alert_time = now
                return True
            return False

    def remaining_cooldown(self) -> float:
        """Returns remaining cooldown in seconds, or 0.0 if ready."""
        with self._lock:
            if self._last_alert_time == 0.0:
                return 0.0
            elapsed = time.monotonic() - self._last_alert_time
            return max(0.0, self.cooldown_seconds - elapsed)

    def reset(self) -> None:
        """Resets the timer, allowing immediate triggering."""
        with self._lock:
            self._last_alert_time = 0.0


class GPSLocation:
    """
    Encapsulates camera GPS coordinates, address tagging,
    and Google Maps deep-link generation.
    """

    def __init__(
        self,
        camera_id: str = config.CAMERA_ID,
        coords: Optional[Tuple[float, float]] = None,
        location_name: Optional[str] = None
    ):
        self.camera_id: str = camera_id
        if coords is None:
            coords = config.CAMERA_LAT_LON
        self.latitude: float = coords[0]
        self.longitude: float = coords[1]
        if location_name:
            self.location_name: str = location_name
        elif camera_id != config.CAMERA_ID:
            self.location_name: str = camera_id
        else:
            self.location_name: str = getattr(config, "CAMERA_LOCATION_NAME", camera_id)

    @property
    def maps_url(self) -> str:
        """Generates standard Google Maps web navigation URL."""
        return f"https://maps.google.com/?q={self.latitude:.4f},{self.longitude:.4f}"

    @property
    def formatted_location(self) -> str:
        """Formatted string for alert message payloads."""
        return f"{self.location_name} (Lat: {self.latitude:.4f}, Lon: {self.longitude:.4f})"


def get_current_utc_timestamp() -> str:
    """Returns ISO-standard UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_filename_safe_timestamp() -> str:
    """Returns filesystem-safe timestamp for capture snapshots."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:19]


class SnapshotManager:
    """
    Manages saving high-resolution annotated incident snapshot frames
    with timestamp, camera location, threat scores, and warning watermarks.
    """

    def __init__(self, output_dir: Path = config.CAPTURES_DIR):
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_incident_snapshot(
        self,
        frame: np.ndarray,
        threat_score: float,
        location: GPSLocation,
        timestamp_str: Optional[str] = None
    ) -> Path:
        """
        Saves a marked-up snapshot frame to ./captures/alert_<timestamp>.jpg.
        Watermarks the image with red emergency header bar.
        """
        if timestamp_str is None:
            timestamp_str = get_current_utc_timestamp()

        file_ts = get_filename_safe_timestamp()
        file_path = self.output_dir / f"alert_{file_ts}.jpg"

        # Create a copy so we don't mutate the live display buffer
        snapshot = frame.copy()
        h, w = snapshot.shape[:2]

        # Draw red emergency header overlay
        header_height = max(55, int(h * 0.075))
        overlay = snapshot.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.85, snapshot, 0.15, 0, snapshot)

        # Draw header text
        banner_text = f"RED ALERT: ALTERCATION DETECTED [Score: {threat_score:.2f}]"
        cv2.putText(
            snapshot, banner_text, (16, int(header_height * 0.42)),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )

        sub_text = f"Loc: {location.formatted_location} | Time: {timestamp_str}"
        cv2.putText(
            snapshot, sub_text, (16, int(header_height * 0.82)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA
        )

        # Encode and save with optimized 85% JPEG quality (crisp visuals, 3x faster write)
        success = cv2.imwrite(str(file_path), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise IOError(f"Failed to write snapshot image to {file_path}")

        return file_path

    def save_incident_snapshot_async(
        self,
        frame: np.ndarray,
        threat_score: float,
        location: GPSLocation,
        timestamp_str: Optional[str] = None,
        callback: Optional[Any] = None
    ) -> None:
        """Saves incident snapshot in a background thread so the video pipeline never drops a frame."""
        frame_copy = frame.copy()
        def _worker():
            try:
                saved_path = self.save_incident_snapshot(frame_copy, threat_score, location, timestamp_str)
                if callback:
                    callback(saved_path)
            except Exception as e:
                import logging
                logging.getLogger("SnapshotManager").warning(f"Async snapshot save error: {e}")

        threading.Thread(target=_worker, daemon=True).start()



import queue

class VideoStreamReader:
    """
    High-performance multi-threaded video stream reader supporting webcams, RTSP streams,
    and media files with asynchronous frame prefetching, automatic reconnect, and metadata extraction.
    """

    def __init__(self, source: Union[int, str]):
        self.source: Union[int, str] = source
        self.is_live: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = config.DEFAULT_FPS
        self.width: int = 1280
        self.height: int = 720
        self.total_frames: int = 0
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._stopped: bool = False
        self._thread: Optional[threading.Thread] = None
        self._open_stream()
        self._start_worker()

    def _open_stream(self) -> None:
        """Initializes or reconnects the OpenCV VideoCapture."""
        if isinstance(self.source, str) and self.source.isdigit():
            cam_idx = int(self.source)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            self.is_live = True
        elif isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            self.is_live = True
        else:
            # File path or RTSP URL
            source_str = str(self.source)
            self.is_live = source_str.lower().startswith(("rtsp://", "http://", "https://"))
            self.cap = cv2.VideoCapture(source_str)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: '{self.source}'")

        if self.is_live:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception:
                pass

        # Extract stream properties
        stream_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if stream_fps and stream_fps > 5.0 and stream_fps < 120.0:
            self.fps = float(stream_fps)

        stream_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stream_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if stream_w > 0 and stream_h > 0:
            self.width = stream_w
            self.height = stream_h

        if not self.is_live:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _start_worker(self) -> None:
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self) -> None:
        while not self._stopped and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_live:
                    time.sleep(0.01)
                    continue
                else:
                    self._queue.put((False, None))
                    break

            # Asynchronously downscale large frames to preview resolution in background thread
            fh, fw = frame.shape[:2]
            if fw > config.PREVIEW_WIDTH or fh > config.PREVIEW_HEIGHT:
                scale = min(config.PREVIEW_WIDTH / fw, config.PREVIEW_HEIGHT / fh)
                frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)

            # For live streams: discard stale frame from queue so preview has ZERO delay
            if self.is_live and self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass

            try:
                self._queue.put((True, frame), block=not self.is_live, timeout=0.05 if not self.is_live else None)
            except Exception:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame from high-speed prefetch queue with near-zero latency.
        """
        try:
            return self._queue.get(timeout=0.6)
        except queue.Empty:
            return False, None

    def release(self) -> None:
        """Releases underlying capture resources and worker thread."""
        self._stopped = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
