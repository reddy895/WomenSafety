"""
ui.py
================================================================================
User Interface module:
1. Terminal Dashboard: Rich telemetry dashboard in console displaying real-time
   FPS, tracked person count, threat score gauges, and alert status badges.
2. OpenCV Visual Renderer: Overlays green/yellow/red bounding boxes, pose skeletons,
   threat gauges, and high-contrast emergency HUD alert banners.
================================================================================
"""

import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live

import config
from tracker import FrameAnalytics, PersonTrack, InteractionPair

# Standard COCO 17 Keypoints Skeleton Pairs for Pose Rendering
SKELETON_EDGES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso & Shoulders
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm
    (5, 7), (7, 9),
    # Right Arm
    (6, 8), (8, 10),
    # Left Leg
    (11, 13), (13, 15),
    # Right Leg
    (12, 14), (14, 16),
]


class TerminalDashboard:
    """
    Renders real-time telemetry logs and status tables in the terminal
    using the 'rich' library.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._last_render_time: float = 0.0

    def generate_dashboard(
        self,
        source_name: str,
        fps: float,
        analytics: FrameAnalytics,
        cooldown_remaining: float,
        last_alert_time: Optional[str] = None,
        last_dispatch_status: str = "IDLE",
        is_simulated: bool = False
    ) -> Panel:
        """
        Builds a comprehensive Rich Panel layout summarizing surveillance metrics.
        """
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right", ratio=1)

        # ----------------------------------------------------------------------
        # SYSTEM HEADER
        # ----------------------------------------------------------------------
        header_text = Text()
        header_text.append("🛡️  CCTV THREAT & ANOMALY SURVEILLANCE NODE", style="bold cyan")
        header_text.append(" | ", style="dim")
        header_text.append("WOMEN'S SAFETY ENGINE", style="bold magenta")
        if is_simulated:
            header_text.append(" [DRY-RUN SIMULATION]", style="dim yellow")

        # ----------------------------------------------------------------------
        # ALERT & THREAT STATUS BADGE
        # ----------------------------------------------------------------------
        threat_val = analytics.max_threat_score
        status_text = Text()

        if analytics.alert_triggered:
            status_text.append("🚨 RED ALERT: THREAT CONFIRMED", style="bold white on red")
        elif cooldown_remaining > 0:
            status_text.append(f"⏳ COOLDOWN ACTIVE ({cooldown_remaining:.1f}s)", style="bold black on cyan")
        elif threat_val >= config.CONFIDENCE_THRESHOLD:
            status_text.append("⚠️  EVALUATING SUSTAINED THREAT...", style="bold yellow on black")
        elif len(analytics.interactions) > 0:
            status_text.append("▲ MONITORING INTERACTION", style="bold yellow")
        else:
            status_text.append("● NORMAL PEDESTRIAN STATE", style="bold green")

        # ----------------------------------------------------------------------
        # THREAT SCORE PROGRESS GAUGE
        # ----------------------------------------------------------------------
        bar_len = 24
        filled = int(round(threat_val * bar_len))
        filled = max(0, min(bar_len, filled))
        empty = bar_len - filled

        if threat_val >= config.CONFIDENCE_THRESHOLD:
            gauge_color = "red"
        elif threat_val >= 0.50:
            gauge_color = "yellow"
        else:
            gauge_color = "green"

        gauge = Text()
        gauge.append("[", style="dim")
        gauge.append("█" * filled, style=f"bold {gauge_color}")
        gauge.append("░" * empty, style="dim white")
        gauge.append(f"] {threat_val:.2f} / 1.00", style=f"bold {gauge_color}")

        # ----------------------------------------------------------------------
        # METRICS GRID
        # ----------------------------------------------------------------------
        metrics_table = Table(box=None, expand=True, show_header=False)
        metrics_table.add_column("Key", style="dim cyan", ratio=1)
        metrics_table.add_column("Value", style="bold white", ratio=2)
        metrics_table.add_column("Key2", style="dim cyan", ratio=1)
        metrics_table.add_column("Value2", style="bold white", ratio=2)

        metrics_table.add_row(
            "Video Source:", str(source_name),
            "Processing Speed:", f"{fps:.1f} FPS"
        )
        metrics_table.add_row(
            "Tracked Persons:", f"{len(analytics.persons)} pedestrians",
            "Active Interactions:", f"{len(analytics.interactions)} pairs"
        )
        metrics_table.add_row(
            "Threat Gauge:", gauge,
            "Alert Status:", status_text
        )
        metrics_table.add_row(
            "Target WhatsApp:", f"{config.RECIPIENT_NUMBERS[0] if config.RECIPIENT_NUMBERS else '+917760489756'} (UAV Engine)",
            "WhatsApp Status:", str(last_dispatch_status)
        )

        # ----------------------------------------------------------------------
        # DETAILED INTERACTION TABLE (If any)
        # ----------------------------------------------------------------------
        interaction_table = None
        if analytics.interactions:
            interaction_table = Table(
                title="Active Interaction Dynamics",
                title_style="bold yellow",
                expand=True,
                show_lines=False
            )
            interaction_table.add_column("Pair", justify="center", style="cyan")
            interaction_table.add_column("Distance", justify="center", style="white")
            interaction_table.add_column("Max Accel", justify="center", style="magenta")
            interaction_table.add_column("Jitter", justify="center", style="blue")
            interaction_table.add_column("Threat Score", justify="center")
            interaction_table.add_column("Classification", justify="center")

            for pair in analytics.interactions:
                if pair.status_label == "BENIGN_HIGH_FIVE":
                    p_style = "bold cyan"
                elif pair.status_label == "BENIGN_HUG":
                    p_style = "bold green"
                elif pair.is_threat:
                    p_style = "bold red"
                elif pair.is_benign:
                    p_style = "bold green"
                else:
                    p_style = "yellow"

                interaction_table.add_row(
                    f"ID #{pair.person_a_id} <-> ID #{pair.person_b_id}",
                    f"{pair.centroid_distance:.1f}px",
                    f"{pair.max_acceleration:.1f}px/f²",
                    f"{pair.bbox_jitter:.2f}",
                    f"[{p_style}]{pair.threat_score:.2f}[/{p_style}]",
                    f"[{p_style}]{pair.status_label}[/{p_style}]"
                )

        # Assemble Master Layout
        content_table = Table.grid(expand=True)
        content_table.add_row(header_text)
        content_table.add_row(Text("─" * 70, style="dim"))
        content_table.add_row(metrics_table)
        if interaction_table:
            content_table.add_row(Text("─" * 70, style="dim"))
            content_table.add_row(interaction_table)

        border_style = "red" if analytics.alert_triggered else ("yellow" if len(analytics.interactions) > 0 else "blue")
        return Panel(
            content_table,
            border_style=border_style,
            title="[bold white]CCTV TELEMETRY DASHBOARD[/bold white]",
            subtitle="Press 'q' in video preview to terminate"
        )


class VisualRenderer:
    """
    Overlays computer vision analytics onto OpenCV video frames:
    - Bounding boxes (Green: benign, Yellow: proximity, Red: anomaly)
    - 17-Keypoint Pose Skeleton with limb acceleration heatmaps
    - Top HUD telemetry banner and flashing emergency strobe banner
    """

    COLOR_GREEN: Tuple[int, int, int] = (50, 220, 50)     # Normal
    COLOR_YELLOW: Tuple[int, int, int] = (0, 215, 255)    # Interaction
    COLOR_RED: Tuple[int, int, int] = (30, 30, 240)       # Altercation
    COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
    COLOR_CYAN: Tuple[int, int, int] = (240, 200, 0)

    def __init__(self, window_name: str = config.DISPLAY_WINDOW_NAME):
        self.window_name = window_name

    def render(
        self,
        frame: np.ndarray,
        analytics: FrameAnalytics,
        fps: float,
        cooldown_remaining: float,
        camera_label: str = config.CAMERA_ID
    ) -> np.ndarray:
        """
        Renders complete visual overlay onto the provided frame.
        """
        display_frame = frame
        h, w = display_frame.shape[:2]

        # 1. Draw Skeleton Keypoints & Limbs
        self._draw_skeletons(display_frame, analytics.persons)

        # 2. Draw Interaction Link Lines between interacting pairs
        self._draw_interactions(display_frame, analytics)

        # 3. Draw Pedestrian Bounding Boxes & IDs
        self._draw_bounding_boxes(display_frame, analytics)

        # 4. Draw Top HUD Telemetry Banner
        self._draw_hud_banner(display_frame, analytics, fps, cooldown_remaining, camera_label)

        # 5. Draw Emergency Threat Banner if alert triggered
        if analytics.alert_triggered or (analytics.max_threat_score >= config.CONFIDENCE_THRESHOLD):
            self._draw_threat_alert_banner(display_frame, analytics, h, w)

        return display_frame

    def _draw_skeletons(self, frame: np.ndarray, persons: List[PersonTrack]) -> None:
        """Draws COCO pose joints and connection bones."""
        for p in persons:
            kpts = p.keypoints_xy
            confs = p.keypoints_conf
            if kpts is None or len(kpts) < 17:
                continue

            # Determine bone color based on status and acceleration
            if p.status == "THREAT":
                bone_color = self.COLOR_RED
            elif p.max_limb_acceleration > config.KEYPOINT_ACCELERATION_THRESHOLD:
                bone_color = self.COLOR_RED
            elif p.status == "INTERACTION":
                bone_color = self.COLOR_YELLOW
            else:
                bone_color = self.COLOR_GREEN

            # Draw bones
            for idx_a, idx_b in SKELETON_EDGES:
                if confs[idx_a] > 0.35 and confs[idx_b] > 0.35:
                    pt_a = (int(kpts[idx_a][0]), int(kpts[idx_a][1]))
                    pt_b = (int(kpts[idx_b][0]), int(kpts[idx_b][1]))
                    # Thicker bones for upper limbs (arms)
                    thickness = 3 if idx_a in (5, 6, 7, 8, 9, 10) else 2
                    cv2.line(frame, pt_a, pt_b, bone_color, thickness, cv2.LINE_AA)

            # Draw keypoint joint circles
            for i in range(17):
                if confs[i] > 0.35:
                    pt = (int(kpts[i][0]), int(kpts[i][1]))
                    cv2.circle(frame, pt, 4, self.COLOR_WHITE, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 2, bone_color, -1, cv2.LINE_AA)

    def _draw_interactions(self, frame: np.ndarray, analytics: FrameAnalytics) -> None:
        """Draws connecting lines between interacting pedestrian centroids."""
        person_map = {p.track_id: p for p in analytics.persons}
        for pair in analytics.interactions:
            p_a = person_map.get(pair.person_a_id)
            p_b = person_map.get(pair.person_b_id)
            if p_a and p_b:
                pt_a = (int(p_a.centroid[0]), int(p_a.centroid[1]))
                pt_b = (int(p_b.centroid[0]), int(p_b.centroid[1]))

                if pair.status_label == "BENIGN_HIGH_FIVE":
                    color = (255, 200, 0) # Cyan/Gold
                    label = "HIGH-FIVE (BENIGN)"
                elif pair.status_label == "BENIGN_HUG":
                    color = (50, 220, 50) # Emerald Green
                    label = "HUG (BENIGN)"
                elif pair.is_threat:
                    color = self.COLOR_RED
                    label = f"ALTERCATION: {pair.threat_score:.2f}"
                elif pair.is_benign:
                    color = self.COLOR_GREEN
                    label = f"BENIGN: {pair.threat_score:.2f}"
                else:
                    color = self.COLOR_YELLOW
                    label = f"PROXIMITY: {pair.threat_score:.2f}"

                cv2.line(frame, pt_a, pt_b, color, 2, cv2.LINE_AA)

                # Midpoint distance text
                mid_x = (pt_a[0] + pt_b[0]) // 2
                mid_y = (pt_a[1] + pt_b[1]) // 2 - 8
                cv2.putText(frame, label, (mid_x - 45, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    def _draw_bounding_boxes(self, frame: np.ndarray, analytics: FrameAnalytics) -> None:
        """Draws bounding boxes with color coding based on status."""
        for p in analytics.persons:
            x1, y1, x2, y2 = p.bbox

            if p.status == "THREAT":
                color = self.COLOR_RED
                thickness = 3
            elif p.status == "INTERACTION":
                color = self.COLOR_YELLOW
                thickness = 2
            else:
                color = self.COLOR_GREEN
                thickness = 2

            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Label badge above box
            tag = f"ID #{p.track_id} [{p.status}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(tag, font, 0.5, 1)

            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 5, y1 - 4), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud_banner(
        self,
        frame: np.ndarray,
        analytics: FrameAnalytics,
        fps: float,
        cooldown_remaining: float,
        camera_label: str
    ) -> None:
        """Renders the top telemetry dashboard HUD on the video window."""
        w = frame.shape[1]
        hud_height = 42

        # High-speed slice blend for top banner
        sub = frame[0:hud_height, :]
        rect = np.full_like(sub, (20, 20, 25))
        cv2.addWeighted(rect, 0.85, sub, 0.15, 0, sub)

        # Telemetry string components
        node_txt = f"📍 {camera_label}"
        fps_txt = f"⚡ FPS: {fps:.1f}"
        ped_txt = f"👥 Tracked: {len(analytics.persons)}"
        threat_val = analytics.max_threat_score
        threat_txt = f"Threat: {threat_val:.2f}"

        if analytics.alert_triggered:
            status_txt = "STATUS: 🚨 RED ALERT"
            status_color = self.COLOR_RED
        elif cooldown_remaining > 0:
            status_txt = f"STATUS: ⏳ COOLDOWN ({cooldown_remaining:.0f}s)"
            status_color = self.COLOR_CYAN
        elif threat_val >= config.CONFIDENCE_THRESHOLD:
            status_txt = "STATUS: ⚠️ THREAT EVALUATION"
            status_color = self.COLOR_YELLOW
        else:
            status_txt = "STATUS: ● NORMAL"
            status_color = self.COLOR_GREEN

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = int(hud_height * 0.65)

        cv2.putText(frame, node_txt, (14, y_pos), font, 0.55, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, fps_txt, (int(w * 0.30), y_pos), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, ped_txt, (int(w * 0.46), y_pos), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, threat_txt, (int(w * 0.65), y_pos), font, 0.52, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (int(w * 0.80), y_pos), font, 0.52, status_color, 2, cv2.LINE_AA)

    def _draw_threat_alert_banner(
        self,
        frame: np.ndarray,
        analytics: FrameAnalytics,
        h: int,
        w: int
    ) -> None:
        """Draws flashing red alert banner across bottom of frame."""
        banner_h = 55
        y_start = h - banner_h

        # Flashing intensity based on timestamp
        flash = int((time.time() * 4) % 2)
        color = (0, 0, 220) if flash else (0, 0, 140)

        # High-speed slice blend for bottom alert banner
        sub = frame[y_start:h, :]
        rect = np.full_like(sub, color)
        cv2.addWeighted(rect, 0.90, sub, 0.10, 0, sub)

        alert_msg = f"⚠️ RED ALERT: PHYSICAL ALTERCATION DETECTED! (Threat: {analytics.max_threat_score:.2f})"
        cv2.putText(
            frame,
            alert_msg,
            (int(w * 0.12), y_start + int(banner_h * 0.65)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
