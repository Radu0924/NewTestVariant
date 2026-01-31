"""
Radar Display Widget

PyQt6-based radar/PPI (Plan Position Indicator) display:
- Circular radar sweep visualization
- Target tracking display
- Range rings and bearing markers
- Threat level color coding
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QRadialGradient, QConicalGradient, QLinearGradient
)
import math
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RadarTarget:
    """Target to display on radar."""
    track_id: int
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float  # meters
    confidence: float
    classification: str
    threat_level: str
    velocity_azimuth: float = 0.0
    velocity_radial: float = 0.0
    timestamp: float = 0.0


class RadarDisplay(QWidget):
    """
    Radar/PPI display widget.

    Shows targets on a circular radar display with range rings,
    bearing markers, and sweep animation.
    """

    # Signals
    target_selected = pyqtSignal(int)  # track_id
    target_double_clicked = pyqtSignal(int)

    # Colors
    BACKGROUND_COLOR = QColor(10, 20, 30)
    GRID_COLOR = QColor(0, 80, 0)
    SWEEP_COLOR = QColor(0, 255, 0, 100)
    TEXT_COLOR = QColor(0, 255, 0)
    TARGET_LOW = QColor(0, 255, 100)
    TARGET_MEDIUM = QColor(255, 200, 0)
    TARGET_HIGH = QColor(255, 50, 50)

    def __init__(
        self,
        parent=None,
        max_range: float = 500.0,
        update_rate: int = 30
    ):
        """
        Initialize radar display.

        Args:
            parent: Parent widget.
            max_range: Maximum display range in meters.
            update_rate: Display update rate in Hz.
        """
        super().__init__(parent)

        self._max_range = max_range
        self._update_rate = update_rate
        self._targets: Dict[int, RadarTarget] = {}
        self._selected_target: Optional[int] = None
        self._sweep_angle = 0.0
        self._sweep_enabled = True
        self._trail_enabled = True
        self._target_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self._max_history = 50

        # Range rings
        self._num_rings = 5
        self._ring_labels = True

        # Display options
        self._show_elevation = True
        self._north_up = True
        self._center_offset = (0, 0)

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_sweep)
        self._timer.start(int(1000 / update_rate))

        # Setup widget
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

    def set_max_range(self, range_m: float) -> None:
        """Set maximum display range."""
        self._max_range = range_m
        self.update()

    def set_sweep_enabled(self, enabled: bool) -> None:
        """Enable/disable sweep animation."""
        self._sweep_enabled = enabled
        self.update()

    def set_trail_enabled(self, enabled: bool) -> None:
        """Enable/disable target trails."""
        self._trail_enabled = enabled
        self.update()

    def update_target(self, target: RadarTarget) -> None:
        """Update or add a target."""
        self._targets[target.track_id] = target

        # Update history
        if target.track_id not in self._target_history:
            self._target_history[target.track_id] = []

        history = self._target_history[target.track_id]
        history.append((target.azimuth, target.distance, time.time()))

        # Trim history
        if len(history) > self._max_history:
            self._target_history[target.track_id] = history[-self._max_history:]

        self.update()

    def remove_target(self, track_id: int) -> None:
        """Remove a target."""
        self._targets.pop(track_id, None)
        self._target_history.pop(track_id, None)
        self.update()

    def clear_targets(self) -> None:
        """Clear all targets."""
        self._targets.clear()
        self._target_history.clear()
        self.update()

    def _update_sweep(self) -> None:
        """Update sweep angle."""
        if self._sweep_enabled:
            self._sweep_angle = (self._sweep_angle + 2) % 360
            self.update()

    def _get_target_color(self, threat_level: str) -> QColor:
        """Get color for threat level."""
        if threat_level == "high":
            return self.TARGET_HIGH
        elif threat_level == "medium":
            return self.TARGET_MEDIUM
        return self.TARGET_LOW

    def _polar_to_screen(
        self,
        azimuth: float,
        distance: float,
        center: QPointF,
        radius: float
    ) -> QPointF:
        """Convert polar coordinates to screen coordinates."""
        # Normalize distance
        norm_dist = min(distance / self._max_range, 1.0) * radius

        # Convert azimuth to radians (0 = North, clockwise)
        angle_rad = math.radians(azimuth - 90)

        x = center.x() + norm_dist * math.cos(angle_rad)
        y = center.y() + norm_dist * math.sin(angle_rad)

        return QPointF(x, y)

    def paintEvent(self, event):
        """Paint the radar display."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get dimensions
        width = self.width()
        height = self.height()
        size = min(width, height)
        radius = (size - 60) / 2
        center = QPointF(width / 2, height / 2)

        # Draw background
        self._draw_background(painter, center, radius)

        # Draw range rings
        self._draw_range_rings(painter, center, radius)

        # Draw bearing markers
        self._draw_bearing_markers(painter, center, radius)

        # Draw sweep
        if self._sweep_enabled:
            self._draw_sweep(painter, center, radius)

        # Draw target trails
        if self._trail_enabled:
            self._draw_trails(painter, center, radius)

        # Draw targets
        self._draw_targets(painter, center, radius)

        # Draw info overlay
        self._draw_info_overlay(painter)

    def _draw_background(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw radar background."""
        # Fill background
        painter.fillRect(self.rect(), self.BACKGROUND_COLOR)

        # Draw circular gradient
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0, QColor(20, 40, 20))
        gradient.setColorAt(0.7, QColor(10, 30, 10))
        gradient.setColorAt(1, QColor(5, 15, 5))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius, radius)

    def _draw_range_rings(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw range rings."""
        painter.setPen(QPen(self.GRID_COLOR, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for i in range(1, self._num_rings + 1):
            ring_radius = radius * i / self._num_rings
            painter.drawEllipse(center, ring_radius, ring_radius)

            # Range label
            if self._ring_labels:
                range_val = self._max_range * i / self._num_rings
                label = f"{range_val:.0f}m"

                painter.setPen(QPen(self.TEXT_COLOR, 1))
                font = QFont("Consolas", 8)
                painter.setFont(font)

                label_pos = center + QPointF(5, -ring_radius + 12)
                painter.drawText(label_pos, label)
                painter.setPen(QPen(self.GRID_COLOR, 1))

    def _draw_bearing_markers(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw bearing markers."""
        painter.setPen(QPen(self.GRID_COLOR, 1))

        for angle in range(0, 360, 30):
            angle_rad = math.radians(angle - 90)

            inner_point = QPointF(
                center.x() + (radius * 0.95) * math.cos(angle_rad),
                center.y() + (radius * 0.95) * math.sin(angle_rad)
            )
            outer_point = QPointF(
                center.x() + radius * math.cos(angle_rad),
                center.y() + radius * math.sin(angle_rad)
            )

            painter.drawLine(inner_point, outer_point)

            # Bearing label
            if angle % 90 == 0:
                labels = {0: "N", 90: "E", 180: "S", 270: "W"}
                label = labels.get(angle, str(angle))

                painter.setPen(QPen(self.TEXT_COLOR, 1))
                font = QFont("Consolas", 10, QFont.Weight.Bold)
                painter.setFont(font)

                label_point = QPointF(
                    center.x() + (radius + 15) * math.cos(angle_rad) - 5,
                    center.y() + (radius + 15) * math.sin(angle_rad) + 5
                )
                painter.drawText(label_point, label)
                painter.setPen(QPen(self.GRID_COLOR, 1))

    def _draw_sweep(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw radar sweep."""
        # Create sweep gradient
        gradient = QConicalGradient(center, -self._sweep_angle + 90)
        gradient.setColorAt(0, QColor(0, 255, 0, 150))
        gradient.setColorAt(0.1, QColor(0, 255, 0, 50))
        gradient.setColorAt(0.2, QColor(0, 255, 0, 0))
        gradient.setColorAt(1, QColor(0, 255, 0, 0))

        # Draw sweep arc
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius, radius)

        # Draw sweep line
        angle_rad = math.radians(self._sweep_angle - 90)
        end_point = QPointF(
            center.x() + radius * math.cos(angle_rad),
            center.y() + radius * math.sin(angle_rad)
        )

        painter.setPen(QPen(QColor(0, 255, 0, 200), 2))
        painter.drawLine(center, end_point)

    def _draw_trails(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw target trails."""
        current_time = time.time()
        trail_duration = 5.0  # seconds

        for track_id, history in self._target_history.items():
            if len(history) < 2:
                continue

            target = self._targets.get(track_id)
            if not target:
                continue

            color = self._get_target_color(target.threat_level)

            path = QPainterPath()
            first = True

            for azimuth, distance, timestamp in history:
                age = current_time - timestamp
                if age > trail_duration:
                    continue

                point = self._polar_to_screen(azimuth, distance, center, radius)

                if first:
                    path.moveTo(point)
                    first = False
                else:
                    path.lineTo(point)

            # Draw trail with fading
            alpha = 100
            trail_color = QColor(color)
            trail_color.setAlpha(alpha)
            painter.setPen(QPen(trail_color, 2))
            painter.drawPath(path)

    def _draw_targets(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float
    ) -> None:
        """Draw targets."""
        for track_id, target in self._targets.items():
            pos = self._polar_to_screen(
                target.azimuth,
                target.distance,
                center,
                radius
            )

            color = self._get_target_color(target.threat_level)
            is_selected = track_id == self._selected_target

            # Target size based on confidence
            size = 8 + int(target.confidence * 6)

            # Draw target
            if is_selected:
                # Selection ring
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(pos, size + 5, size + 5)

            # Target dot
            painter.setPen(QPen(color.darker(120), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pos, size, size)

            # Velocity vector
            if abs(target.velocity_azimuth) > 0.1 or abs(target.velocity_radial) > 0.1:
                vel_scale = 20
                vel_angle = math.radians(target.azimuth + target.velocity_azimuth * 10 - 90)
                vel_end = QPointF(
                    pos.x() + vel_scale * math.cos(vel_angle),
                    pos.y() + vel_scale * math.sin(vel_angle)
                )
                painter.setPen(QPen(color, 2))
                painter.drawLine(pos, vel_end)

            # Label
            painter.setPen(QPen(self.TEXT_COLOR, 1))
            font = QFont("Consolas", 8)
            painter.setFont(font)

            label = f"{target.classification[:3]}"
            label_pos = pos + QPointF(size + 3, -size)
            painter.drawText(label_pos, label)

            # Distance label
            dist_label = f"{target.distance:.0f}m"
            painter.drawText(pos + QPointF(size + 3, 3), dist_label)

    def _draw_info_overlay(self, painter: QPainter) -> None:
        """Draw info overlay."""
        painter.setPen(QPen(self.TEXT_COLOR, 1))
        font = QFont("Consolas", 9)
        painter.setFont(font)

        # Target count
        info_text = f"Targets: {len(self._targets)}"
        painter.drawText(10, 20, info_text)

        # Range
        range_text = f"Range: {self._max_range:.0f}m"
        painter.drawText(10, 35, range_text)

        # Selected target info
        if self._selected_target and self._selected_target in self._targets:
            target = self._targets[self._selected_target]

            info_lines = [
                f"Track: {target.track_id}",
                f"Class: {target.classification}",
                f"Az: {target.azimuth:.1f} deg",
                f"Dist: {target.distance:.1f}m",
                f"Conf: {target.confidence:.1%}",
                f"Threat: {target.threat_level.upper()}"
            ]

            y = 60
            for line in info_lines:
                painter.drawText(10, y, line)
                y += 15

    def mousePressEvent(self, event):
        """Handle mouse press for target selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Find clicked target
            width = self.width()
            height = self.height()
            size = min(width, height)
            radius = (size - 60) / 2
            center = QPointF(width / 2, height / 2)

            click_pos = event.position()
            click_threshold = 20

            for track_id, target in self._targets.items():
                target_pos = self._polar_to_screen(
                    target.azimuth,
                    target.distance,
                    center,
                    radius
                )

                dist = math.sqrt(
                    (click_pos.x() - target_pos.x()) ** 2 +
                    (click_pos.y() - target_pos.y()) ** 2
                )

                if dist < click_threshold:
                    self._selected_target = track_id
                    self.target_selected.emit(track_id)
                    self.update()
                    return

            # Deselect if clicked on empty area
            self._selected_target = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Handle double click."""
        if self._selected_target:
            self.target_double_clicked.emit(self._selected_target)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()

        if delta > 0:
            self._max_range = max(50, self._max_range * 0.9)
        else:
            self._max_range = min(2000, self._max_range * 1.1)

        self.update()


class MiniRadarDisplay(RadarDisplay):
    """Smaller radar display for sidebar use."""

    def __init__(self, parent=None):
        super().__init__(parent, max_range=500, update_rate=15)
        self.setMinimumSize(200, 200)
        self.setMaximumSize(300, 300)
        self._num_rings = 3
        self._ring_labels = False
        self._sweep_enabled = False
