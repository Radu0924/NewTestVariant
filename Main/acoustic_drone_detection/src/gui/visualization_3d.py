"""
3D Visualization Widget

PyQt6 + OpenGL-based 3D visualization:
- 3D target positions
- Microphone array visualization
- Detection zones
- Camera controls
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Target3D:
    """3D target for visualization."""
    track_id: int
    x: float
    y: float
    z: float
    classification: str
    threat_level: str
    confidence: float


@dataclass
class Camera3D:
    """3D camera state."""
    azimuth: float = 45.0  # degrees
    elevation: float = 30.0  # degrees
    distance: float = 500.0  # meters
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0


class Simple3DView(QWidget):
    """
    Simple 3D visualization using QPainter.

    Provides basic 3D projection without OpenGL dependency.
    """

    # Colors
    BACKGROUND = QColor(15, 15, 25)
    GRID_COLOR = QColor(40, 60, 40)
    AXIS_X = QColor(255, 100, 100)
    AXIS_Y = QColor(100, 255, 100)
    AXIS_Z = QColor(100, 100, 255)
    MIC_COLOR = QColor(200, 200, 50)
    TARGET_LOW = QColor(0, 255, 100)
    TARGET_MEDIUM = QColor(255, 200, 0)
    TARGET_HIGH = QColor(255, 50, 50)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._camera = Camera3D()
        self._targets: Dict[int, Target3D] = {}
        self._mic_positions: List[Tuple[float, float, float]] = []

        # Display options
        self._show_grid = True
        self._show_axes = True
        self._show_mics = True
        self._show_range_sphere = True
        self._max_range = 500.0

        # Mouse interaction
        self._last_mouse_pos = None
        self._rotating = False

        # Animation
        self._auto_rotate = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)  # ~30 FPS

        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

    def set_microphone_positions(
        self,
        positions: List[Tuple[float, float, float]]
    ) -> None:
        """Set microphone array positions."""
        self._mic_positions = positions
        self.update()

    def set_max_range(self, range_m: float) -> None:
        """Set maximum display range."""
        self._max_range = range_m
        self.update()

    def update_target(self, target: Target3D) -> None:
        """Update or add a target."""
        self._targets[target.track_id] = target
        self.update()

    def remove_target(self, track_id: int) -> None:
        """Remove a target."""
        self._targets.pop(track_id, None)
        self.update()

    def clear_targets(self) -> None:
        """Clear all targets."""
        self._targets.clear()
        self.update()

    def set_auto_rotate(self, enabled: bool) -> None:
        """Enable/disable auto rotation."""
        self._auto_rotate = enabled

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._camera = Camera3D()
        self.update()

    def _on_timer(self) -> None:
        """Timer callback for animation."""
        if self._auto_rotate:
            self._camera.azimuth = (self._camera.azimuth + 0.5) % 360
            self.update()

    def _project_3d_to_2d(
        self,
        x: float,
        y: float,
        z: float,
        width: int,
        height: int
    ) -> Tuple[float, float, float]:
        """
        Project 3D point to 2D screen coordinates.

        Returns (screen_x, screen_y, depth)
        """
        # Camera transform
        cam_az = math.radians(self._camera.azimuth)
        cam_el = math.radians(self._camera.elevation)

        # Rotate around Y axis (azimuth)
        x_rot = x * math.cos(cam_az) - y * math.sin(cam_az)
        y_rot = x * math.sin(cam_az) + y * math.cos(cam_az)
        z_rot = z

        # Rotate around X axis (elevation)
        y_final = y_rot * math.cos(cam_el) - z_rot * math.sin(cam_el)
        z_final = y_rot * math.sin(cam_el) + z_rot * math.cos(cam_el)
        x_final = x_rot

        # Perspective projection
        scale = self._max_range / 2
        fov = 1.5  # Field of view factor

        if z_final < -self._camera.distance:
            return (0, 0, -1000)  # Behind camera

        depth = z_final + self._camera.distance
        perspective = fov * self._camera.distance / max(depth, 0.1)

        screen_x = width / 2 + x_final * perspective * width / scale / 4
        screen_y = height / 2 - y_final * perspective * height / scale / 4

        return (screen_x, screen_y, depth)

    def _get_target_color(self, threat_level: str) -> QColor:
        """Get color for threat level."""
        if threat_level == "high":
            return self.TARGET_HIGH
        elif threat_level == "medium":
            return self.TARGET_MEDIUM
        return self.TARGET_LOW

    def paintEvent(self, event):
        """Paint the 3D scene."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(self.rect(), self.BACKGROUND)

        # Draw grid
        if self._show_grid:
            self._draw_grid(painter, width, height)

        # Draw axes
        if self._show_axes:
            self._draw_axes(painter, width, height)

        # Draw range sphere
        if self._show_range_sphere:
            self._draw_range_sphere(painter, width, height)

        # Draw microphones
        if self._show_mics:
            self._draw_microphones(painter, width, height)

        # Draw targets (sorted by depth)
        self._draw_targets(painter, width, height)

        # Draw info
        self._draw_info(painter, width, height)

    def _draw_grid(self, painter: QPainter, width: int, height: int) -> None:
        """Draw ground grid."""
        painter.setPen(QPen(self.GRID_COLOR, 1))

        grid_size = self._max_range
        grid_step = grid_size / 10

        for i in range(-10, 11):
            x = i * grid_step

            # Lines parallel to Y
            p1 = self._project_3d_to_2d(x, -grid_size, 0, width, height)
            p2 = self._project_3d_to_2d(x, grid_size, 0, width, height)

            if p1[2] > 0 and p2[2] > 0:
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

            # Lines parallel to X
            p1 = self._project_3d_to_2d(-grid_size, x, 0, width, height)
            p2 = self._project_3d_to_2d(grid_size, x, 0, width, height)

            if p1[2] > 0 and p2[2] > 0:
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

    def _draw_axes(self, painter: QPainter, width: int, height: int) -> None:
        """Draw coordinate axes."""
        axis_length = self._max_range / 4
        origin = self._project_3d_to_2d(0, 0, 0, width, height)

        # X axis (red)
        x_end = self._project_3d_to_2d(axis_length, 0, 0, width, height)
        if origin[2] > 0 and x_end[2] > 0:
            painter.setPen(QPen(self.AXIS_X, 2))
            painter.drawLine(int(origin[0]), int(origin[1]), int(x_end[0]), int(x_end[1]))
            painter.drawText(int(x_end[0]) + 5, int(x_end[1]), "X")

        # Y axis (green)
        y_end = self._project_3d_to_2d(0, axis_length, 0, width, height)
        if origin[2] > 0 and y_end[2] > 0:
            painter.setPen(QPen(self.AXIS_Y, 2))
            painter.drawLine(int(origin[0]), int(origin[1]), int(y_end[0]), int(y_end[1]))
            painter.drawText(int(y_end[0]) + 5, int(y_end[1]), "Y")

        # Z axis (blue)
        z_end = self._project_3d_to_2d(0, 0, axis_length, width, height)
        if origin[2] > 0 and z_end[2] > 0:
            painter.setPen(QPen(self.AXIS_Z, 2))
            painter.drawLine(int(origin[0]), int(origin[1]), int(z_end[0]), int(z_end[1]))
            painter.drawText(int(z_end[0]) + 5, int(z_end[1]), "Z")

    def _draw_range_sphere(self, painter: QPainter, width: int, height: int) -> None:
        """Draw range indicator sphere."""
        painter.setPen(QPen(self.GRID_COLOR, 1, Qt.PenStyle.DashLine))

        # Draw circles at different heights
        for z in [0, self._max_range / 4, self._max_range / 2]:
            points = []
            for angle in range(0, 361, 10):
                rad = math.radians(angle)
                radius = self._max_range * 0.8

                x = radius * math.cos(rad)
                y = radius * math.sin(rad)

                p = self._project_3d_to_2d(x, y, z, width, height)
                if p[2] > 0:
                    points.append((int(p[0]), int(p[1])))

            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

    def _draw_microphones(self, painter: QPainter, width: int, height: int) -> None:
        """Draw microphone positions."""
        painter.setPen(QPen(self.MIC_COLOR, 1))
        painter.setBrush(QBrush(self.MIC_COLOR))

        for mx, my, mz in self._mic_positions:
            # Scale mic positions (usually in cm, display in m)
            p = self._project_3d_to_2d(mx * 100, my * 100, mz * 100, width, height)

            if p[2] > 0:
                painter.drawEllipse(int(p[0]) - 3, int(p[1]) - 3, 6, 6)

    def _draw_targets(self, painter: QPainter, width: int, height: int) -> None:
        """Draw targets sorted by depth."""
        # Sort targets by depth
        target_depth = []
        for track_id, target in self._targets.items():
            p = self._project_3d_to_2d(target.x, target.y, target.z, width, height)
            if p[2] > 0:
                target_depth.append((track_id, target, p))

        target_depth.sort(key=lambda x: -x[2][2])  # Far to near

        for track_id, target, (sx, sy, depth) in target_depth:
            color = self._get_target_color(target.threat_level)

            # Size based on distance (perspective)
            size = max(5, int(15 * self._camera.distance / max(depth, 1)))

            # Draw target
            painter.setPen(QPen(color.darker(120), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(sx) - size, int(sy) - size, size * 2, size * 2)

            # Draw vertical line to ground
            ground_p = self._project_3d_to_2d(target.x, target.y, 0, width, height)
            if ground_p[2] > 0:
                painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(sx), int(sy), int(ground_p[0]), int(ground_p[1]))

            # Label
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            font = QFont("Consolas", 9)
            painter.setFont(font)
            painter.drawText(int(sx) + size + 3, int(sy) - 5, f"T{track_id}")
            painter.drawText(int(sx) + size + 3, int(sy) + 10, target.classification[:8])

    def _draw_info(self, painter: QPainter, width: int, height: int) -> None:
        """Draw info overlay."""
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        font = QFont("Consolas", 9)
        painter.setFont(font)

        info = [
            f"Targets: {len(self._targets)}",
            f"Range: {self._max_range:.0f}m",
            f"Az: {self._camera.azimuth:.0f} deg",
            f"El: {self._camera.elevation:.0f} deg"
        ]

        y = 20
        for line in info:
            painter.drawText(10, y, line)
            y += 15

        # Controls hint
        painter.drawText(10, height - 10, "Drag to rotate, Scroll to zoom")

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating = True
            self._last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating = False

    def mouseMoveEvent(self, event):
        """Handle mouse move for rotation."""
        if self._rotating and self._last_mouse_pos:
            delta = event.position() - self._last_mouse_pos

            self._camera.azimuth = (self._camera.azimuth - delta.x() * 0.5) % 360
            self._camera.elevation = max(-89, min(89,
                self._camera.elevation + delta.y() * 0.5
            ))

            self._last_mouse_pos = event.position()
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()

        if delta > 0:
            self._camera.distance = max(50, self._camera.distance * 0.9)
        else:
            self._camera.distance = min(2000, self._camera.distance * 1.1)

        self.update()


class Visualization3DWidget(QWidget):
    """3D visualization widget with controls."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create 3D view
        self._view = Simple3DView(self)

        # Create controls
        controls = QHBoxLayout()

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._view.reset_camera)
        controls.addWidget(self._reset_btn)

        self._rotate_btn = QPushButton("Auto Rotate")
        self._rotate_btn.setCheckable(True)
        self._rotate_btn.toggled.connect(self._view.set_auto_rotate)
        controls.addWidget(self._rotate_btn)

        controls.addWidget(QLabel("Range:"))
        self._range_slider = QSlider(Qt.Orientation.Horizontal)
        self._range_slider.setRange(100, 1000)
        self._range_slider.setValue(500)
        self._range_slider.valueChanged.connect(self._on_range_changed)
        controls.addWidget(self._range_slider)

        self._range_label = QLabel("500m")
        controls.addWidget(self._range_label)

        controls.addStretch()

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._view, 1)

    def _on_range_changed(self, value: int) -> None:
        """Handle range slider change."""
        self._view.set_max_range(value)
        self._range_label.setText(f"{value}m")

    def update_target(self, target: Target3D) -> None:
        """Update or add a target."""
        self._view.update_target(target)

    def remove_target(self, track_id: int) -> None:
        """Remove a target."""
        self._view.remove_target(track_id)

    def clear_targets(self) -> None:
        """Clear all targets."""
        self._view.clear_targets()

    def set_microphone_positions(
        self,
        positions: List[Tuple[float, float, float]]
    ) -> None:
        """Set microphone positions."""
        self._view.set_microphone_positions(positions)
