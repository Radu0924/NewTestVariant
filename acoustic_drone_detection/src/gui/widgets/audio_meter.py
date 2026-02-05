"""
Audio Level Meter Widget

Reusable audio level meter for displaying audio levels
across multiple channels with peak hold indicators.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QLinearGradient
import time
from typing import List


class AudioLevelMeter(QWidget):
    """Audio level meter widget."""

    def __init__(self, parent=None, channels: int = 1, orientation: str = "horizontal"):
        super().__init__(parent)

        self._channels = channels
        self._levels: List[float] = [0.0] * channels
        self._peak_levels: List[float] = [0.0] * channels
        self._peak_hold_time = 1.0
        self._peak_timestamps: List[float] = [0.0] * channels
        self._orientation = orientation

        if orientation == "horizontal":
            self.setMinimumHeight(20 * channels)
        else:
            self.setMinimumWidth(20 * channels)

    def set_channels(self, channels: int) -> None:
        """Set number of channels."""
        self._channels = channels
        self._levels = [0.0] * channels
        self._peak_levels = [0.0] * channels
        self._peak_timestamps = [0.0] * channels
        if self._orientation == "horizontal":
            self.setMinimumHeight(20 * channels)
        else:
            self.setMinimumWidth(20 * channels)
        self.update()

    def set_levels(self, levels: List[float]) -> None:
        """Set current audio levels (0.0 to 1.0)."""
        current_time = time.time()

        for i, level in enumerate(levels[:self._channels]):
            self._levels[i] = max(0.0, min(1.0, level))

            # Update peak
            if level > self._peak_levels[i]:
                self._peak_levels[i] = level
                self._peak_timestamps[i] = current_time
            elif current_time - self._peak_timestamps[i] > self._peak_hold_time:
                self._peak_levels[i] = level

        self.update()

    def paintEvent(self, event):
        """Paint the level meters."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        if self._orientation == "horizontal":
            self._paint_horizontal(painter, width, height)
        else:
            self._paint_vertical(painter, width, height)

    def _paint_horizontal(self, painter: QPainter, width: int, height: int):
        """Paint horizontal meters."""
        channel_height = height // max(1, self._channels)
        margin = 2

        for i in range(self._channels):
            y = i * channel_height + margin
            h = channel_height - 2 * margin

            # Background
            painter.fillRect(0, y, width, h, QColor(30, 30, 40))

            # Level bar with gradient
            level_width = int(self._levels[i] * width)
            if level_width > 0:
                gradient = QLinearGradient(0, 0, width, 0)
                gradient.setColorAt(0, QColor(0, 200, 0))
                gradient.setColorAt(0.7, QColor(200, 200, 0))
                gradient.setColorAt(0.9, QColor(255, 100, 0))
                gradient.setColorAt(1.0, QColor(255, 0, 0))

                painter.fillRect(0, y, level_width, h, gradient)

            # Peak indicator
            peak_x = int(self._peak_levels[i] * width)
            if peak_x > 0:
                color = QColor(255, 255, 0)
                if self._peak_levels[i] > 0.9:
                    color = QColor(255, 0, 0)
                painter.fillRect(peak_x - 2, y, 3, h, color)

            # Channel label
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(5, y + h - 3, f"Ch{i+1}")

    def _paint_vertical(self, painter: QPainter, width: int, height: int):
        """Paint vertical meters."""
        channel_width = width // max(1, self._channels)
        margin = 2

        for i in range(self._channels):
            x = i * channel_width + margin
            w = channel_width - 2 * margin

            # Background
            painter.fillRect(x, 0, w, height, QColor(30, 30, 40))

            # Level bar with gradient (from bottom up)
            level_height = int(self._levels[i] * height)
            if level_height > 0:
                gradient = QLinearGradient(0, height, 0, 0)
                gradient.setColorAt(0, QColor(0, 200, 0))
                gradient.setColorAt(0.7, QColor(200, 200, 0))
                gradient.setColorAt(0.9, QColor(255, 100, 0))
                gradient.setColorAt(1.0, QColor(255, 0, 0))

                painter.fillRect(x, height - level_height, w, level_height, gradient)

            # Peak indicator
            peak_y = height - int(self._peak_levels[i] * height)
            if self._peak_levels[i] > 0:
                color = QColor(255, 255, 0)
                if self._peak_levels[i] > 0.9:
                    color = QColor(255, 0, 0)
                painter.fillRect(x, peak_y - 1, w, 3, color)

            # Channel label
            painter.setPen(QColor(150, 150, 150))
            painter.save()
            painter.translate(x + w - 3, height - 5)
            painter.rotate(-90)
            painter.drawText(0, 0, f"Ch{i+1}")
            painter.restore()
