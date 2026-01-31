"""
Spectrum View Widget

PyQt6-based spectrum analyzer and spectrogram display:
- Real-time FFT spectrum
- Scrolling spectrogram (waterfall)
- Frequency markers for drone signatures
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QImage, QPixmap
import numpy as np
from typing import Optional, List, Tuple
from collections import deque


class SpectrumView(QWidget):
    """
    Real-time spectrum analyzer widget.

    Displays FFT magnitude spectrum with peak detection
    and frequency markers.
    """

    # Colors
    BACKGROUND = QColor(20, 20, 30)
    GRID_COLOR = QColor(40, 40, 60)
    SPECTRUM_COLOR = QColor(0, 200, 255)
    PEAK_COLOR = QColor(255, 200, 0)
    MARKER_COLOR = QColor(255, 100, 100)
    TEXT_COLOR = QColor(200, 200, 200)

    def __init__(
        self,
        parent=None,
        sample_rate: int = 48000,
        fft_size: int = 2048
    ):
        """
        Initialize spectrum view.

        Args:
            parent: Parent widget.
            sample_rate: Audio sample rate.
            fft_size: FFT size.
        """
        super().__init__(parent)

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._spectrum_data: Optional[np.ndarray] = None
        self._peak_hold: Optional[np.ndarray] = None
        self._peak_decay = 0.95

        # Display range
        self._min_freq = 0
        self._max_freq = sample_rate // 2
        self._min_db = -100
        self._max_db = 0

        # Frequency markers
        self._markers: List[Tuple[float, str]] = []

        # Peak detection
        self._detected_peaks: List[Tuple[float, float]] = []
        self._peak_threshold = -60  # dB

        # Setup widget
        self.setMinimumSize(400, 150)

    def set_frequency_range(self, min_freq: float, max_freq: float) -> None:
        """Set display frequency range."""
        self._min_freq = min_freq
        self._max_freq = min(max_freq, self._sample_rate // 2)
        self.update()

    def set_db_range(self, min_db: float, max_db: float) -> None:
        """Set display dB range."""
        self._min_db = min_db
        self._max_db = max_db
        self.update()

    def add_marker(self, frequency: float, label: str) -> None:
        """Add a frequency marker."""
        self._markers.append((frequency, label))
        self.update()

    def clear_markers(self) -> None:
        """Clear all markers."""
        self._markers.clear()
        self.update()

    def update_spectrum(self, audio_data: np.ndarray) -> None:
        """
        Update spectrum with new audio data.

        Args:
            audio_data: Audio samples (mono).
        """
        # Compute FFT
        if len(audio_data) < self._fft_size:
            audio_data = np.pad(audio_data, (0, self._fft_size - len(audio_data)))
        else:
            audio_data = audio_data[:self._fft_size]

        # Apply window
        window = np.hanning(len(audio_data))
        windowed = audio_data * window

        # FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        self._spectrum_data = magnitude_db

        # Update peak hold
        if self._peak_hold is None:
            self._peak_hold = magnitude_db.copy()
        else:
            self._peak_hold = np.maximum(
                self._peak_hold * self._peak_decay,
                magnitude_db
            )

        # Detect peaks
        self._detect_peaks(magnitude_db)

        self.update()

    def _detect_peaks(self, spectrum_db: np.ndarray) -> None:
        """Detect spectral peaks."""
        self._detected_peaks = []

        freq_resolution = self._sample_rate / self._fft_size

        # Simple peak detection
        for i in range(1, len(spectrum_db) - 1):
            if (spectrum_db[i] > spectrum_db[i-1] and
                spectrum_db[i] > spectrum_db[i+1] and
                spectrum_db[i] > self._peak_threshold):

                freq = i * freq_resolution
                if self._min_freq <= freq <= self._max_freq:
                    self._detected_peaks.append((freq, spectrum_db[i]))

        # Sort by magnitude and keep top peaks
        self._detected_peaks.sort(key=lambda x: -x[1])
        self._detected_peaks = self._detected_peaks[:10]

    def _freq_to_x(self, freq: float, width: float) -> float:
        """Convert frequency to x coordinate."""
        freq_range = self._max_freq - self._min_freq
        if freq_range <= 0:
            return 0
        return (freq - self._min_freq) / freq_range * width

    def _db_to_y(self, db: float, height: float) -> float:
        """Convert dB to y coordinate."""
        db_range = self._max_db - self._min_db
        if db_range <= 0:
            return height
        return height - (db - self._min_db) / db_range * height

    def paintEvent(self, event):
        """Paint the spectrum."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(self.rect(), self.BACKGROUND)

        # Draw grid
        self._draw_grid(painter, width, height)

        # Draw markers
        self._draw_markers(painter, width, height)

        # Draw spectrum
        if self._spectrum_data is not None:
            self._draw_spectrum(painter, width, height)

        # Draw peaks
        self._draw_peaks(painter, width, height)

        # Draw labels
        self._draw_labels(painter, width, height)

    def _draw_grid(self, painter: QPainter, width: int, height: int) -> None:
        """Draw frequency and dB grid."""
        painter.setPen(QPen(self.GRID_COLOR, 1))

        # Frequency grid (every 1kHz)
        freq_step = 1000
        freq = self._min_freq - (self._min_freq % freq_step) + freq_step

        while freq < self._max_freq:
            x = self._freq_to_x(freq, width)
            painter.drawLine(int(x), 0, int(x), height)
            freq += freq_step

        # dB grid (every 10dB)
        db_step = 10
        db = self._min_db - (self._min_db % db_step) + db_step

        while db < self._max_db:
            y = self._db_to_y(db, height)
            painter.drawLine(0, int(y), width, int(y))
            db += db_step

    def _draw_markers(self, painter: QPainter, width: int, height: int) -> None:
        """Draw frequency markers."""
        painter.setPen(QPen(self.MARKER_COLOR, 1, Qt.PenStyle.DashLine))

        for freq, label in self._markers:
            if self._min_freq <= freq <= self._max_freq:
                x = self._freq_to_x(freq, width)
                painter.drawLine(int(x), 0, int(x), height)

                # Label
                painter.setPen(QPen(self.MARKER_COLOR, 1))
                font = QFont("Consolas", 8)
                painter.setFont(font)
                painter.drawText(int(x) + 2, 12, label)
                painter.setPen(QPen(self.MARKER_COLOR, 1, Qt.PenStyle.DashLine))

    def _draw_spectrum(self, painter: QPainter, width: int, height: int) -> None:
        """Draw spectrum curve."""
        if self._spectrum_data is None:
            return

        freq_resolution = self._sample_rate / self._fft_size

        # Draw filled spectrum
        painter.setPen(Qt.PenStyle.NoPen)
        gradient_color = QColor(self.SPECTRUM_COLOR)
        gradient_color.setAlpha(100)
        painter.setBrush(QBrush(gradient_color))

        # Build polygon
        points = []
        points.append((0, height))

        for i, db in enumerate(self._spectrum_data):
            freq = i * freq_resolution
            if freq < self._min_freq:
                continue
            if freq > self._max_freq:
                break

            x = self._freq_to_x(freq, width)
            y = self._db_to_y(db, height)
            points.append((x, max(0, min(height, y))))

        points.append((width, height))

        # Draw as polygon
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtCore import QPointF

        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        painter.drawPolygon(polygon)

        # Draw line
        painter.setPen(QPen(self.SPECTRUM_COLOR, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for i in range(1, len(points) - 1):
            if i > 1:
                painter.drawLine(
                    int(points[i-1][0]), int(points[i-1][1]),
                    int(points[i][0]), int(points[i][1])
                )

        # Draw peak hold
        if self._peak_hold is not None:
            painter.setPen(QPen(self.PEAK_COLOR, 1))

            for i, db in enumerate(self._peak_hold):
                freq = i * freq_resolution
                if freq < self._min_freq:
                    continue
                if freq > self._max_freq:
                    break

                x = self._freq_to_x(freq, width)
                y = self._db_to_y(db, height)

                if i > 0:
                    prev_freq = (i - 1) * freq_resolution
                    if prev_freq >= self._min_freq:
                        prev_x = self._freq_to_x(prev_freq, width)
                        prev_y = self._db_to_y(self._peak_hold[i-1], height)
                        painter.drawLine(int(prev_x), int(prev_y), int(x), int(y))

    def _draw_peaks(self, painter: QPainter, width: int, height: int) -> None:
        """Draw detected peaks."""
        painter.setPen(QPen(self.PEAK_COLOR, 1))
        font = QFont("Consolas", 8)
        painter.setFont(font)

        for freq, db in self._detected_peaks[:5]:  # Top 5 peaks
            x = self._freq_to_x(freq, width)
            y = self._db_to_y(db, height)

            # Peak marker
            painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)

            # Frequency label
            if freq >= 1000:
                label = f"{freq/1000:.1f}kHz"
            else:
                label = f"{freq:.0f}Hz"

            painter.drawText(int(x) + 5, int(y) - 5, label)

    def _draw_labels(self, painter: QPainter, width: int, height: int) -> None:
        """Draw axis labels."""
        painter.setPen(QPen(self.TEXT_COLOR, 1))
        font = QFont("Consolas", 8)
        painter.setFont(font)

        # Frequency labels
        freq_step = 2000
        freq = self._min_freq - (self._min_freq % freq_step) + freq_step

        while freq < self._max_freq:
            x = self._freq_to_x(freq, width)
            if freq >= 1000:
                label = f"{freq/1000:.0f}k"
            else:
                label = f"{freq:.0f}"
            painter.drawText(int(x) - 10, height - 2, label)
            freq += freq_step

        # dB labels
        db_step = 20
        db = self._min_db - (self._min_db % db_step) + db_step

        while db < self._max_db:
            y = self._db_to_y(db, height)
            painter.drawText(2, int(y) + 4, f"{db:.0f}dB")
            db += db_step


class SpectrogramView(QWidget):
    """
    Scrolling spectrogram (waterfall) display.

    Shows time-frequency representation with color-mapped intensity.
    """

    def __init__(
        self,
        parent=None,
        sample_rate: int = 48000,
        fft_size: int = 1024,
        history_seconds: float = 10.0
    ):
        """
        Initialize spectrogram view.

        Args:
            parent: Parent widget.
            sample_rate: Audio sample rate.
            fft_size: FFT size.
            history_seconds: History length in seconds.
        """
        super().__init__(parent)

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._history_seconds = history_seconds

        # Calculate history size
        self._hop_size = fft_size // 4
        self._frames_per_second = sample_rate / self._hop_size
        self._history_frames = int(history_seconds * self._frames_per_second)

        # Spectrogram data
        self._num_bins = fft_size // 2 + 1
        self._spectrogram = deque(maxlen=self._history_frames)

        # Display range
        self._min_freq = 0
        self._max_freq = sample_rate // 2
        self._min_db = -100
        self._max_db = -20

        # Color map
        self._colormap = self._create_colormap()

        # Image buffer
        self._image: Optional[QImage] = None

        # Setup widget
        self.setMinimumSize(400, 200)

    def _create_colormap(self) -> List[QColor]:
        """Create colormap for intensity display."""
        colors = []
        for i in range(256):
            t = i / 255.0

            if t < 0.25:
                # Black to blue
                r, g, b = 0, 0, int(t * 4 * 255)
            elif t < 0.5:
                # Blue to cyan
                r, g, b = 0, int((t - 0.25) * 4 * 255), 255
            elif t < 0.75:
                # Cyan to yellow
                r = int((t - 0.5) * 4 * 255)
                g = 255
                b = int((0.75 - t) * 4 * 255)
            else:
                # Yellow to red
                r = 255
                g = int((1.0 - t) * 4 * 255)
                b = 0

            colors.append(QColor(r, g, b))

        return colors

    def set_frequency_range(self, min_freq: float, max_freq: float) -> None:
        """Set display frequency range."""
        self._min_freq = min_freq
        self._max_freq = min(max_freq, self._sample_rate // 2)
        self._update_image()

    def set_db_range(self, min_db: float, max_db: float) -> None:
        """Set display dB range."""
        self._min_db = min_db
        self._max_db = max_db
        self._update_image()

    def update_spectrogram(self, audio_data: np.ndarray) -> None:
        """
        Update spectrogram with new audio data.

        Args:
            audio_data: Audio samples (mono).
        """
        # Process in frames
        for i in range(0, len(audio_data) - self._fft_size, self._hop_size):
            frame = audio_data[i:i + self._fft_size]

            # Apply window
            window = np.hanning(self._fft_size)
            windowed = frame * window

            # FFT
            fft_result = np.fft.rfft(windowed)
            magnitude = np.abs(fft_result)

            # Convert to dB
            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            self._spectrogram.append(magnitude_db)

        self._update_image()
        self.update()

    def _update_image(self) -> None:
        """Update the spectrogram image."""
        if not self._spectrogram:
            return

        width = len(self._spectrogram)
        height = self._num_bins

        # Create image
        self._image = QImage(width, height, QImage.Format.Format_RGB32)

        freq_resolution = self._sample_rate / self._fft_size

        for x, spectrum in enumerate(self._spectrogram):
            for y in range(height):
                freq = y * freq_resolution

                # Check frequency range
                if freq < self._min_freq or freq > self._max_freq:
                    color = self._colormap[0]
                else:
                    # Normalize dB to color index
                    db = spectrum[y]
                    normalized = (db - self._min_db) / (self._max_db - self._min_db)
                    normalized = max(0, min(1, normalized))
                    color_idx = int(normalized * 255)
                    color = self._colormap[color_idx]

                # Flip y axis (low frequencies at bottom)
                self._image.setPixelColor(x, height - 1 - y, color)

    def paintEvent(self, event):
        """Paint the spectrogram."""
        painter = QPainter(self)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        # Draw spectrogram
        if self._image is not None:
            scaled = self._image.scaled(
                width, height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.drawImage(0, 0, scaled)

        # Draw frequency labels
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Consolas", 8)
        painter.setFont(font)

        freq_range = self._max_freq - self._min_freq
        freq_step = freq_range / 5

        for i in range(6):
            freq = self._min_freq + i * freq_step
            y = height - int(i / 5 * height)

            if freq >= 1000:
                label = f"{freq/1000:.1f}k"
            else:
                label = f"{freq:.0f}"

            painter.drawText(2, y - 2, label)

        # Time labels
        time_step = self._history_seconds / 5
        for i in range(6):
            t = i * time_step
            x = int(i / 5 * width)
            painter.drawText(x + 2, height - 2, f"-{self._history_seconds - t:.1f}s")

    def clear(self) -> None:
        """Clear spectrogram data."""
        self._spectrogram.clear()
        self._image = None
        self.update()


class CombinedSpectrumWidget(QWidget):
    """Combined spectrum and spectrogram widget."""

    def __init__(self, parent=None, sample_rate: int = 48000):
        super().__init__(parent)

        self._sample_rate = sample_rate

        # Create widgets
        self._spectrum = SpectrumView(self, sample_rate)
        self._spectrogram = SpectrogramView(self, sample_rate)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Add controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Freq Range:"))

        self._freq_combo = QComboBox()
        self._freq_combo.addItems(["0-24kHz", "0-8kHz", "0-4kHz", "0-2kHz"])
        self._freq_combo.currentIndexChanged.connect(self._on_freq_changed)
        controls.addWidget(self._freq_combo)

        controls.addStretch()
        layout.addLayout(controls)

        layout.addWidget(self._spectrum, 1)
        layout.addWidget(self._spectrogram, 2)

    def _on_freq_changed(self, index: int) -> None:
        """Handle frequency range change."""
        ranges = [(0, 24000), (0, 8000), (0, 4000), (0, 2000)]
        min_f, max_f = ranges[index]

        self._spectrum.set_frequency_range(min_f, max_f)
        self._spectrogram.set_frequency_range(min_f, max_f)

    def update_audio(self, audio_data: np.ndarray) -> None:
        """Update with new audio data."""
        self._spectrum.update_spectrum(audio_data)
        self._spectrogram.update_spectrogram(audio_data)

    def add_marker(self, frequency: float, label: str) -> None:
        """Add frequency marker."""
        self._spectrum.add_marker(frequency, label)
