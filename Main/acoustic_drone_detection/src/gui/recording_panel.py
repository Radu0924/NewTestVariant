"""
Recording Panel Widget

Audio recording controls:
- Start/stop recording
- Recording format settings
- Audio level meters
- Playback controls
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QProgressBar,
    QGroupBox, QFileDialog, QLineEdit, QCheckBox
)
from PyQt6.QtCore import QTimer, pyqtSignal
import time
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

from .widgets import AudioLevelMeter


@dataclass
class RecordingSettings:
    """Recording configuration."""
    output_directory: str = ""
    format: str = "wav"
    bit_depth: int = 16
    auto_split: bool = False
    split_duration: int = 60
    auto_record_on_detection: bool = False


class RecordingPanel(QWidget):
    """
    Audio recording control panel.

    Provides controls for recording, playback, and level monitoring.
    """

    # Signals
    recording_started = pyqtSignal(str)  # filename
    recording_stopped = pyqtSignal(str)  # filename
    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._recording = False
        self._record_start_time: Optional[float] = None
        self._current_filename: Optional[str] = None
        self._output_dir = str(Path.home() / "DroneDetectionRecordings")

        self._setup_ui()

        # Timer for updating display
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)
        self._timer.start(100)

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Level meters
        meters_group = QGroupBox("Audio Levels")
        meters_layout = QVBoxLayout(meters_group)

        self._level_meter = AudioLevelMeter(channels=8)
        meters_layout.addWidget(self._level_meter)

        layout.addWidget(meters_group)

        # Recording controls
        controls_group = QGroupBox("Recording")
        controls_layout = QVBoxLayout(controls_group)

        # Status and time
        status_layout = QHBoxLayout()

        self._status_label = QLabel("Stopped")
        self._status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self._status_label)

        status_layout.addStretch()

        self._time_label = QLabel("00:00:00")
        self._time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        status_layout.addWidget(self._time_label)

        controls_layout.addLayout(status_layout)

        # Recording progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setMinimumHeight(5)
        controls_layout.addWidget(self._progress_bar)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self._record_btn = QPushButton("Record")
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_clicked)
        self._record_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #cc4444;
            }
        """)
        buttons_layout.addWidget(self._record_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        buttons_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        buttons_layout.addWidget(self._stop_btn)

        controls_layout.addLayout(buttons_layout)

        # Filename
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))

        self._filename_label = QLabel("-")
        self._filename_label.setStyleSheet("color: #888;")
        file_layout.addWidget(self._filename_label, 1)

        controls_layout.addLayout(file_layout)

        layout.addWidget(controls_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)

        # Output directory
        settings_layout.addWidget(QLabel("Output:"), 0, 0)

        self._output_edit = QLineEdit(self._output_dir)
        self._output_edit.setReadOnly(True)
        settings_layout.addWidget(self._output_edit, 0, 1)

        self._browse_btn = QPushButton("...")
        self._browse_btn.setMaximumWidth(30)
        self._browse_btn.clicked.connect(self._on_browse)
        settings_layout.addWidget(self._browse_btn, 0, 2)

        # Format
        settings_layout.addWidget(QLabel("Format:"), 1, 0)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["WAV (16-bit)", "WAV (24-bit)", "WAV (32-bit float)", "FLAC"])
        settings_layout.addWidget(self._format_combo, 1, 1, 1, 2)

        # Auto split
        self._auto_split_check = QCheckBox("Auto-split every")
        settings_layout.addWidget(self._auto_split_check, 2, 0)

        split_layout = QHBoxLayout()
        self._split_spin = QSpinBox()
        self._split_spin.setRange(1, 3600)
        self._split_spin.setValue(60)
        split_layout.addWidget(self._split_spin)
        split_layout.addWidget(QLabel("seconds"))
        split_layout.addStretch()
        settings_layout.addLayout(split_layout, 2, 1, 1, 2)

        # Auto record on detection
        self._auto_record_check = QCheckBox("Auto-record on detection")
        settings_layout.addWidget(self._auto_record_check, 3, 0, 1, 3)

        layout.addWidget(settings_group)

        # Disk space
        space_layout = QHBoxLayout()
        space_layout.addWidget(QLabel("Disk Space:"))

        self._space_bar = QProgressBar()
        self._space_bar.setMaximum(100)
        self._space_bar.setValue(50)
        space_layout.addWidget(self._space_bar)

        self._space_label = QLabel("500 GB free")
        space_layout.addWidget(self._space_label)

        layout.addLayout(space_layout)

        layout.addStretch()

    def _on_record_clicked(self) -> None:
        """Handle record button click."""
        if self._record_btn.isChecked():
            self._start_recording()
        else:
            self._stop_recording()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self._stop_recording()

    def _start_recording(self) -> None:
        """Start recording."""
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        format_ext = "wav"
        if "FLAC" in self._format_combo.currentText():
            format_ext = "flac"

        self._current_filename = f"recording_{timestamp}.{format_ext}"

        # Create output directory
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        full_path = str(Path(self._output_dir) / self._current_filename)

        self._recording = True
        self._record_start_time = time.time()

        # Update UI
        self._status_label.setText("Recording")
        self._status_label.setStyleSheet("font-weight: bold; color: #ff4444;")
        self._stop_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._record_btn.setChecked(True)
        self._filename_label.setText(self._current_filename)

        # Emit signal
        self.recording_started.emit(full_path)

    def _stop_recording(self) -> None:
        """Stop recording."""
        if not self._recording:
            return

        full_path = str(Path(self._output_dir) / self._current_filename) if self._current_filename else ""

        self._recording = False
        self._record_start_time = None

        # Update UI
        self._status_label.setText("Stopped")
        self._status_label.setStyleSheet("font-weight: bold;")
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._record_btn.setChecked(False)
        self._time_label.setText("00:00:00")
        self._progress_bar.setValue(0)

        # Emit signal
        self.recording_stopped.emit(full_path)

    def _on_browse(self) -> None:
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._output_dir
        )

        if directory:
            self._output_dir = directory
            self._output_edit.setText(directory)
            self.settings_changed.emit()

    def _update_display(self) -> None:
        """Update display elements."""
        # Update recording time
        if self._recording and self._record_start_time:
            elapsed = time.time() - self._record_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self._time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            # Update progress (pulse effect)
            pulse = int((elapsed * 2) % 100)
            self._progress_bar.setValue(pulse)

        # Update disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self._output_dir)
            percent_used = int(used / total * 100)
            free_gb = free / (1024 ** 3)

            self._space_bar.setValue(percent_used)
            self._space_label.setText(f"{free_gb:.1f} GB free")

            if percent_used > 90:
                self._space_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
            elif percent_used > 70:
                self._space_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; }")
            else:
                self._space_bar.setStyleSheet("")
        except Exception:
            pass

    def update_levels(self, levels: List[float]) -> None:
        """Update audio level meters."""
        self._level_meter.set_levels(levels)

    def set_num_channels(self, channels: int) -> None:
        """Set number of audio channels."""
        self._level_meter.set_channels(channels)

    def get_settings(self) -> RecordingSettings:
        """Get current recording settings."""
        format_text = self._format_combo.currentText()
        if "24-bit" in format_text:
            bit_depth = 24
        elif "32-bit" in format_text:
            bit_depth = 32
        else:
            bit_depth = 16

        fmt = "flac" if "FLAC" in format_text else "wav"

        return RecordingSettings(
            output_directory=self._output_dir,
            format=fmt,
            bit_depth=bit_depth,
            auto_split=self._auto_split_check.isChecked(),
            split_duration=self._split_spin.value(),
            auto_record_on_detection=self._auto_record_check.isChecked()
        )

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def output_directory(self) -> str:
        """Get output directory."""
        return self._output_dir


class MiniRecordingWidget(QWidget):
    """Compact recording widget for status bar."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self._indicator = QLabel()
        self._indicator.setFixedSize(12, 12)
        self._indicator.setStyleSheet("""
            background-color: #444;
            border-radius: 6px;
        """)
        layout.addWidget(self._indicator)

        self._label = QLabel("REC")
        self._label.setStyleSheet("color: #888;")
        layout.addWidget(self._label)

        self._time_label = QLabel("00:00")
        self._time_label.setStyleSheet("color: #888; font-family: monospace;")
        layout.addWidget(self._time_label)

        self._recording = False
        self._blink_state = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._blink)
        self._timer.start(500)

    def set_recording(self, recording: bool, elapsed: float = 0) -> None:
        """Update recording state."""
        self._recording = recording

        if recording:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self._time_label.setText(f"{minutes:02d}:{seconds:02d}")
            self._label.setStyleSheet("color: #ff4444; font-weight: bold;")
        else:
            self._time_label.setText("--:--")
            self._label.setStyleSheet("color: #888;")
            self._indicator.setStyleSheet("""
                background-color: #444;
                border-radius: 6px;
            """)

    def _blink(self) -> None:
        """Blink indicator when recording."""
        if self._recording:
            self._blink_state = not self._blink_state
            if self._blink_state:
                self._indicator.setStyleSheet("""
                    background-color: #ff4444;
                    border-radius: 6px;
                """)
            else:
                self._indicator.setStyleSheet("""
                    background-color: #aa2222;
                    border-radius: 6px;
                """)

    def mousePressEvent(self, event):
        """Handle click."""
        self.clicked.emit()
        super().mousePressEvent(event)
