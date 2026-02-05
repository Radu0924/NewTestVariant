"""
Main Window Module

PyQt6 main application window for the drone detection system:
- Integrates all display widgets
- Provides menu and toolbar
- Manages system lifecycle
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QDockWidget, QToolBar, QStatusBar, QMenuBar, QMenu,
    QLabel, QPushButton, QMessageBox, QFileDialog, QSplitter,
    QTabWidget, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor
import sys
import time
from typing import Optional
from pathlib import Path

# Import widgets
from .radar_display import RadarDisplay, RadarTarget
from .spectrum_view import CombinedSpectrumWidget
from .visualization_3d import Visualization3DWidget, Target3D
from .settings_panel import SettingsPanel
from .alerts_panel import AlertsPanel, AlertEvent
from .recording_panel import RecordingPanel, MiniRecordingWidget
from .widgets import AudioLevelMeter


class StatusIndicator(QWidget):
    """System status indicator widget."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self._indicator = QLabel()
        self._indicator.setFixedSize(12, 12)
        layout.addWidget(self._indicator)

        self._label = QLabel("Stopped")
        layout.addWidget(self._label)

        self.set_status("stopped")

    def set_status(self, status: str) -> None:
        """Set status indicator."""
        colors = {
            "running": ("#44ff88", "Running"),
            "stopped": ("#888888", "Stopped"),
            "starting": ("#ffaa00", "Starting..."),
            "error": ("#ff4444", "Error")
        }

        color, text = colors.get(status, ("#888888", status))
        self._indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 6px;
        """)
        self._label.setText(text)


class MainWindow(QMainWindow):
    """
    Main application window.

    Integrates all detection system components and display widgets.
    """

    def __init__(self, detection_system=None):
        super().__init__()

        self._detection_system = detection_system
        self._running = False
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        # Connect to detection system if provided
        if detection_system:
            detection_system.add_callback(self._on_detection)
            try:
                self._settings_panel.load_from_config(detection_system._config_manager.config)
            except Exception:
                pass

    def _setup_ui(self):
        """Setup the main UI."""
        self.setWindowTitle("Acoustic Drone Detection System")
        self.setMinimumSize(1200, 800)

        # Central widget with main layout
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Radar and 3D view
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Radar display
        self._radar = RadarDisplay()
        self._radar.target_selected.connect(self._on_target_selected)
        left_layout.addWidget(self._radar, 2)

        # 3D view
        self._view_3d = Visualization3DWidget()
        left_layout.addWidget(self._view_3d, 1)

        splitter.addWidget(left_widget)

        # Center panel - Spectrum and alerts
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Spectrum view
        self._spectrum = CombinedSpectrumWidget()
        center_layout.addWidget(self._spectrum, 1)

        # Alerts panel
        self._alerts = AlertsPanel()
        self._alerts.alert_selected.connect(self._on_alert_selected)
        center_layout.addWidget(self._alerts, 1)

        splitter.addWidget(center_widget)

        # Set splitter sizes
        splitter.setSizes([500, 700])

        main_layout.addWidget(splitter)

        # Right dock - Settings and recording
        self._settings_dock = QDockWidget("Settings", self)
        self._settings_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        # Tab widget for settings panels
        settings_tabs = QTabWidget()

        self._settings_panel = SettingsPanel()
        self._settings_panel.apply_clicked.connect(self._apply_settings)
        settings_tabs.addTab(self._settings_panel, "Settings")

        self._recording_panel = RecordingPanel()
        self._recording_panel.recording_started.connect(self._on_recording_started)
        self._recording_panel.recording_stopped.connect(self._on_recording_stopped)
        settings_tabs.addTab(self._recording_panel, "Recording")

        self._settings_dock.setWidget(settings_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._settings_dock)

        # Audio levels dock at bottom
        self._audio_dock = QDockWidget("Audio Levels", self)
        self._audio_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        channel_count = 8
        if self._detection_system is not None:
            try:
                channel_count = self._detection_system._audio_capture.enabled_channels_count
            except Exception:
                channel_count = 8
        self._audio_meter = AudioLevelMeter(channels=channel_count)
        self._audio_meter.setMinimumHeight(60)
        self._audio_meter.setMaximumHeight(100)
        self._audio_dock.setWidget(self._audio_meter)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._audio_dock)

        # Apply dark theme
        self._apply_theme()

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_session = QAction("&New Session", self)
        new_session.setShortcut(QKeySequence.StandardKey.New)
        new_session.triggered.connect(self._on_new_session)
        file_menu.addAction(new_session)

        open_action = QAction("&Open Recording...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_recording)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Data...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self._settings_action = QAction("&Settings Panel", self)
        self._settings_action.setCheckable(True)
        self._settings_action.setChecked(True)
        self._settings_action.triggered.connect(self._toggle_settings)
        view_menu.addAction(self._settings_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Detection menu
        detection_menu = menubar.addMenu("&Detection")

        self._start_action = QAction("&Start Detection", self)
        self._start_action.setShortcut(QKeySequence("F5"))
        self._start_action.triggered.connect(self._start_detection)
        detection_menu.addAction(self._start_action)

        self._stop_action = QAction("S&top Detection", self)
        self._stop_action.setShortcut(QKeySequence("F6"))
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._stop_detection)
        detection_menu.addAction(self._stop_action)

        detection_menu.addSeparator()

        clear_action = QAction("&Clear Alerts", self)
        clear_action.triggered.connect(self._alerts.clear_alerts)
        detection_menu.addAction(clear_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        calibration_action = QAction("&Calibration...", self)
        calibration_action.triggered.connect(self._show_calibration)
        tools_menu.addAction(calibration_action)

        array_action = QAction("&Array Configuration...", self)
        array_action.triggered.connect(self._show_array_config)
        tools_menu.addAction(array_action)

        tools_menu.addSeparator()

        batch_action = QAction("&Batch Analysis...", self)
        batch_action.triggered.connect(self._show_batch_analysis)
        tools_menu.addAction(batch_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start/Stop buttons
        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a5a2a;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3a7a3a;
            }
        """)
        self._start_btn.clicked.connect(self._start_detection)
        toolbar.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a2a2a;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #7a3a3a;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
            }
        """)
        self._stop_btn.clicked.connect(self._stop_detection)
        toolbar.addWidget(self._stop_btn)

        toolbar.addSeparator()

        # Load Files button
        self._load_btn = QPushButton("Load Files")
        self._load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a6e;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a8e;
            }
        """)
        self._load_btn.clicked.connect(self._on_open_recording)
        toolbar.addWidget(self._load_btn)

        # Settings button
        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a6e;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a8e;
            }
        """)
        self._settings_btn.clicked.connect(self._toggle_settings)
        toolbar.addWidget(self._settings_btn)

        # Export button
        self._export_btn = QPushButton("Export")
        self._export_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a6e;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a8e;
            }
        """)
        self._export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(self._export_btn)

        toolbar.addSeparator()

        # Quick settings
        toolbar.addWidget(QLabel(" Range: "))
        self._range_label = QLabel("500m")
        toolbar.addWidget(self._range_label)

        toolbar.addSeparator()

        # Recording indicator
        self._mini_recording = MiniRecordingWidget()
        self._mini_recording.clicked.connect(self._show_recording_panel)
        toolbar.addWidget(self._mini_recording)

    def _setup_statusbar(self):
        """Setup status bar."""
        statusbar = self.statusBar()

        # Status indicator
        self._status_indicator = StatusIndicator()
        statusbar.addWidget(self._status_indicator)

        # Separator
        statusbar.addWidget(QLabel(" | "))

        # FPS
        self._fps_label = QLabel("FPS: --")
        statusbar.addWidget(self._fps_label)

        # Latency
        self._latency_label = QLabel("Latency: --ms")
        statusbar.addWidget(self._latency_label)

        # Spacer
        statusbar.addWidget(QLabel(""), 1)

        # Target count
        self._target_label = QLabel("Targets: 0")
        statusbar.addPermanentWidget(self._target_label)

        # CPU/GPU
        self._cpu_label = QLabel("CPU: --%")
        statusbar.addPermanentWidget(self._cpu_label)

        self._gpu_label = QLabel("GPU: --")
        statusbar.addPermanentWidget(self._gpu_label)

    def _connect_signals(self):
        """Connect internal signals."""
        pass

    def _apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QMenuBar {
                background-color: #2a2a4e;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #3a3a6e;
            }
            QMenu {
                background-color: #2a2a4e;
                color: #e0e0e0;
            }
            QMenu::item:selected {
                background-color: #3a3a6e;
            }
            QToolBar {
                background-color: #2a2a4e;
                border: none;
                spacing: 5px;
                padding: 5px;
            }
            QStatusBar {
                background-color: #2a2a4e;
            }
            QDockWidget {
                color: #e0e0e0;
            }
            QDockWidget::title {
                background-color: #2a2a4e;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a6e;
            }
            QTabBar::tab {
                background-color: #2a2a4e;
                color: #e0e0e0;
                padding: 8px 15px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a6e;
            }
            QGroupBox {
                border: 1px solid #3a3a6e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #00ff88;
                subcontrol-origin: margin;
                left: 10px;
            }
            QPushButton {
                background-color: #3a3a6e;
                color: #e0e0e0;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a8e;
            }
            QPushButton:pressed {
                background-color: #2a2a4e;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #2a2a4e;
                color: #e0e0e0;
                border: 1px solid #3a3a6e;
                padding: 3px;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                background-color: #2a2a4e;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #00ff88;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollBar {
                background-color: #1a1a2e;
            }
            QScrollBar::handle {
                background-color: #3a3a6e;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #1a1a2e;
                gridline-color: #3a3a6e;
            }
            QHeaderView::section {
                background-color: #2a2a4e;
                color: #00ff88;
                padding: 5px;
                border: none;
            }
        """)

    def _start_detection(self):
        """Start detection system."""
        if self._detection_system:
            success = self._detection_system.start()
            if success:
                self._running = True
                self._update_timer.start(33)  # ~30 FPS
                self._status_indicator.set_status("running")
                self._start_btn.setEnabled(False)
                self._stop_btn.setEnabled(True)
                self._start_action.setEnabled(False)
                self._stop_action.setEnabled(True)
        else:
            # Demo mode without system
            self._running = True
            self._update_timer.start(33)
            self._status_indicator.set_status("running")
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)

    def _stop_detection(self):
        """Stop detection system."""
        if self._detection_system:
            self._detection_system.stop()

        self._running = False
        self._update_timer.stop()
        self._status_indicator.set_status("stopped")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)

    def _on_detection(self, event):
        """Handle detection event from system."""
        # Update radar
        target = RadarTarget(
            track_id=event.track_id or 0,
            azimuth=event.azimuth,
            elevation=event.elevation,
            distance=event.distance,
            confidence=event.confidence,
            classification=event.classification,
            threat_level=event.threat_level
        )
        self._radar.update_target(target)

        # Update 3D view
        import math
        az_rad = math.radians(event.azimuth)
        x = event.distance * math.cos(az_rad)
        y = event.distance * math.sin(az_rad)
        z = event.distance * math.tan(math.radians(event.elevation))

        target_3d = Target3D(
            track_id=event.track_id or 0,
            x=x,
            y=y,
            z=z,
            classification=event.classification,
            threat_level=event.threat_level,
            confidence=event.confidence
        )
        self._view_3d.update_target(target_3d)

        # Add alert
        alert = AlertEvent(
            timestamp=event.timestamp,
            track_id=event.track_id,
            azimuth=event.azimuth,
            elevation=event.elevation,
            distance=event.distance,
            confidence=event.confidence,
            classification=event.classification,
            threat_level=event.threat_level,
            snr=event.snr
        )
        self._alerts.add_alert(alert)

    def _update_display(self):
        """Update display with current data."""
        if self._detection_system and self._detection_system.is_running:
            metrics = self._detection_system.performance_metrics

            if metrics:
                self._fps_label.setText(f"FPS: {metrics.fps:.1f}")
                self._latency_label.setText(f"Latency: {metrics.processing_latency_ms:.1f}ms")
                self._cpu_label.setText(f"CPU: {metrics.cpu_percent:.1f}%")

                if metrics.gpu_percent is not None:
                    self._gpu_label.setText(f"GPU: {metrics.gpu_percent:.1f}%")
                else:
                    self._gpu_label.setText("GPU: N/A")

            # Update target count
            tracks = self._detection_system.get_tracks()
            self._target_label.setText(f"Targets: {len(tracks)}")

    def _on_target_selected(self, track_id: int):
        """Handle target selection on radar."""
        pass

    def _on_alert_selected(self, track_id: int):
        """Handle alert selection."""
        pass

    def _apply_settings(self):
        """Apply settings changes."""
        if not self._detection_system:
            return

        try:
            audio_settings = self._settings_panel.get_audio_settings()
            detection_settings = self._settings_panel.get_detection_settings()
            display_settings = self._settings_panel.get_display_settings()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Settings", str(e))
            return

        # Update config via ConfigManager
        try:
            cfg_mgr = self._detection_system._config_manager
            cfg_mgr.update(
                "audio",
                device_index=audio_settings.device_index if audio_settings.device_index >= 0 else None,
                sample_rate=audio_settings.sample_rate,
                buffer_size=audio_settings.buffer_size,
                num_channels=audio_settings.num_channels,
                requested_channels=audio_settings.requested_channels,
                enabled_channels=audio_settings.enabled_channels,
                channel_map=audio_settings.channel_map
            )
            cfg_mgr.update(
                "detection",
                freq_min=detection_settings.min_frequency,
                freq_max=detection_settings.max_frequency,
                min_confidence=detection_settings.min_confidence,
                detection_threshold=detection_settings.detection_threshold,
                max_detection_range=detection_settings.max_range
            )
            cfg_mgr.save()
        except Exception as e:
            QMessageBox.critical(self, "Apply Failed", str(e))
            return

        was_running = self._detection_system.is_running
        if was_running:
            self._detection_system.stop()

        # Recreate detection system to apply audio changes
        config_path = getattr(cfg_mgr, "_config_path", None)
        try:
            self._detection_system = self._detection_system.__class__(
                config_path=config_path,
                array_config_path=None
            )
            self._detection_system.add_callback(self._on_detection)

            # Update audio meter channels
            try:
                self._audio_meter.set_channels(
                    self._detection_system._audio_capture.enabled_channels_count
                )
            except Exception:
                pass

            if was_running:
                self._detection_system.start()

            QMessageBox.information(self, "Settings Applied", "Settings applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Restart Failed", str(e))

    def _on_recording_started(self, filename: str):
        """Handle recording started."""
        self._mini_recording.set_recording(True)

    def _on_recording_stopped(self, filename: str):
        """Handle recording stopped."""
        self._mini_recording.set_recording(False)

    def _toggle_settings(self):
        """Toggle settings panel visibility."""
        self._settings_dock.setVisible(self._settings_action.isChecked())

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_new_session(self):
        """Start new session."""
        reply = QMessageBox.question(
            self,
            "New Session",
            "Start a new session? Current data will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._alerts.clear_alerts()
            self._radar.clear_targets()
            self._view_3d.clear_targets()

    def _on_open_recording(self):
        """Open recording file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Recording",
            "",
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)"
        )

        if filename:
            # Open recording for analysis
            pass

    def _on_export(self):
        """Export detection data."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            if self._detection_system:
                self._detection_system.export_data(filename)

    def _show_calibration(self):
        """Show calibration dialog."""
        QMessageBox.information(
            self,
            "Calibration",
            "Calibration wizard not yet implemented."
        )

    def _show_array_config(self):
        """Show array configuration dialog."""
        QMessageBox.information(
            self,
            "Array Configuration",
            "Array configuration editor not yet implemented."
        )

    def _show_batch_analysis(self):
        """Show batch analysis dialog."""
        QMessageBox.information(
            self,
            "Batch Analysis",
            "Batch analysis tool not yet implemented."
        )

    def _show_recording_panel(self):
        """Show recording panel."""
        self._settings_dock.show()
        # Switch to recording tab
        tabs = self._settings_dock.widget()
        if isinstance(tabs, QTabWidget):
            tabs.setCurrentIndex(1)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Acoustic Drone Detection",
            """<h2>Acoustic Drone Detection System</h2>
            <p>Version 1.0.0</p>
            <p>A professional-grade acoustic drone detection,
            localization, and classification system.</p>
            <p>Features:</p>
            <ul>
            <li>Multi-channel audio processing</li>
            <li>DOA estimation (MUSIC, MVDR, SRP-PHAT)</li>
            <li>Multi-target tracking (EKF, UKF)</li>
            <li>Drone classification</li>
            <li>GPU acceleration</li>
            </ul>
            """
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self._running:
            reply = QMessageBox.question(
                self,
                "Exit",
                "Detection is running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self._stop_detection()

        event.accept()


def main(detection_system=None):
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("Acoustic Drone Detection")
    app.setStyle("Fusion")

    window = MainWindow(detection_system)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
