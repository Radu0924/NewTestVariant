"""
Settings Panel Widget

Configuration and settings interface:
- Audio device selection
- Detection parameters
- Display options
- Array configuration
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QCheckBox, QPushButton, QTabWidget, QScrollArea,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AudioSettings:
    """Audio configuration."""
    device_index: int = -1
    sample_rate: int = 48000
    buffer_size: int = 512
    num_channels: int = 8
    requested_channels: Optional[int] = None
    enabled_channels: Optional[list] = None
    channel_map: Optional[list] = None


@dataclass
class DetectionSettings:
    """Detection configuration."""
    min_frequency: int = 80
    max_frequency: int = 8000
    min_confidence: float = 0.5
    max_range: float = 500.0
    detection_threshold: float = 0.5


@dataclass
class DisplaySettings:
    """Display configuration."""
    update_rate: int = 30
    show_spectrum: bool = True
    show_spectrogram: bool = True
    show_3d_view: bool = True
    radar_range: float = 500.0
    sweep_enabled: bool = True


class SettingsPanel(QWidget):
    """
    Settings and configuration panel.

    Provides controls for all system parameters.
    """

    # Signals
    settings_changed = pyqtSignal(str, str, object)  # section, key, value
    audio_device_changed = pyqtSignal(int)
    apply_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._audio_devices: list = []
        self._modified = False

        self._setup_ui()
        self._refresh_audio_devices()
        self._validate_channel_inputs()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()

        # Audio tab
        audio_widget = self._create_audio_tab()
        tabs.addTab(audio_widget, "Audio")

        # Detection tab
        detection_widget = self._create_detection_tab()
        tabs.addTab(detection_widget, "Detection")

        # Display tab
        display_widget = self._create_display_tab()
        tabs.addTab(display_widget, "Display")

        # Array tab
        array_widget = self._create_array_tab()
        tabs.addTab(array_widget, "Array")

        # Advanced tab
        advanced_widget = self._create_advanced_tab()
        tabs.addTab(advanced_widget, "Advanced")

        # Environment tab
        environment_widget = self._create_environment_tab()
        tabs.addTab(environment_widget, "Environment")

        layout.addWidget(tabs)

        # Buttons
        buttons = QHBoxLayout()

        self._refresh_btn = QPushButton("Refresh Devices")
        self._refresh_btn.clicked.connect(self._refresh_audio_devices)
        buttons.addWidget(self._refresh_btn)

        buttons.addStretch()

        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._on_reset)
        buttons.addWidget(self._reset_btn)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(self._apply_btn)

        layout.addLayout(buttons)

    def _create_audio_tab(self) -> QWidget:
        """Create audio settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Device selection
        device_group = QGroupBox("Audio Device")
        device_layout = QFormLayout(device_group)

        self._device_combo = QComboBox()
        self._device_combo.currentIndexChanged.connect(self._on_setting_changed)
        device_layout.addRow("Input Device:", self._device_combo)

        layout.addWidget(device_group)

        # Sample rate
        format_group = QGroupBox("Audio Format")
        format_layout = QFormLayout(format_group)

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItems(["44100", "48000", "96000", "192000"])
        self._sample_rate_combo.setCurrentText("48000")
        self._sample_rate_combo.currentIndexChanged.connect(self._on_setting_changed)
        format_layout.addRow("Sample Rate:", self._sample_rate_combo)

        self._buffer_spin = QSpinBox()
        self._buffer_spin.setRange(64, 4096)
        self._buffer_spin.setValue(512)
        self._buffer_spin.setSingleStep(64)
        self._buffer_spin.valueChanged.connect(self._on_setting_changed)
        format_layout.addRow("Buffer Size:", self._buffer_spin)

        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(8, 32)
        self._channels_spin.setValue(8)
        self._channels_spin.valueChanged.connect(self._on_setting_changed)
        format_layout.addRow("Channels:", self._channels_spin)

        layout.addWidget(format_group)

        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QFormLayout(channel_group)

        self._requested_channels_spin = QSpinBox()
        self._requested_channels_spin.setRange(8, 32)
        self._requested_channels_spin.setValue(8)
        self._requested_channels_spin.valueChanged.connect(self._on_setting_changed)
        channel_layout.addRow("Requested Channels:", self._requested_channels_spin)

        self._enabled_channels_text = QComboBox()
        self._enabled_channels_text.setEditable(True)
        self._enabled_channels_text.addItem("")
        self._enabled_channels_text.lineEdit().setPlaceholderText("e.g. 0,1,2,3,4,5,6,7")
        self._enabled_channels_text.currentIndexChanged.connect(self._on_setting_changed)
        channel_layout.addRow("Enabled Channels:", self._enabled_channels_text)

        self._channel_map_text = QComboBox()
        self._channel_map_text.setEditable(True)
        self._channel_map_text.addItem("")
        self._channel_map_text.lineEdit().setPlaceholderText("optional map, e.g. 3,2,1,0,...")
        self._channel_map_text.currentIndexChanged.connect(self._on_setting_changed)
        channel_layout.addRow("Channel Map:", self._channel_map_text)

        layout.addWidget(channel_group)
        layout.addStretch()

        return widget

    def _create_detection_tab(self) -> QWidget:
        """Create detection settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Frequency range
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QFormLayout(freq_group)

        self._min_freq_spin = QSpinBox()
        self._min_freq_spin.setRange(20, 1000)
        self._min_freq_spin.setValue(80)
        self._min_freq_spin.setSuffix(" Hz")
        self._min_freq_spin.valueChanged.connect(self._on_setting_changed)
        freq_layout.addRow("Minimum:", self._min_freq_spin)

        self._max_freq_spin = QSpinBox()
        self._max_freq_spin.setRange(1000, 24000)
        self._max_freq_spin.setValue(8000)
        self._max_freq_spin.setSuffix(" Hz")
        self._max_freq_spin.valueChanged.connect(self._on_setting_changed)
        freq_layout.addRow("Maximum:", self._max_freq_spin)

        layout.addWidget(freq_group)

        # Detection parameters
        detect_group = QGroupBox("Detection Parameters")
        detect_layout = QFormLayout(detect_group)

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.1, 1.0)
        self._confidence_spin.setValue(0.5)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.valueChanged.connect(self._on_setting_changed)
        detect_layout.addRow("Min Confidence:", self._confidence_spin)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.1, 1.0)
        self._threshold_spin.setValue(0.5)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.valueChanged.connect(self._on_setting_changed)
        detect_layout.addRow("Detection Threshold:", self._threshold_spin)

        self._max_range_spin = QSpinBox()
        self._max_range_spin.setRange(50, 2000)
        self._max_range_spin.setValue(500)
        self._max_range_spin.setSuffix(" m")
        self._max_range_spin.valueChanged.connect(self._on_setting_changed)
        detect_layout.addRow("Max Range:", self._max_range_spin)

        layout.addWidget(detect_group)

        # Tracking
        track_group = QGroupBox("Tracking")
        track_layout = QFormLayout(track_group)

        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItems(["EKF", "UKF", "Particle Filter"])
        self._algorithm_combo.currentIndexChanged.connect(self._on_setting_changed)
        track_layout.addRow("Algorithm:", self._algorithm_combo)

        self._max_targets_spin = QSpinBox()
        self._max_targets_spin.setRange(1, 50)
        self._max_targets_spin.setValue(10)
        self._max_targets_spin.valueChanged.connect(self._on_setting_changed)
        track_layout.addRow("Max Targets:", self._max_targets_spin)

        self._track_timeout_spin = QDoubleSpinBox()
        self._track_timeout_spin.setRange(0.5, 30.0)
        self._track_timeout_spin.setValue(5.0)
        self._track_timeout_spin.setSuffix(" s")
        self._track_timeout_spin.valueChanged.connect(self._on_setting_changed)
        track_layout.addRow("Track Timeout:", self._track_timeout_spin)

        layout.addWidget(track_group)

        # Target drone types
        drone_types_group = QGroupBox("Target Drone Types")
        drone_types_layout = QVBoxLayout(drone_types_group)

        self._quadcopter_check = QCheckBox("Quadcopter")
        self._quadcopter_check.setChecked(True)
        self._quadcopter_check.stateChanged.connect(self._on_setting_changed)
        drone_types_layout.addWidget(self._quadcopter_check)

        self._hexacopter_check = QCheckBox("Hexacopter")
        self._hexacopter_check.setChecked(True)
        self._hexacopter_check.stateChanged.connect(self._on_setting_changed)
        drone_types_layout.addWidget(self._hexacopter_check)

        self._fixed_wing_check = QCheckBox("Fixed Wing")
        self._fixed_wing_check.setChecked(True)
        self._fixed_wing_check.stateChanged.connect(self._on_setting_changed)
        drone_types_layout.addWidget(self._fixed_wing_check)

        self._vtol_check = QCheckBox("VTOL")
        self._vtol_check.setChecked(True)
        self._vtol_check.stateChanged.connect(self._on_setting_changed)
        drone_types_layout.addWidget(self._vtol_check)

        layout.addWidget(drone_types_group)
        layout.addStretch()

        return widget

    def _create_display_tab(self) -> QWidget:
        """Create display settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Update rate
        rate_group = QGroupBox("Performance")
        rate_layout = QFormLayout(rate_group)

        self._update_rate_spin = QSpinBox()
        self._update_rate_spin.setRange(10, 60)
        self._update_rate_spin.setValue(30)
        self._update_rate_spin.setSuffix(" Hz")
        self._update_rate_spin.valueChanged.connect(self._on_setting_changed)
        rate_layout.addRow("Update Rate:", self._update_rate_spin)

        layout.addWidget(rate_group)

        # Panels
        panels_group = QGroupBox("Panels")
        panels_layout = QVBoxLayout(panels_group)

        self._show_spectrum_check = QCheckBox("Show Spectrum Analyzer")
        self._show_spectrum_check.setChecked(True)
        self._show_spectrum_check.stateChanged.connect(self._on_setting_changed)
        panels_layout.addWidget(self._show_spectrum_check)

        self._show_spectrogram_check = QCheckBox("Show Spectrogram")
        self._show_spectrogram_check.setChecked(True)
        self._show_spectrogram_check.stateChanged.connect(self._on_setting_changed)
        panels_layout.addWidget(self._show_spectrogram_check)

        self._show_3d_check = QCheckBox("Show 3D View")
        self._show_3d_check.setChecked(True)
        self._show_3d_check.stateChanged.connect(self._on_setting_changed)
        panels_layout.addWidget(self._show_3d_check)

        layout.addWidget(panels_group)

        # Radar
        radar_group = QGroupBox("Radar Display")
        radar_layout = QFormLayout(radar_group)

        self._radar_range_spin = QSpinBox()
        self._radar_range_spin.setRange(50, 2000)
        self._radar_range_spin.setValue(500)
        self._radar_range_spin.setSuffix(" m")
        self._radar_range_spin.valueChanged.connect(self._on_setting_changed)
        radar_layout.addRow("Display Range:", self._radar_range_spin)

        self._sweep_check = QCheckBox("Sweep Animation")
        self._sweep_check.setChecked(True)
        self._sweep_check.stateChanged.connect(self._on_setting_changed)
        radar_layout.addRow("", self._sweep_check)

        self._trails_check = QCheckBox("Target Trails")
        self._trails_check.setChecked(True)
        self._trails_check.stateChanged.connect(self._on_setting_changed)
        radar_layout.addRow("", self._trails_check)

        layout.addWidget(radar_group)
        layout.addStretch()

        return widget

    def _create_array_tab(self) -> QWidget:
        """Create array configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Array geometry
        geom_group = QGroupBox("Array Geometry")
        geom_layout = QFormLayout(geom_group)

        self._array_type_combo = QComboBox()
        self._array_type_combo.addItems([
            "Circular", "Spherical", "Planar", "Linear", "Custom"
        ])
        self._array_type_combo.currentIndexChanged.connect(self._on_setting_changed)
        geom_layout.addRow("Type:", self._array_type_combo)

        self._num_mics_spin = QSpinBox()
        self._num_mics_spin.setRange(4, 32)
        self._num_mics_spin.setValue(8)
        self._num_mics_spin.valueChanged.connect(self._on_setting_changed)
        geom_layout.addRow("Microphones:", self._num_mics_spin)

        self._array_radius_spin = QDoubleSpinBox()
        self._array_radius_spin.setRange(0.01, 1.0)
        self._array_radius_spin.setValue(0.1)
        self._array_radius_spin.setSingleStep(0.01)
        self._array_radius_spin.setSuffix(" m")
        self._array_radius_spin.valueChanged.connect(self._on_setting_changed)
        geom_layout.addRow("Radius:", self._array_radius_spin)

        layout.addWidget(geom_group)

        # Load/Save
        file_group = QGroupBox("Configuration File")
        file_layout = QHBoxLayout(file_group)

        self._load_array_btn = QPushButton("Load Array Config...")
        self._load_array_btn.clicked.connect(self._on_load_array)
        file_layout.addWidget(self._load_array_btn)

        self._save_array_btn = QPushButton("Save Array Config...")
        self._save_array_btn.clicked.connect(self._on_save_array)
        file_layout.addWidget(self._save_array_btn)

        layout.addWidget(file_group)

        # Calibration
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout(calib_group)

        self._calibrate_btn = QPushButton("Run Calibration...")
        calib_layout.addWidget(self._calibrate_btn)

        self._load_calib_btn = QPushButton("Load Calibration...")
        calib_layout.addWidget(self._load_calib_btn)

        layout.addWidget(calib_group)
        layout.addStretch()

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GPU
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QVBoxLayout(gpu_group)

        self._use_gpu_check = QCheckBox("Enable GPU Acceleration")
        self._use_gpu_check.setChecked(True)
        self._use_gpu_check.stateChanged.connect(self._on_setting_changed)
        gpu_layout.addWidget(self._use_gpu_check)

        self._gpu_status_label = QLabel("GPU: Detecting...")
        gpu_layout.addWidget(self._gpu_status_label)

        layout.addWidget(gpu_group)

        # DOA Algorithm
        doa_group = QGroupBox("DOA Estimation")
        doa_layout = QFormLayout(doa_group)

        self._doa_algo_combo = QComboBox()
        self._doa_algo_combo.addItems([
            "MUSIC", "MVDR", "SRP-PHAT", "ESPRIT", "Delay-Sum"
        ])
        self._doa_algo_combo.currentIndexChanged.connect(self._on_setting_changed)
        doa_layout.addRow("Algorithm:", self._doa_algo_combo)

        self._azimuth_res_spin = QSpinBox()
        self._azimuth_res_spin.setRange(1, 360)
        self._azimuth_res_spin.setValue(360)
        self._azimuth_res_spin.valueChanged.connect(self._on_setting_changed)
        doa_layout.addRow("Azimuth Points:", self._azimuth_res_spin)

        self._elevation_res_spin = QSpinBox()
        self._elevation_res_spin.setRange(1, 180)
        self._elevation_res_spin.setValue(91)
        self._elevation_res_spin.valueChanged.connect(self._on_setting_changed)
        doa_layout.addRow("Elevation Points:", self._elevation_res_spin)

        layout.addWidget(doa_group)

        # Classification
        class_group = QGroupBox("Classification")
        class_layout = QFormLayout(class_group)

        self._classifier_combo = QComboBox()
        self._classifier_combo.addItems(["Rule-Based", "ML (CNN)", "Hybrid"])
        self._classifier_combo.currentIndexChanged.connect(self._on_setting_changed)
        class_layout.addRow("Classifier:", self._classifier_combo)

        self._load_model_btn = QPushButton("Load ML Model...")
        class_layout.addRow("", self._load_model_btn)

        layout.addWidget(class_group)
        layout.addStretch()

        return widget

    def _create_environment_tab(self) -> QWidget:
        """Create environment settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Environment presets
        preset_group = QGroupBox("Environment Preset")
        preset_layout = QVBoxLayout(preset_group)

        self._indoor_radio = QCheckBox("Indoor")
        self._indoor_radio.setChecked(True)
        self._indoor_radio.stateChanged.connect(self._on_environment_preset_changed)
        preset_layout.addWidget(self._indoor_radio)

        self._outdoor_radio = QCheckBox("Outdoor")
        self._outdoor_radio.stateChanged.connect(self._on_environment_preset_changed)
        preset_layout.addWidget(self._outdoor_radio)

        layout.addWidget(preset_group)

        # Environmental noise filters
        filters_group = QGroupBox("Noise Filters")
        filters_layout = QVBoxLayout(filters_group)

        self._wind_filter_check = QCheckBox("Wind Noise Suppression")
        self._wind_filter_check.setChecked(False)
        self._wind_filter_check.stateChanged.connect(self._on_setting_changed)
        filters_layout.addWidget(self._wind_filter_check)

        self._traffic_filter_check = QCheckBox("Traffic Noise Suppression")
        self._traffic_filter_check.setChecked(False)
        self._traffic_filter_check.stateChanged.connect(self._on_setting_changed)
        filters_layout.addWidget(self._traffic_filter_check)

        self._rain_filter_check = QCheckBox("Rain Noise Suppression")
        self._rain_filter_check.setChecked(False)
        self._rain_filter_check.stateChanged.connect(self._on_setting_changed)
        filters_layout.addWidget(self._rain_filter_check)

        self._emi_filter_check = QCheckBox("EMI (Electromagnetic Interference) Filter")
        self._emi_filter_check.setChecked(False)
        self._emi_filter_check.stateChanged.connect(self._on_setting_changed)
        filters_layout.addWidget(self._emi_filter_check)

        layout.addWidget(filters_group)
        layout.addStretch()

        return widget

    def _on_environment_preset_changed(self):
        """Handle environment preset change."""
        sender = self.sender()
        if sender == self._indoor_radio and self._indoor_radio.isChecked():
            self._outdoor_radio.setChecked(False)
            # Indoor preset: disable wind, enable EMI filter
            self._wind_filter_check.setChecked(False)
            self._traffic_filter_check.setChecked(False)
            self._rain_filter_check.setChecked(False)
            self._emi_filter_check.setChecked(True)
        elif sender == self._outdoor_radio and self._outdoor_radio.isChecked():
            self._indoor_radio.setChecked(False)
            # Outdoor preset: enable wind and traffic filters
            self._wind_filter_check.setChecked(True)
            self._traffic_filter_check.setChecked(True)
            self._rain_filter_check.setChecked(False)
            self._emi_filter_check.setChecked(False)
        self._on_setting_changed()

    def _refresh_audio_devices(self):
        """Refresh audio device list."""
        self._device_combo.clear()

        try:
            import sounddevice as sd
            devices = sd.query_devices()

            self._audio_devices = []
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self._audio_devices.append({
                        'index': i,
                        'name': dev['name'],
                        'channels': dev['max_input_channels']
                    })
                    self._device_combo.addItem(
                        f"{dev['name']} ({dev['max_input_channels']} ch)"
                    )

        except Exception as e:
            self._device_combo.addItem("No audio devices found")

    def _on_setting_changed(self):
        """Handle setting change."""
        self._modified = True
        self._validate_channel_inputs()

    def _on_apply(self):
        """Apply settings."""
        self.apply_clicked.emit()
        self._modified = False

    def _on_reset(self):
        """Reset to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.reset_clicked.emit()
            self._load_defaults()

    def _load_defaults(self):
        """Load default values."""
        self._sample_rate_combo.setCurrentText("48000")
        self._buffer_spin.setValue(512)
        self._channels_spin.setValue(8)
        self._requested_channels_spin.setValue(8)
        self._enabled_channels_text.setCurrentText("")
        self._channel_map_text.setCurrentText("")
        self._min_freq_spin.setValue(80)
        self._max_freq_spin.setValue(8000)
        self._confidence_spin.setValue(0.5)
        self._threshold_spin.setValue(0.5)
        self._max_range_spin.setValue(500)
        self._modified = False

    def load_from_config(self, config):
        """Load settings from a SystemConfig instance."""
        try:
            audio_cfg = config.audio
            self._sample_rate_combo.setCurrentText(str(audio_cfg.sample_rate))
            self._buffer_spin.setValue(audio_cfg.buffer_size)
            self._channels_spin.setValue(audio_cfg.num_channels)
            if audio_cfg.requested_channels is None:
                self._requested_channels_spin.setValue(audio_cfg.num_channels)
            else:
                self._requested_channels_spin.setValue(audio_cfg.requested_channels)

            enabled_text = ""
            if audio_cfg.enabled_channels:
                enabled_text = ",".join(str(x) for x in audio_cfg.enabled_channels)
            self._enabled_channels_text.setCurrentText(enabled_text)

            map_text = ""
            if audio_cfg.channel_map:
                map_text = ",".join(str(x) for x in audio_cfg.channel_map)
            self._channel_map_text.setCurrentText(map_text)
        except Exception:
            pass

        self._validate_channel_inputs()

    def _on_load_array(self):
        """Load array configuration."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Array Configuration",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if filename:
            # Load array config
            pass

    def _on_save_array(self):
        """Save array configuration."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Array Configuration",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if filename:
            # Save array config
            pass

    def get_audio_settings(self) -> AudioSettings:
        """Get current audio settings."""
        device_idx = -1
        if self._device_combo.currentIndex() >= 0 and self._audio_devices:
            device_idx = self._audio_devices[self._device_combo.currentIndex()]['index']

        enabled_text = self._enabled_channels_text.currentText().strip()
        channel_map_text = self._channel_map_text.currentText().strip()

        enabled_channels = self._parse_channel_list(enabled_text)
        channel_map = self._parse_channel_list(channel_map_text)

        requested_channels = self._requested_channels_spin.value()
        if requested_channels == self._channels_spin.value():
            requested_channels = None

        return AudioSettings(
            device_index=device_idx,
            sample_rate=int(self._sample_rate_combo.currentText()),
            buffer_size=self._buffer_spin.value(),
            num_channels=self._channels_spin.value(),
            requested_channels=requested_channels,
            enabled_channels=enabled_channels,
            channel_map=channel_map
        )

    @staticmethod
    def _parse_channel_list(text: str) -> Optional[list]:
        """Parse comma-separated channel list into int list."""
        if not text:
            return None
        cleaned = text.replace(" ", "")
        if not cleaned:
            return None
        parts = [p for p in cleaned.split(",") if p != ""]
        channels = []
        for part in parts:
            try:
                channels.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid channel index: {part}")
        # Check duplicates
        if len(set(channels)) != len(channels):
            raise ValueError("Duplicate channel indices are not allowed")
        return channels

    def _validate_channel_inputs(self) -> None:
        """Validate channel list inputs and update UI."""
        enabled_text = self._enabled_channels_text.currentText().strip()
        map_text = self._channel_map_text.currentText().strip()

        def set_field_state(combo: QComboBox, ok: bool) -> None:
            edit = combo.lineEdit()
            if not edit:
                return
            if ok:
                edit.setStyleSheet("")
            else:
                edit.setStyleSheet("border: 1px solid red;")

        ok_enabled = True
        if enabled_text:
            try:
                channels = self._parse_channel_list(enabled_text)
                if channels is not None and len(channels) < 8:
                    ok_enabled = False
            except ValueError:
                ok_enabled = False
        set_field_state(self._enabled_channels_text, ok_enabled)

        ok_map = True
        if map_text:
            try:
                channels = self._parse_channel_list(map_text)
                if channels is not None and len(channels) < 8:
                    ok_map = False
            except ValueError:
                ok_map = False
        set_field_state(self._channel_map_text, ok_map)

    def get_detection_settings(self) -> DetectionSettings:
        """Get current detection settings."""
        return DetectionSettings(
            min_frequency=self._min_freq_spin.value(),
            max_frequency=self._max_freq_spin.value(),
            min_confidence=self._confidence_spin.value(),
            max_range=self._max_range_spin.value(),
            detection_threshold=self._threshold_spin.value()
        )

    def get_display_settings(self) -> DisplaySettings:
        """Get current display settings."""
        return DisplaySettings(
            update_rate=self._update_rate_spin.value(),
            show_spectrum=self._show_spectrum_check.isChecked(),
            show_spectrogram=self._show_spectrogram_check.isChecked(),
            show_3d_view=self._show_3d_check.isChecked(),
            radar_range=self._radar_range_spin.value(),
            sweep_enabled=self._sweep_check.isChecked()
        )

    def set_gpu_status(self, available: bool, name: str = ""):
        """Update GPU status display."""
        if available:
            self._gpu_status_label.setText(f"GPU: {name}")
            self._gpu_status_label.setStyleSheet("color: green;")
        else:
            self._gpu_status_label.setText("GPU: Not available")
            self._gpu_status_label.setStyleSheet("color: gray;")
