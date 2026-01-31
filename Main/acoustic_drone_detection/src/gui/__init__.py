"""
GUI Module

PyQt6-based graphical user interface:
- MainWindow: Main application window
- RadarDisplay: Radar/PPI display widget
- SpectrumView: Spectrum analyzer and spectrogram
- Visualization3D: 3D target visualization
- SettingsPanel: Configuration interface
- AlertsPanel: Detection alerts display
- RecordingPanel: Audio recording controls
"""

from .main_window import MainWindow, main
from .radar_display import RadarDisplay, RadarTarget, MiniRadarDisplay
from .spectrum_view import SpectrumView, SpectrogramView, CombinedSpectrumWidget
from .visualization_3d import Simple3DView, Visualization3DWidget, Target3D
from .settings_panel import SettingsPanel, AudioSettings, DetectionSettings, DisplaySettings
from .alerts_panel import AlertsPanel, AlertEvent, AlertCard, MiniAlertsWidget
from .recording_panel import RecordingPanel, RecordingSettings, AudioLevelMeter

__all__ = [
    # Main window
    'MainWindow',
    'main',

    # Radar
    'RadarDisplay',
    'RadarTarget',
    'MiniRadarDisplay',

    # Spectrum
    'SpectrumView',
    'SpectrogramView',
    'CombinedSpectrumWidget',

    # 3D View
    'Simple3DView',
    'Visualization3DWidget',
    'Target3D',

    # Settings
    'SettingsPanel',
    'AudioSettings',
    'DetectionSettings',
    'DisplaySettings',

    # Alerts
    'AlertsPanel',
    'AlertEvent',
    'AlertCard',
    'MiniAlertsWidget',

    # Recording
    'RecordingPanel',
    'RecordingSettings',
    'AudioLevelMeter'
]
