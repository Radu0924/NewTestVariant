"""
Configuration Manager Module

Handles loading, saving, and managing YAML configuration files.
Provides centralized access to all system configuration parameters.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
import copy
import threading


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 48000
    bit_depth: int = 24
    num_channels: int = 12
    buffer_size: int = 512
    device_name: Optional[str] = None
    device_index: Optional[int] = None
    requested_channels: Optional[int] = None
    enabled_channels: Optional[list] = None
    channel_map: Optional[list] = None


@dataclass
class ArrayConfig:
    """Microphone array configuration."""
    geometry_type: str = "circular"  # circular, spherical, planar, linear, custom
    num_microphones: int = 12
    mic_spacing: float = 0.10  # meters
    positions: list = field(default_factory=list)
    center_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class DetectionConfig:
    """Detection parameters configuration."""
    freq_min: int = 80
    freq_max: int = 8000
    detection_threshold: float = 0.5
    min_confidence: float = 0.3
    max_detection_range: float = 500.0
    angular_precision: str = "standard"  # standard, high, ultra
    update_rate: int = 30  # Hz
    tdoa_min_confidence: float = 0.15  # Min TDOA confidence for MLAT
    target_types: list = field(default_factory=lambda: [
        "quadcopter", "hexacopter", "fpv_racing", "fixed_wing", "unknown"
    ])


@dataclass
class EnvironmentConfig:
    """Environment preset configuration."""
    preset: str = "outdoor_calm"  # indoor, outdoor_calm, outdoor_windy, urban, rural
    reverb_compensation: bool = False
    wind_filter: str = "off"  # off, light, aggressive, adaptive
    traffic_filter: bool = False
    rain_filter: bool = False
    emi_filter: bool = False
    machinery_filter: bool = False
    expected_noise_floor: str = "medium"  # low, medium, high


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    gpu_mode: str = "auto"  # auto, force_gpu, force_cpu
    gpu_signal_processing: bool = True
    gpu_ml_inference: bool = True
    processing_priority: str = "high"  # low, normal, high
    num_threads: int = 4


@dataclass
class RecordingConfig:
    """Recording configuration."""
    auto_record_on_detection: bool = True
    recording_format: str = "flac"  # wav, flac, mp3, ogg
    save_location: str = "data/recordings"
    pre_buffer_seconds: float = 5.0
    post_buffer_seconds: float = 10.0


@dataclass
class TrackingConfig:
    """Multi-target tracking configuration."""
    algorithm: str = "ekf"  # ekf, ukf, particle, mht
    max_targets: int = 10
    track_timeout: float = 5.0  # seconds
    association_threshold: float = 20.0  # degrees
    prediction_model: str = "linear"  # linear, curvilinear


@dataclass
class SystemConfig:
    """Main system configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    array: ArrayConfig = field(default_factory=ArrayConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


class ConfigManager:
    """
    Centralized configuration manager for the drone detection system.

    Provides thread-safe access to configuration parameters with support for:
    - Loading from YAML files
    - Saving configurations
    - Runtime configuration updates
    - Configuration validation
    """

    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()

    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern for global configuration access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the main configuration YAML file.
        """
        if self._initialized:
            return

        self._config_path = config_path
        self._config = SystemConfig()
        self._observers: list = []
        self._config_lock = threading.RLock()

        if config_path and os.path.exists(config_path):
            self.load(config_path)

        self._initialized = True

    @property
    def config(self) -> SystemConfig:
        """Get the current configuration."""
        with self._config_lock:
            return copy.deepcopy(self._config)

    def load(self, config_path: str) -> bool:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            True if configuration was loaded successfully.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            with self._config_lock:
                self._apply_dict_to_config(data)
                self._validate_audio_config(self._config.audio)
                self._config_path = config_path

            self._notify_observers()
            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to a YAML file.

        Args:
            config_path: Path to save to. Uses current path if not specified.

        Returns:
            True if configuration was saved successfully.
        """
        save_path = config_path or self._config_path
        if not save_path:
            return False

        try:
            with self._config_lock:
                data = self._config_to_dict()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            return True

        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def update(self, section: str, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            section: Configuration section name (audio, detection, etc.)
            **kwargs: Key-value pairs to update.
        """
        with self._config_lock:
            section_config = getattr(self._config, section, None)
            if section_config:
                for key, value in kwargs.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

            # Validate after update
            self._validate_audio_config(self._config.audio)

        self._notify_observers()

    def _validate_audio_config(self, audio_config: AudioConfig) -> None:
        """Validate audio configuration for channel selection."""
        enabled = audio_config.enabled_channels
        channel_map = audio_config.channel_map

        if channel_map is not None:
            enabled_list = list(channel_map)
        elif enabled is not None:
            enabled_list = list(enabled)
        else:
            return

        if len(enabled_list) < 8:
            raise ValueError("At least 8 enabled channels are required")

        # Determine upper bound for indices
        if audio_config.requested_channels is not None:
            max_channels = audio_config.requested_channels
        else:
            max_channels = audio_config.num_channels

        if max_channels is not None and max_channels > 0:
            if max_channels < len(enabled_list):
                raise ValueError(
                    "requested_channels/num_channels is smaller than enabled channels count"
                )

        seen = set()
        for idx in enabled_list:
            if not isinstance(idx, int):
                raise ValueError("Enabled channel indices must be integers")
            if idx < 0:
                raise ValueError("Enabled channel indices must be >= 0")
            if max_channels is not None and max_channels > 0 and idx >= max_channels:
                raise ValueError(
                    "Enabled channel index exceeds requested/num_channels"
                )
            if idx in seen:
                raise ValueError("Duplicate channel index detected in enabled channels")
            seen.add(idx)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            section: Configuration section name.
            key: Configuration key within the section.
            default: Default value if not found.

        Returns:
            Configuration value or default.
        """
        with self._config_lock:
            section_config = getattr(self._config, section, None)
            if section_config:
                return getattr(section_config, key, default)
            return default

    def add_observer(self, callback) -> None:
        """Add an observer to be notified of configuration changes."""
        self._observers.append(callback)

    def remove_observer(self, callback) -> None:
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self) -> None:
        """Notify all observers of configuration changes."""
        for observer in self._observers:
            try:
                observer(self._config)
            except Exception as e:
                print(f"Error notifying observer: {e}")

    def _apply_dict_to_config(self, data: Dict[str, Any]) -> None:
        """Apply dictionary data to configuration dataclasses."""
        for section_name, section_data in data.items():
            if hasattr(self._config, section_name) and isinstance(section_data, dict):
                section_config = getattr(self._config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration dataclasses to dictionary."""
        return {
            'audio': asdict(self._config.audio),
            'array': asdict(self._config.array),
            'detection': asdict(self._config.detection),
            'environment': asdict(self._config.environment),
            'performance': asdict(self._config.performance),
            'recording': asdict(self._config.recording),
            'tracking': asdict(self._config.tracking),
        }

    def load_array_config(self, array_config_path: str) -> bool:
        """
        Load microphone array configuration from a separate file.

        Args:
            array_config_path: Path to the array configuration YAML file.

        Returns:
            True if loaded successfully.
        """
        try:
            with open(array_config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            with self._config_lock:
                for key, value in data.items():
                    if hasattr(self._config.array, key):
                        setattr(self._config.array, key, value)

            self._notify_observers()
            return True

        except Exception as e:
            print(f"Error loading array configuration: {e}")
            return False

    def load_drone_profile(self, profile_path: str) -> Dict[str, Any]:
        """
        Load a drone acoustic profile from file.

        Args:
            profile_path: Path to the drone profile YAML file.

        Returns:
            Dictionary containing the drone profile data.
        """
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading drone profile: {e}")
            return {}

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Audio validation
        if self._config.audio.sample_rate not in [44100, 48000, 96000, 192000]:
            errors.append(f"Invalid sample rate: {self._config.audio.sample_rate}")

        if not 8 <= self._config.audio.num_channels <= 20:
            errors.append(f"Number of channels must be 8-20: {self._config.audio.num_channels}")

        # Detection validation
        if self._config.detection.freq_min >= self._config.detection.freq_max:
            errors.append("Minimum frequency must be less than maximum frequency")

        if not 0.0 <= self._config.detection.detection_threshold <= 1.0:
            errors.append("Detection threshold must be between 0 and 1")

        # Array validation
        if self._config.array.num_microphones < 4:
            errors.append("At least 4 microphones required for 3D localization")

        return len(errors) == 0, errors

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        with self._config_lock:
            self._config = SystemConfig()
        self._notify_observers()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return ConfigManager()
