"""
Utilities Package

Provides utility modules for configuration, logging, data export, and performance monitoring.
"""

from .config_manager import (
    ConfigManager,
    SystemConfig,
    AudioConfig,
    ArrayConfig,
    DetectionConfig,
    EnvironmentConfig,
    PerformanceConfig,
    RecordingConfig,
    TrackingConfig,
    get_config
)

from .logger import (
    DroneDetectionLogger,
    PerformanceTimer,
    get_logger,
    setup_logging
)

from .data_export import (
    DetectionRecord,
    JSONExporter,
    CSVExporter,
    SQLiteExporter,
    DataExportManager,
    create_detection_record
)

from .performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    TimingContext,
    get_monitor,
    timing
)

__all__ = [
    # Config
    'ConfigManager', 'SystemConfig', 'AudioConfig', 'ArrayConfig',
    'DetectionConfig', 'EnvironmentConfig', 'PerformanceConfig',
    'RecordingConfig', 'TrackingConfig', 'get_config',
    # Logger
    'DroneDetectionLogger', 'PerformanceTimer', 'get_logger', 'setup_logging',
    # Export
    'DetectionRecord', 'JSONExporter', 'CSVExporter', 'SQLiteExporter',
    'DataExportManager', 'create_detection_record',
    # Performance
    'PerformanceMetrics', 'PerformanceMonitor', 'TimingContext',
    'get_monitor', 'timing'
]
