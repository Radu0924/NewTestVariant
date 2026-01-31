"""
Hardware Abstraction Package

Provides hardware abstraction for audio interfaces, microphone arrays,
calibration, and GPU management.
"""

from .audio_interface import (
    AudioInterfaceManager,
    AudioInputStream,
    AudioStreamConfig,
    AudioDeviceCapabilities
)

from .array_geometry import (
    ArrayGeometryManager,
    ArrayGeometry,
    MicrophonePosition,
    GeometryGenerator,
    GeometryValidator,
    GeometryIO
)

from .calibration import (
    CalibrationManager,
    ArrayCalibration,
    MicrophoneCalibration,
    SensitivityCalibrator,
    PhaseCalibrator,
    NoiseFloorAnalyzer
)

from .gpu_detector import (
    GPUManager,
    GPUInfo,
    GPUBackend,
    CUDADetector,
    PyTorchGPUDetector,
    CuPyDetector,
    get_gpu_manager,
    is_gpu_available,
    should_use_gpu
)

__all__ = [
    # Audio Interface
    'AudioInterfaceManager', 'AudioInputStream', 'AudioStreamConfig',
    'AudioDeviceCapabilities',
    # Array Geometry
    'ArrayGeometryManager', 'ArrayGeometry', 'MicrophonePosition',
    'GeometryGenerator', 'GeometryValidator', 'GeometryIO',
    # Calibration
    'CalibrationManager', 'ArrayCalibration', 'MicrophoneCalibration',
    'SensitivityCalibrator', 'PhaseCalibrator', 'NoiseFloorAnalyzer',
    # GPU
    'GPUManager', 'GPUInfo', 'GPUBackend', 'CUDADetector',
    'PyTorchGPUDetector', 'CuPyDetector', 'get_gpu_manager',
    'is_gpu_available', 'should_use_gpu'
]
