"""
Core Processing Engine Package

Provides the main signal processing, localization, and tracking components.
"""

from .audio_capture import (
    AudioCapture,
    RingBuffer,
    DropoutDetector,
    AudioDeviceInfo,
    ChannelStatus
)

from .signal_processor import (
    SignalProcessor,
    FilterBank,
    FilterConfig,
    NoiseGate,
    AGC,
    SpectralAnalyzer,
    SpectralConfig,
    SpectralFeatures,
    GCCProcessor
)

from .tdoa_engine import (
    TDOAEngine,
    TDOAResult,
    TDOAConfig,
    InterpolationMethods
)

from .beamforming import (
    BeamformingEngine,
    BeamformingConfig,
    DOAResult,
    SteeringVectorGenerator,
    CovarianceEstimator,
    DelayAndSumBeamformer,
    MVDRBeamformer,
    MUSICBeamformer,
    ESPRITBeamformer,
    SRPPHATBeamformer
)

from .localization import (
    LocalizationEngine,
    LocalizationConfig,
    LocalizationResult,
    Position3D,
    MultilaterationSolver,
    DistanceEstimator,
    HybridLocalizer
)

from .tracking import (
    MultiTargetTracker,
    TrackingConfig,
    Track,
    TrackState,
    Detection,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    DataAssociation
)

__all__ = [
    # Audio Capture
    'AudioCapture', 'RingBuffer', 'DropoutDetector', 'AudioDeviceInfo', 'ChannelStatus',
    # Signal Processing
    'SignalProcessor', 'FilterBank', 'FilterConfig', 'NoiseGate', 'AGC',
    'SpectralAnalyzer', 'SpectralConfig', 'SpectralFeatures', 'GCCProcessor',
    # TDOA
    'TDOAEngine', 'TDOAResult', 'TDOAConfig', 'InterpolationMethods',
    # Beamforming
    'BeamformingEngine', 'BeamformingConfig', 'DOAResult', 'SteeringVectorGenerator',
    'CovarianceEstimator', 'DelayAndSumBeamformer', 'MVDRBeamformer',
    'MUSICBeamformer', 'ESPRITBeamformer', 'SRPPHATBeamformer',
    # Localization
    'LocalizationEngine', 'LocalizationConfig', 'LocalizationResult', 'Position3D',
    'MultilaterationSolver', 'DistanceEstimator', 'HybridLocalizer',
    # Tracking
    'MultiTargetTracker', 'TrackingConfig', 'Track', 'TrackState', 'Detection',
    'ExtendedKalmanFilter', 'UnscentedKalmanFilter', 'DataAssociation'
]
