"""
Detection and Classification Package

Provides drone detection, classification, and signature management.
"""

from .detector import (
    DroneDetector,
    DetectorConfig,
    DetectionResult,
    DetectionStatus,
    EnergyDetector,
    HarmonicDetector,
    SpectralPatternMatcher
)

from .classifier import (
    DroneClassifier,
    ClassifierConfig,
    ClassificationResult,
    DroneType,
    FeatureExtractor,
    RuleBasedClassifier,
    MLClassifier
)

from .signature_db import (
    SignatureDatabase,
    AcousticSignature,
    SignatureMatch,
    create_signature_from_audio
)

__all__ = [
    # Detector
    'DroneDetector', 'DetectorConfig', 'DetectionResult', 'DetectionStatus',
    'EnergyDetector', 'HarmonicDetector', 'SpectralPatternMatcher',
    # Classifier
    'DroneClassifier', 'ClassifierConfig', 'ClassificationResult', 'DroneType',
    'FeatureExtractor', 'RuleBasedClassifier', 'MLClassifier',
    # Signature DB
    'SignatureDatabase', 'AcousticSignature', 'SignatureMatch',
    'create_signature_from_audio'
]
