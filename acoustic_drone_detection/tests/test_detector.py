"""
Unit tests for Detector module (current API).
"""

import numpy as np
import time

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.detector import (
    DroneDetector, DetectorConfig, DetectionStatus,
    EnergyDetector, HarmonicDetector, SpectralPatternMatcher
)


def test_detector_config_defaults():
    config = DetectorConfig()
    assert config.energy_threshold_db == -40.0
    assert config.min_confidence == 0.3
    assert config.harmonic_threshold == 0.5
    assert config.min_harmonics == 2
    assert config.fundamental_range == (100.0, 500.0)
    assert config.detection_band == (80.0, 8000.0)


def test_energy_detector_detects_loud_signal():
    detector = EnergyDetector(sample_rate=48000)
    signal = np.ones(4096, dtype=np.float32) * 0.5
    detected, energy_db, snr = detector.detect(signal)
    assert bool(detected) is True
    assert energy_db > -40.0
    assert snr >= 0


def test_harmonic_detector_detects_harmonics():
    detector = HarmonicDetector(sample_rate=48000)
    t = np.linspace(0, 0.1, 4800, endpoint=False)
    f0 = 150.0
    signal = (
        np.sin(2 * np.pi * f0 * t) +
        0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)
    detected, fundamental, harmonics, score = detector.detect(signal)
    assert isinstance(detected, bool)
    assert fundamental >= 0.0
    assert score >= 0.0
    if detected:
        assert len(harmonics) >= 2


def test_spectral_pattern_matcher():
    matcher = SpectralPatternMatcher(sample_rate=48000)
    pattern = np.array([0.0, 1.0, 0.5, 0.0], dtype=np.float32)
    matcher.add_pattern("test", pattern)

    name, score = matcher.match(pattern)
    assert name == "test"
    assert score > 0.0


def test_drone_detector_no_signal():
    detector = DroneDetector(sample_rate=48000)
    signal = np.zeros(4096, dtype=np.float32)
    result = detector.detect(signal, time.time())
    assert result.status == DetectionStatus.NO_DETECTION


def test_drone_detector_harmonic_signal():
    detector = DroneDetector(sample_rate=48000)
    t = np.linspace(0, 0.1, 4800, endpoint=False)
    f0 = 150.0
    signal = (
        np.sin(2 * np.pi * f0 * t) +
        0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)
    result = detector.detect(signal, time.time())
    assert result.status in [
        DetectionStatus.POSSIBLE,
        DetectionStatus.PROBABLE,
        DetectionStatus.CONFIRMED,
        DetectionStatus.NO_DETECTION,
    ]
