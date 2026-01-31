"""
Unit tests for Detector module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.detector import (
    DroneDetector, DetectorConfig, DetectionResult, DetectionStatus,
    EnergyDetector, HarmonicDetector, SpectralPatternMatcher
)


class TestDetectorConfig:
    """Tests for DetectorConfig dataclass."""

    def test_default_values(self):
        config = DetectorConfig()
        assert config.min_confidence == 0.5
        assert config.detection_band == (80, 8000)
        assert config.num_harmonics == 5
        assert config.energy_threshold == -50.0

    def test_custom_values(self):
        config = DetectorConfig(
            min_confidence=0.7,
            detection_band=(100, 4000),
            num_harmonics=3,
            energy_threshold=-40.0
        )
        assert config.min_confidence == 0.7
        assert config.detection_band == (100, 4000)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation(self):
        result = DetectionResult(
            status=DetectionStatus.DETECTED,
            confidence=0.85,
            snr=15.0,
            dominant_frequencies=np.array([1000.0, 2000.0]),
            spectral_features={'energy': 0.5}
        )
        assert result.status == DetectionStatus.DETECTED
        assert result.confidence == 0.85
        assert result.snr == 15.0
        assert len(result.dominant_frequencies) == 2


class TestEnergyDetector:
    """Tests for EnergyDetector class."""

    def test_init(self):
        detector = EnergyDetector(sample_rate=48000, threshold=-40.0)
        assert detector._sample_rate == 48000
        assert detector._threshold == -40.0

    def test_compute_energy_db(self):
        detector = EnergyDetector(sample_rate=48000)

        # Create signal with known RMS
        signal = np.ones(1000) * 0.1
        energy_db = detector.compute_energy_db(signal)

        # RMS of 0.1 should be about -20 dB
        assert -25 < energy_db < -15

    def test_detect_above_threshold(self):
        detector = EnergyDetector(sample_rate=48000, threshold=-30.0)

        # Create loud signal
        signal = np.ones(1000) * 0.5
        detected, energy = detector.detect(signal)

        assert detected is True
        assert energy > -30.0

    def test_detect_below_threshold(self):
        detector = EnergyDetector(sample_rate=48000, threshold=-20.0)

        # Create quiet signal
        signal = np.ones(1000) * 0.01
        detected, energy = detector.detect(signal)

        assert detected is False


class TestHarmonicDetector:
    """Tests for HarmonicDetector class."""

    def test_init(self):
        detector = HarmonicDetector(
            sample_rate=48000,
            num_harmonics=5,
            min_fundamental=80,
            max_fundamental=400
        )
        assert detector._sample_rate == 48000
        assert detector._num_harmonics == 5

    def test_detect_harmonics_synthetic(self):
        detector = HarmonicDetector(sample_rate=48000, num_harmonics=3)

        # Create signal with harmonics at 100, 200, 300 Hz
        t = np.linspace(0, 1, 48000)
        signal = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.25 * np.sin(2 * np.pi * 300 * t)
        )

        result = detector.detect(signal[:4096])

        assert result['detected'] is True
        # Fundamental should be near 100 Hz
        assert abs(result['fundamental'] - 100) < 20

    def test_detect_no_harmonics(self):
        detector = HarmonicDetector(sample_rate=48000)

        # Create random noise
        signal = np.random.randn(4096) * 0.01

        result = detector.detect(signal)

        # Should not detect clear harmonics in noise
        assert result['confidence'] < 0.5


class TestSpectralPatternMatcher:
    """Tests for SpectralPatternMatcher class."""

    def test_init(self):
        matcher = SpectralPatternMatcher(sample_rate=48000)
        assert matcher._sample_rate == 48000

    def test_add_pattern(self):
        matcher = SpectralPatternMatcher(sample_rate=48000)

        pattern = {
            'name': 'test_drone',
            'frequencies': [100, 200, 300],
            'weights': [1.0, 0.5, 0.25]
        }

        matcher.add_pattern(pattern)
        assert 'test_drone' in matcher._patterns

    def test_match_pattern(self):
        matcher = SpectralPatternMatcher(sample_rate=48000)

        # Add a pattern
        pattern = {
            'name': 'test_drone',
            'frequencies': [100, 200, 300],
            'weights': [1.0, 0.5, 0.25]
        }
        matcher.add_pattern(pattern)

        # Create signal matching the pattern
        t = np.linspace(0, 1, 48000)
        signal = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.25 * np.sin(2 * np.pi * 300 * t)
        )

        matches = matcher.match(signal[:4096])

        assert len(matches) > 0
        assert matches[0]['pattern'] == 'test_drone'


class TestDroneDetector:
    """Tests for main DroneDetector class."""

    def test_init_default(self):
        detector = DroneDetector(sample_rate=48000)
        assert detector._sample_rate == 48000
        assert detector._config is not None

    def test_init_with_config(self):
        config = DetectorConfig(min_confidence=0.7)
        detector = DroneDetector(sample_rate=48000, config=config)
        assert detector._config.min_confidence == 0.7

    def test_detect_no_signal(self):
        detector = DroneDetector(sample_rate=48000)

        # Silent signal
        signal = np.zeros(4096)
        result = detector.detect(signal, time.time())

        assert result.status == DetectionStatus.NO_DETECTION
        assert result.confidence < 0.5

    def test_detect_noise(self):
        detector = DroneDetector(sample_rate=48000)

        # Random noise
        signal = np.random.randn(4096) * 0.01
        result = detector.detect(signal, time.time())

        # Should not confidently detect drone in noise
        assert result.confidence < 0.7

    def test_detect_drone_like_signal(self):
        detector = DroneDetector(sample_rate=48000)

        # Create drone-like signal with rotor harmonics
        t = np.linspace(0, 0.1, 4800)
        fundamental = 150  # Typical rotor frequency

        signal = np.zeros_like(t)
        for i in range(1, 6):  # 5 harmonics
            signal += (1.0 / i) * np.sin(2 * np.pi * fundamental * i * t)

        # Add some noise
        signal += np.random.randn(len(signal)) * 0.1
        signal = signal / np.max(np.abs(signal)) * 0.5

        result = detector.detect(signal, time.time())

        # Should detect drone-like pattern
        assert result.status in [DetectionStatus.DETECTED, DetectionStatus.POSSIBLE]
        assert len(result.dominant_frequencies) > 0

    def test_reset(self):
        detector = DroneDetector(sample_rate=48000)

        # Process some signals
        signal = np.random.randn(4096) * 0.1
        detector.detect(signal, time.time())
        detector.detect(signal, time.time())

        # Reset
        detector.reset()

        # Internal state should be cleared
        assert detector._detection_history == []


class TestDetectionStatus:
    """Tests for DetectionStatus enum."""

    def test_values(self):
        assert DetectionStatus.NO_DETECTION.value == "no_detection"
        assert DetectionStatus.POSSIBLE.value == "possible"
        assert DetectionStatus.DETECTED.value == "detected"
        assert DetectionStatus.CONFIRMED.value == "confirmed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
