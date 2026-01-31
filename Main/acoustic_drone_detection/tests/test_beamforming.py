"""
Unit tests for Beamforming module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.beamforming import (
    BeamformingEngine, BeamformingConfig, DOAResult,
    SteeringVectorGenerator, DelayAndSumBeamformer,
    MVDRBeamformer, MUSICEstimator
)


class TestBeamformingConfig:
    """Tests for BeamformingConfig dataclass."""

    def test_default_values(self):
        config = BeamformingConfig()
        assert config.num_azimuth == 360
        assert config.num_elevation == 91
        assert config.frequency_range == (100, 8000)
        assert config.speed_of_sound == 343.0

    def test_custom_values(self):
        config = BeamformingConfig(
            num_azimuth=180,
            num_elevation=45,
            frequency_range=(200, 4000),
            speed_of_sound=340.0
        )
        assert config.num_azimuth == 180
        assert config.num_elevation == 45


class TestDOAResult:
    """Tests for DOAResult dataclass."""

    def test_creation(self):
        result = DOAResult(
            azimuth=45.0,
            elevation=30.0,
            confidence=0.9,
            power_map=np.zeros((91, 360))
        )
        assert result.azimuth == 45.0
        assert result.elevation == 30.0
        assert result.confidence == 0.9


class TestSteeringVectorGenerator:
    """Tests for SteeringVectorGenerator class."""

    @pytest.fixture
    def mic_positions(self):
        """Create circular array positions."""
        num_mics = 8
        radius = 0.1
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        return np.array([
            [radius * np.cos(a), radius * np.sin(a), 0]
            for a in angles
        ])

    def test_init(self, mic_positions):
        gen = SteeringVectorGenerator(mic_positions, sample_rate=48000)
        assert gen._num_mics == 8
        assert gen._sample_rate == 48000

    def test_compute_steering_vector(self, mic_positions):
        gen = SteeringVectorGenerator(mic_positions, sample_rate=48000)

        # Compute steering vector for specific direction
        sv = gen.compute_steering_vector(azimuth=0, elevation=0, frequency=1000)

        assert sv.shape == (8,)
        assert np.iscomplexobj(sv)
        # Magnitude should be 1 for each element (unit vectors)
        assert np.allclose(np.abs(sv), 1.0)

    def test_compute_steering_matrix(self, mic_positions):
        gen = SteeringVectorGenerator(mic_positions, sample_rate=48000)

        # Compute steering matrix for multiple directions
        azimuths = np.array([0, 90, 180, 270])
        elevations = np.array([0, 0, 0, 0])
        frequency = 1000

        sm = gen.compute_steering_matrix(azimuths, elevations, frequency)

        assert sm.shape == (8, 4)  # num_mics x num_directions


class TestDelayAndSumBeamformer:
    """Tests for DelayAndSumBeamformer class."""

    @pytest.fixture
    def mic_positions(self):
        """Create circular array positions."""
        num_mics = 8
        radius = 0.1
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        return np.array([
            [radius * np.cos(a), radius * np.sin(a), 0]
            for a in angles
        ])

    def test_init(self, mic_positions):
        bf = DelayAndSumBeamformer(mic_positions, sample_rate=48000)
        assert bf._num_mics == 8

    def test_compute_power(self, mic_positions):
        bf = DelayAndSumBeamformer(mic_positions, sample_rate=48000)

        # Create multichannel signal
        num_samples = 1024
        signal = np.random.randn(8, num_samples) * 0.1

        # Compute power for specific direction
        power = bf.compute_power(signal, azimuth=0, elevation=0)

        assert isinstance(power, float)
        assert power >= 0

    def test_scan(self, mic_positions):
        bf = DelayAndSumBeamformer(mic_positions, sample_rate=48000)

        # Create multichannel signal
        num_samples = 1024
        signal = np.random.randn(8, num_samples) * 0.1

        # Scan across directions
        azimuths = np.arange(0, 360, 10)
        elevations = np.array([0])

        power_map = bf.scan(signal, azimuths, elevations)

        assert power_map.shape == (1, 36)  # num_elevations x num_azimuths


class TestMUSICEstimator:
    """Tests for MUSICEstimator class."""

    @pytest.fixture
    def mic_positions(self):
        """Create circular array positions."""
        num_mics = 8
        radius = 0.1
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        return np.array([
            [radius * np.cos(a), radius * np.sin(a), 0]
            for a in angles
        ])

    def test_init(self, mic_positions):
        music = MUSICEstimator(mic_positions, sample_rate=48000, num_sources=1)
        assert music._num_mics == 8
        assert music._num_sources == 1

    def test_estimate_covariance(self, mic_positions):
        music = MUSICEstimator(mic_positions, sample_rate=48000)

        # Create multichannel signal
        signal = np.random.randn(8, 1024) * 0.1

        cov = music.estimate_covariance(signal)

        assert cov.shape == (8, 8)
        # Covariance matrix should be Hermitian
        assert np.allclose(cov, cov.conj().T)

    def test_compute_spectrum(self, mic_positions):
        music = MUSICEstimator(mic_positions, sample_rate=48000)

        # Create multichannel signal
        signal = np.random.randn(8, 1024) * 0.1

        # Compute MUSIC spectrum
        azimuths = np.arange(0, 360, 10)
        elevations = np.array([0])

        spectrum = music.compute_spectrum(signal, azimuths, elevations, frequency=1000)

        assert spectrum.shape == (1, 36)


class TestBeamformingEngine:
    """Tests for main BeamformingEngine class."""

    @pytest.fixture
    def mic_positions(self):
        """Create circular array positions."""
        num_mics = 8
        radius = 0.1
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        return np.array([
            [radius * np.cos(a), radius * np.sin(a), 0]
            for a in angles
        ])

    def test_init_default(self, mic_positions):
        engine = BeamformingEngine(mic_positions, sample_rate=48000)
        assert engine._sample_rate == 48000
        assert engine._config is not None

    def test_init_with_config(self, mic_positions):
        config = BeamformingConfig(num_azimuth=180)
        engine = BeamformingEngine(mic_positions, sample_rate=48000, config=config)
        assert engine._config.num_azimuth == 180

    def test_estimate_doa(self, mic_positions):
        engine = BeamformingEngine(mic_positions, sample_rate=48000)

        # Create multichannel signal
        signal = np.random.randn(8, 2048) * 0.1

        result = engine.estimate_doa(signal)

        assert isinstance(result, DOAResult)
        assert -180 <= result.azimuth <= 180
        assert -90 <= result.elevation <= 90
        assert 0 <= result.confidence <= 1

    def test_estimate_doa_with_source(self, mic_positions):
        """Test DOA estimation with synthetic source."""
        engine = BeamformingEngine(mic_positions, sample_rate=48000)

        # Create signal from specific direction
        # Source at 45 degrees azimuth, 0 elevation
        source_az = 45
        source_el = 0
        frequency = 1000

        # Compute delays for this direction
        t = np.linspace(0, 0.05, 2400)
        source_signal = np.sin(2 * np.pi * frequency * t)

        # Apply delays to each microphone
        speed_of_sound = 343.0
        signal = np.zeros((8, len(t)))

        for i, pos in enumerate(mic_positions):
            # Compute path length difference
            source_dir = np.array([
                np.cos(np.radians(source_az)) * np.cos(np.radians(source_el)),
                np.sin(np.radians(source_az)) * np.cos(np.radians(source_el)),
                np.sin(np.radians(source_el))
            ])
            delay = np.dot(pos, source_dir) / speed_of_sound
            delay_samples = int(delay * 48000)

            signal[i] = np.roll(source_signal, delay_samples)

        result = engine.estimate_doa(signal)

        # Should estimate direction close to source
        assert abs(result.azimuth - source_az) < 20 or abs(result.azimuth - source_az + 360) < 20

    def test_beamform_output(self, mic_positions):
        engine = BeamformingEngine(mic_positions, sample_rate=48000)

        # Create multichannel signal
        signal = np.random.randn(8, 1024) * 0.1

        # Beamform in specific direction
        output = engine.beamform(signal, azimuth=0, elevation=0)

        assert output.shape == (1024,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
