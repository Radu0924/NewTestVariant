"""
Unit tests for Beamforming module (current API).
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.beamforming import (
    BeamformingEngine, BeamformingConfig, DOAResult, SteeringVectorGenerator
)


def test_beamforming_config_defaults():
    config = BeamformingConfig()
    assert config.num_azimuth == 360
    assert config.num_elevation == 91
    assert config.frequency_range == (100.0, 8000.0)


def test_steering_vector_generator():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, -0.1, 0.0],
        [0.07, 0.07, 0.0],
        [-0.07, 0.07, 0.0],
        [0.07, -0.07, 0.0],
    ])
    gen = SteeringVectorGenerator(positions, sample_rate=48000)
    sv = gen.compute_steering_vector(azimuth=0.0, elevation=0.0, frequency=1000.0)
    assert sv.shape == (positions.shape[0],)
    assert np.iscomplexobj(sv)


def test_beamforming_engine_estimate_doa():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, -0.1, 0.0],
        [0.07, 0.07, 0.0],
        [-0.07, 0.07, 0.0],
        [0.07, -0.07, 0.0],
    ])
    engine = BeamformingEngine(positions, sample_rate=48000)
    signal = np.random.randn(8, 2048).astype(np.float32) * 0.1
    result = engine.estimate_doa(signal)
    assert isinstance(result, DOAResult)
    assert 0.0 <= result.azimuth <= 360.0
    assert -90.0 <= result.elevation <= 90.0
