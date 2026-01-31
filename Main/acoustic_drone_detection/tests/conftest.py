"""
Pytest configuration and fixtures for drone detection tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 48000


@pytest.fixture
def circular_mic_array():
    """
    8-microphone circular array with 10cm radius.

    Returns:
        np.ndarray: Microphone positions (8, 3)
    """
    num_mics = 8
    radius = 0.1  # meters
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)

    positions = np.array([
        [radius * np.cos(a), radius * np.sin(a), 0]
        for a in angles
    ])

    return positions


@pytest.fixture
def linear_mic_array():
    """
    8-microphone linear array with 5cm spacing.

    Returns:
        np.ndarray: Microphone positions (8, 3)
    """
    num_mics = 8
    spacing = 0.05  # meters

    positions = np.array([
        [i * spacing, 0, 0]
        for i in range(num_mics)
    ])

    return positions


@pytest.fixture
def test_audio_signal(sample_rate):
    """
    Generate test audio signal with known frequency content.

    Returns:
        np.ndarray: 1-second audio signal
    """
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 1kHz tone with harmonics
    signal = (
        np.sin(2 * np.pi * 1000 * t) +
        0.5 * np.sin(2 * np.pi * 2000 * t) +
        0.25 * np.sin(2 * np.pi * 3000 * t)
    )

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5

    return signal


@pytest.fixture
def multichannel_test_signal(sample_rate, circular_mic_array):
    """
    Generate multichannel test signal simulating source at specific angle.

    Returns:
        tuple: (signal, source_azimuth, source_elevation)
    """
    source_azimuth = 45.0  # degrees
    source_elevation = 0.0  # degrees
    source_frequency = 1000  # Hz
    speed_of_sound = 343.0

    duration = 0.1
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Source signal
    source_signal = np.sin(2 * np.pi * source_frequency * t)

    # Compute delays for each microphone
    source_dir = np.array([
        np.cos(np.radians(source_azimuth)) * np.cos(np.radians(source_elevation)),
        np.sin(np.radians(source_azimuth)) * np.cos(np.radians(source_elevation)),
        np.sin(np.radians(source_elevation))
    ])

    num_mics = len(circular_mic_array)
    multichannel = np.zeros((num_mics, num_samples))

    for i, pos in enumerate(circular_mic_array):
        delay = np.dot(pos, source_dir) / speed_of_sound
        delay_samples = int(delay * sample_rate)
        multichannel[i] = np.roll(source_signal, delay_samples)

    return multichannel, source_azimuth, source_elevation


@pytest.fixture
def noise_signal(sample_rate):
    """
    Generate white noise signal.

    Returns:
        np.ndarray: 1-second noise signal
    """
    duration = 1.0
    num_samples = int(sample_rate * duration)
    return np.random.randn(num_samples) * 0.1


@pytest.fixture
def drone_like_signal(sample_rate):
    """
    Generate drone-like signal with rotor harmonics.

    Returns:
        np.ndarray: Audio signal resembling drone acoustic signature
    """
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    fundamental = 150  # Hz - typical rotor frequency

    signal = np.zeros_like(t)

    # Add harmonics
    for i in range(1, 6):
        amplitude = 1.0 / i
        signal += amplitude * np.sin(2 * np.pi * fundamental * i * t)

    # Add some broadband noise
    signal += np.random.randn(len(signal)) * 0.1

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5

    return signal


@pytest.fixture
def temp_directory(tmp_path):
    """
    Create temporary directory for test files.

    Returns:
        Path: Temporary directory path
    """
    return tmp_path


@pytest.fixture
def mock_audio_capture():
    """
    Mock audio capture for testing without hardware.

    Returns:
        Mock object simulating AudioCapture
    """
    from unittest.mock import Mock

    mock = Mock()
    mock.start.return_value = True
    mock.stop.return_value = None
    mock.is_running = False

    def read_callback(samples=None, timeout=None):
        if samples is None:
            samples = 4800
        return np.random.randn(8, samples) * 0.1

    mock.read.side_effect = read_callback

    return mock


# Markers for slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests requiring audio hardware"
    )
