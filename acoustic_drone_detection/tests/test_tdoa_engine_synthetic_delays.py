"""
Tests for TDOAEngine with synthetic delays.
Verifies that the engine correctly estimates known time delays.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tdoa_engine import TDOAEngine, TDOAConfig


def test_tdoa_engine_zero_delay():
    """Test TDOA estimation with zero delay (identical signals)."""
    sample_rate = 48000
    config = TDOAConfig()
    engine = TDOAEngine(sample_rate=sample_rate, config=config)

    # Create identical signals
    samples = 1024
    t = np.linspace(0, samples / sample_rate, samples, dtype=np.float32)
    signal = np.sin(2 * np.pi * 1000 * t)

    result = engine.estimate_tdoa(signal, signal, channel_i=0, channel_j=1)

    assert abs(result.delay_seconds) < 1e-4, \
        f"Expected near-zero delay, got {result.delay_seconds}"
    assert result.confidence > 0.5


def test_tdoa_engine_known_delay():
    """Test TDOA estimation with a known delay."""
    sample_rate = 48000
    config = TDOAConfig()
    engine = TDOAEngine(sample_rate=sample_rate, config=config)

    # Create signal with known delay
    samples = 2048
    delay_samples = 10
    expected_delay = delay_samples / sample_rate

    t = np.linspace(0, samples / sample_rate, samples, dtype=np.float32)
    signal1 = np.sin(2 * np.pi * 1000 * t)

    # Create delayed version
    signal2 = np.zeros_like(signal1)
    signal2[delay_samples:] = signal1[:-delay_samples]

    result = engine.estimate_tdoa(signal1, signal2, channel_i=0, channel_j=1)

    assert abs(result.delay_seconds - expected_delay) < 3e-4, \
        f"Expected delay ~{expected_delay:.6f}, got {result.delay_seconds:.6f}"


def test_tdoa_engine_negative_delay():
    """Test TDOA estimation with negative delay (swapped signal order)."""
    sample_rate = 48000
    config = TDOAConfig()
    engine = TDOAEngine(sample_rate=sample_rate, config=config)

    samples = 2048
    delay_samples = 8
    # When signal2 is delayed relative to signal1, estimate_tdoa(s1,s2) returns positive
    # To get negative delay, we swap signal order in the call
    expected_delay = -delay_samples / sample_rate

    t = np.linspace(0, samples / sample_rate, samples, dtype=np.float32)
    signal1 = np.sin(2 * np.pi * 800 * t)

    # Create delayed signal2 (signal2 arrives LATER than signal1)
    signal2 = np.zeros_like(signal1)
    signal2[delay_samples:] = signal1[:-delay_samples]

    # Swap order: estimate delay of signal1 relative to signal2
    # This gives negative delay since signal1 arrives EARLIER
    result = engine.estimate_tdoa(signal2, signal1, channel_i=0, channel_j=1)

    # Delay should be negative since signal1 arrives earlier than signal2
    assert abs(result.delay_seconds - expected_delay) < 3e-4, \
        f"Expected delay ~{expected_delay:.6f}, got {result.delay_seconds:.6f}"


def test_tdoa_engine_broadband_signal():
    """Test TDOA with broadband noise signal."""
    sample_rate = 48000
    config = TDOAConfig()
    engine = TDOAEngine(sample_rate=sample_rate, config=config)

    samples = 4096
    delay_samples = 5
    expected_delay = delay_samples / sample_rate

    np.random.seed(42)
    signal1 = np.random.randn(samples).astype(np.float32)

    signal2 = np.zeros_like(signal1)
    signal2[delay_samples:] = signal1[:-delay_samples]

    result = engine.estimate_tdoa(signal1, signal2, channel_i=0, channel_j=1)

    assert abs(result.delay_seconds - expected_delay) < 3e-4
    assert result.confidence > 0.3
