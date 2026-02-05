"""
Unit tests for Signal Processor module (current API).
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.signal_processor import (
    SignalProcessor, FilterConfig, FilterBank, GCCProcessor
)


def test_filter_config_defaults():
    config = FilterConfig()
    assert config.bandpass_low == 80.0
    assert config.bandpass_high == 8000.0
    assert config.bandpass_order == 4
    assert isinstance(config.notch_freqs, list)
    assert 50.0 in config.notch_freqs


def test_filter_bank_shapes():
    fb = FilterBank(sample_rate=48000)
    t = np.linspace(0, 0.1, 4800, endpoint=False)
    signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    bandpassed = fb.apply_bandpass(signal)
    assert bandpassed.shape == signal.shape

    notched = fb.apply_notch(signal)
    assert notched.shape == signal.shape


def test_signal_processor_preprocess_multichannel():
    sp = SignalProcessor(sample_rate=48000)
    data = np.random.randn(2, 2048).astype(np.float32) * 0.1
    out = sp.preprocess(data, apply_agc=True, apply_noise_gate=False)
    assert out.shape == data.shape


def test_gcc_processor_estimate_tdoa():
    gcc = GCCProcessor(sample_rate=48000)
    sig1 = np.random.randn(2048).astype(np.float32)
    delay = 12
    sig2 = np.roll(sig1, delay)
    delay_sec, conf = gcc.estimate_tdoa(sig1, sig2, max_delay_seconds=0.01)
    assert isinstance(delay_sec, float)
    assert conf >= 0.0
