"""
Tests that AGC state is isolated per channel.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.signal_processor import SignalProcessor, FilterConfig


def test_agc_state_isolated_per_channel():
    sample_rate = 48000
    processor = SignalProcessor(sample_rate=sample_rate, filter_config=FilterConfig())

    # Channel 0: loud signal, Channel 1: quiet signal
    samples = 2048
    ch0 = np.ones(samples, dtype=np.float32) * 1.0
    ch1 = np.ones(samples, dtype=np.float32) * 0.01
    data = np.vstack([ch0, ch1])

    out = processor.preprocess(data, apply_agc=True, apply_noise_gate=False)

    # If AGC is per-channel, ch1 should be boosted significantly
    rms_ch0 = np.sqrt(np.mean(out[0] ** 2))
    rms_ch1 = np.sqrt(np.mean(out[1] ** 2))

    assert rms_ch0 > 0.2
    assert rms_ch1 > 0.2
