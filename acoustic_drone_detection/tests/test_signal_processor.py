"""
Unit tests for Signal Processor module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.signal_processor import (
    SignalProcessor, FilterConfig, FilterBank, NoiseGate,
    AutomaticGainControl, SpectralAnalyzer, GCCProcessor
)


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self):
        config = FilterConfig()
        assert config.bandpass_low == 80
        assert config.bandpass_high == 8000
        assert config.notch_frequencies == []
        assert config.filter_order == 4

    def test_custom_values(self):
        config = FilterConfig(
            bandpass_low=100,
            bandpass_high=4000,
            notch_frequencies=[50, 100],
            filter_order=6
        )
        assert config.bandpass_low == 100
        assert config.bandpass_high == 4000
        assert config.notch_frequencies == [50, 100]
        assert config.filter_order == 6


class TestFilterBank:
    """Tests for FilterBank class."""

    def test_init(self):
        fb = FilterBank(sample_rate=48000, filter_order=4)
        assert fb._sample_rate == 48000
        assert fb._filter_order == 4
        assert fb._filters == {}

    def test_design_bandpass(self):
        fb = FilterBank(sample_rate=48000)
        sos = fb.design_bandpass(80, 8000)
        assert sos is not None
        assert sos.shape[1] == 6  # SOS format has 6 coefficients per section

    def test_design_highpass(self):
        fb = FilterBank(sample_rate=48000)
        sos = fb.design_highpass(80)
        assert sos is not None
        assert sos.shape[1] == 6

    def test_design_lowpass(self):
        fb = FilterBank(sample_rate=48000)
        sos = fb.design_lowpass(8000)
        assert sos is not None
        assert sos.shape[1] == 6

    def test_design_notch(self):
        fb = FilterBank(sample_rate=48000)
        sos = fb.design_notch(50)
        assert sos is not None

    def test_apply_filter(self):
        fb = FilterBank(sample_rate=48000)

        # Create test signal
        t = np.linspace(0, 1, 48000)
        signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave

        # Design and apply bandpass filter
        sos = fb.design_bandpass(500, 2000)
        filtered = fb.apply_filter(signal, sos)

        assert filtered.shape == signal.shape
        # Signal should pass through (within passband)
        assert np.max(np.abs(filtered)) > 0.5


class TestNoiseGate:
    """Tests for NoiseGate class."""

    def test_init(self):
        gate = NoiseGate(threshold=-40.0, attack=0.001, release=0.1)
        assert gate._threshold == -40.0
        assert gate._attack == 0.001
        assert gate._release == 0.1

    def test_process_silent(self):
        gate = NoiseGate(threshold=-40.0)

        # Create silent signal
        signal = np.zeros(1000)
        processed = gate.process(signal)

        # Should remain silent
        assert np.allclose(processed, 0)

    def test_process_loud(self):
        gate = NoiseGate(threshold=-60.0)

        # Create loud signal
        signal = np.ones(1000) * 0.5
        processed = gate.process(signal)

        # Should pass through
        assert np.max(np.abs(processed)) > 0


class TestAutomaticGainControl:
    """Tests for AutomaticGainControl class."""

    def test_init(self):
        agc = AutomaticGainControl(target_level=-20.0)
        assert agc._target_level == -20.0
        assert agc._current_gain == 1.0

    def test_process(self):
        agc = AutomaticGainControl(target_level=-20.0)

        # Create test signal
        signal = np.random.randn(1000) * 0.1
        processed = agc.process(signal)

        assert processed.shape == signal.shape

    def test_reset(self):
        agc = AutomaticGainControl()
        agc._current_gain = 2.0
        agc.reset()
        assert agc._current_gain == 1.0


class TestSpectralAnalyzer:
    """Tests for SpectralAnalyzer class."""

    def test_init(self):
        sa = SpectralAnalyzer(sample_rate=48000, fft_size=2048)
        assert sa._sample_rate == 48000
        assert sa._fft_size == 2048

    def test_compute_spectrum(self):
        sa = SpectralAnalyzer(sample_rate=48000, fft_size=1024)

        # Create test signal
        signal = np.random.randn(1024)
        freqs, magnitude = sa.compute_spectrum(signal)

        assert len(freqs) == 513  # FFT_size/2 + 1
        assert len(magnitude) == 513
        assert freqs[0] == 0
        assert freqs[-1] == 24000  # Nyquist

    def test_compute_stft(self):
        sa = SpectralAnalyzer(sample_rate=48000, fft_size=1024)

        # Create longer signal
        signal = np.random.randn(4800)
        times, freqs, stft = sa.compute_stft(signal)

        assert len(times) > 0
        assert len(freqs) == 513
        assert stft.shape[0] == 513

    def test_find_peaks(self):
        sa = SpectralAnalyzer(sample_rate=48000, fft_size=2048)

        # Create signal with known frequency
        t = np.linspace(0, 1, 48000)
        signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz

        freqs, magnitude = sa.compute_spectrum(signal[:2048])
        peaks = sa.find_peaks(freqs, magnitude, num_peaks=5)

        # Should find peak near 1000Hz
        assert len(peaks) > 0
        assert any(abs(p[0] - 1000) < 50 for p in peaks)


class TestGCCProcessor:
    """Tests for GCCProcessor class."""

    def test_init(self):
        gcc = GCCProcessor(sample_rate=48000, max_delay_samples=100)
        assert gcc._sample_rate == 48000
        assert gcc._max_delay == 100

    def test_gcc_phat(self):
        gcc = GCCProcessor(sample_rate=48000, max_delay_samples=50)

        # Create two identical signals with delay
        sig1 = np.random.randn(1024)
        delay = 10
        sig2 = np.roll(sig1, delay)

        # Compute GCC-PHAT
        delays, correlation = gcc.gcc_phat(sig1, sig2)

        # Peak should be near the actual delay
        peak_idx = np.argmax(correlation)
        estimated_delay = delays[peak_idx]

        # Allow some tolerance
        assert abs(estimated_delay - delay) < 5

    def test_find_delay(self):
        gcc = GCCProcessor(sample_rate=48000, max_delay_samples=50)

        # Create signals with known delay
        sig1 = np.random.randn(1024)
        delay = 15
        sig2 = np.roll(sig1, delay)

        estimated = gcc.find_delay(sig1, sig2)

        # Should be close to actual delay
        assert abs(estimated - delay) < 5


class TestSignalProcessor:
    """Tests for main SignalProcessor class."""

    def test_init_default(self):
        sp = SignalProcessor(sample_rate=48000)
        assert sp._sample_rate == 48000
        assert sp._filter_config is not None

    def test_init_with_config(self):
        config = FilterConfig(bandpass_low=100, bandpass_high=4000)
        sp = SignalProcessor(sample_rate=48000, filter_config=config)
        assert sp._filter_config == config

    def test_preprocess_mono(self):
        sp = SignalProcessor(sample_rate=48000)

        # Create test signal
        signal = np.random.randn(4800) * 0.1
        processed = sp.preprocess(signal)

        assert processed.shape == signal.shape

    def test_preprocess_multichannel(self):
        sp = SignalProcessor(sample_rate=48000)

        # Create multichannel signal (8 channels, 4800 samples)
        signal = np.random.randn(8, 4800) * 0.1
        processed = sp.preprocess(signal)

        assert processed.shape == signal.shape

    def test_compute_rms(self):
        sp = SignalProcessor(sample_rate=48000)

        # Create known signal
        signal = np.ones(1000) * 0.5
        rms = sp.compute_rms(signal)

        assert abs(rms - 0.5) < 0.01

    def test_compute_snr(self):
        sp = SignalProcessor(sample_rate=48000)

        # Create signal with known SNR
        signal_power = 1.0
        noise_power = 0.01
        signal = np.sqrt(signal_power) * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.1, 4800))
        noise = np.sqrt(noise_power) * np.random.randn(4800)

        snr = sp.compute_snr(signal, noise)

        # Expected SNR is about 20 dB
        assert 15 < snr < 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
