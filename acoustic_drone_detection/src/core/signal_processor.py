"""
Signal Processing Module

Provides comprehensive signal processing capabilities:
- Bandpass and notch filtering
- FFT and STFT analysis
- MFCC feature extraction
- Adaptive noise reduction
- GCC-PHAT cross-correlation

Supports high-performance C++ backend when available.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, rfft, rfftfreq
from scipy.signal import butter, sosfilt, iirnotch, stft, find_peaks
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
import logging

# Try to import C++ backend for performance-critical operations
try:
    from .cpp_backend import (
        SignalProcessor as CppSignalProcessor,
        is_cpp_available,
        is_cuda_available
    )
    _CPP_BACKEND_AVAILABLE = is_cpp_available()
except ImportError:
    _CPP_BACKEND_AVAILABLE = False
    def is_cuda_available():
        return False

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Filter configuration parameters."""
    bandpass_low: float = 80.0
    bandpass_high: float = 8000.0
    bandpass_order: int = 4
    notch_freqs: List[float] = None  # 50Hz, 60Hz, etc.
    notch_q: float = 30.0

    def __post_init__(self):
        if self.notch_freqs is None:
            self.notch_freqs = [50.0, 60.0, 100.0, 120.0]


@dataclass
class SpectralConfig:
    """Spectral analysis configuration."""
    fft_size: int = 2048
    hop_size: int = 512
    window_type: str = "hann"  # hann, hamming, blackman, kaiser
    window_param: float = 0.0  # For kaiser window


@dataclass
class SpectralFeatures:
    """Container for spectral features."""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    power_spectrum: np.ndarray
    dominant_frequencies: np.ndarray
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float


class FilterBank:
    """
    Multi-stage filter bank for audio preprocessing.

    Provides bandpass filtering, notch filters for interference removal,
    and adaptive noise gating.
    """

    def __init__(self, sample_rate: int, config: Optional[FilterConfig] = None):
        """
        Initialize the filter bank.

        Args:
            sample_rate: Sample rate in Hz.
            config: Filter configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or FilterConfig()
        self._lock = threading.Lock()

        self._init_filters()

    def _init_filters(self) -> None:
        """Initialize all filter coefficients."""
        nyquist = self._sample_rate / 2

        # Bandpass filter
        low = self._config.bandpass_low / nyquist
        high = min(self._config.bandpass_high / nyquist, 0.99)
        self._bandpass_sos = butter(
            self._config.bandpass_order,
            [low, high],
            btype='band',
            output='sos'
        )

        # Notch filters for powerline interference
        self._notch_filters = []
        for freq in self._config.notch_freqs:
            if freq < nyquist:
                b, a = iirnotch(freq, self._config.notch_q, self._sample_rate)
                self._notch_filters.append((b, a))

        # High-pass filter for wind noise
        self._highpass_sos = butter(2, 100.0 / nyquist, btype='high', output='sos')

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter.

        Args:
            data: Input audio data.

        Returns:
            Filtered audio data.
        """
        return sosfilt(self._bandpass_sos, data, axis=-1)

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filters for interference removal.

        Args:
            data: Input audio data.

        Returns:
            Filtered audio data.
        """
        result = data.copy()
        for b, a in self._notch_filters:
            result = signal.filtfilt(b, a, result, axis=-1)
        return result

    def apply_highpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter for wind noise reduction.

        Args:
            data: Input audio data.

        Returns:
            Filtered audio data.
        """
        return sosfilt(self._highpass_sos, data, axis=-1)

    def apply_all(self, data: np.ndarray, apply_notch: bool = True) -> np.ndarray:
        """
        Apply all filters in sequence.

        Args:
            data: Input audio data.
            apply_notch: Whether to apply notch filters.

        Returns:
            Fully filtered audio data.
        """
        result = self.apply_bandpass(data)
        if apply_notch:
            result = self.apply_notch(result)
        return result

    def update_config(self, config: FilterConfig) -> None:
        """Update filter configuration."""
        with self._lock:
            self._config = config
            self._init_filters()


class NoiseGate:
    """
    Adaptive noise gate.

    Reduces low-level noise while preserving signal content.
    """

    def __init__(
        self,
        threshold_db: float = -40.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        sample_rate: int = 48000
    ):
        """
        Initialize noise gate.

        Args:
            threshold_db: Gate threshold in dB.
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.
            sample_rate: Sample rate in Hz.
        """
        self._threshold = 10 ** (threshold_db / 20)
        self._attack_coef = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self._release_coef = np.exp(-1.0 / (release_ms * sample_rate / 1000))

        self._envelope = 0.0
        self._gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process audio through noise gate.

        Args:
            data: Input audio data.

        Returns:
            Gated audio data.
        """
        result = np.zeros_like(data)

        for i in range(len(data)):
            # Envelope follower
            abs_sample = abs(data[i])
            if abs_sample > self._envelope:
                self._envelope = self._attack_coef * self._envelope + \
                                 (1 - self._attack_coef) * abs_sample
            else:
                self._envelope = self._release_coef * self._envelope + \
                                 (1 - self._release_coef) * abs_sample

            # Gate
            if self._envelope > self._threshold:
                self._gain = 1.0
            else:
                self._gain = self._envelope / (self._threshold + 1e-12)

            result[i] = data[i] * self._gain

        return result

    def reset(self) -> None:
        """Reset gate state."""
        self._envelope = 0.0
        self._gain = 1.0


class AGC:
    """
    Automatic Gain Control.

    Normalizes signal levels across channels.
    """

    def __init__(
        self,
        target_level: float = 0.5,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        sample_rate: int = 48000,
        max_gain_db: float = 40.0
    ):
        """
        Initialize AGC.

        Args:
            target_level: Target RMS level (0-1).
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.
            sample_rate: Sample rate in Hz.
            max_gain_db: Maximum gain in dB.
        """
        self._target = target_level
        self._attack_coef = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self._release_coef = np.exp(-1.0 / (release_ms * sample_rate / 1000))
        self._max_gain = 10 ** (max_gain_db / 20)

        self._gain = 1.0
        self._envelope = 0.0

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply automatic gain control.

        Args:
            data: Input audio data.

        Returns:
            Gain-controlled audio data.
        """
        # Estimate signal level
        rms = np.sqrt(np.mean(data ** 2))

        # Update envelope
        if rms > self._envelope:
            self._envelope = self._attack_coef * self._envelope + \
                             (1 - self._attack_coef) * rms
        else:
            self._envelope = self._release_coef * self._envelope + \
                             (1 - self._release_coef) * rms

        # Calculate gain
        if self._envelope > 1e-6:
            desired_gain = self._target / self._envelope
            self._gain = min(desired_gain, self._max_gain)
        else:
            self._gain = 1.0

        return data * self._gain

    def reset(self) -> None:
        """Reset AGC state."""
        self._gain = 1.0
        self._envelope = 0.0


class SpectralAnalyzer:
    """
    Spectral analysis engine.

    Provides FFT, STFT, and feature extraction for audio signals.
    """

    WINDOW_FUNCTIONS = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'kaiser': lambda n: np.kaiser(n, 5.0),
        'bartlett': np.bartlett
    }

    def __init__(
        self,
        sample_rate: int = 48000,
        config: Optional[SpectralConfig] = None
    ):
        """
        Initialize spectral analyzer.

        Args:
            sample_rate: Sample rate in Hz.
            config: Spectral analysis configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or SpectralConfig()
        self._init_window()

    def _init_window(self) -> None:
        """Initialize window function."""
        window_func = self.WINDOW_FUNCTIONS.get(
            self._config.window_type,
            np.hanning
        )
        self._window = window_func(self._config.fft_size)

    def analyze(self, data: np.ndarray) -> SpectralFeatures:
        """
        Perform spectral analysis.

        Args:
            data: Input audio data (1D).

        Returns:
            SpectralFeatures object.
        """
        # Zero-pad if necessary
        if len(data) < self._config.fft_size:
            data = np.pad(data, (0, self._config.fft_size - len(data)))

        # Apply window
        windowed = data[:self._config.fft_size] * self._window

        # Compute FFT
        spectrum = rfft(windowed)
        frequencies = rfftfreq(self._config.fft_size, 1 / self._sample_rate)

        # Magnitude and phase
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        # Power spectrum
        power_spectrum = magnitudes ** 2

        # Find dominant frequencies
        peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)
        dominant_frequencies = frequencies[peaks][:10] if len(peaks) > 0 else np.array([])

        # Spectral features
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            spectral_centroid = np.sum(frequencies * power_spectrum) / total_power
            spectral_bandwidth = np.sqrt(
                np.sum(((frequencies - spectral_centroid) ** 2) * power_spectrum) / total_power
            )
        else:
            spectral_centroid = 0.0
            spectral_bandwidth = 0.0

        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(power_spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.95 * total_power)
        spectral_rolloff = frequencies[min(rolloff_idx, len(frequencies) - 1)]

        return SpectralFeatures(
            frequencies=frequencies,
            magnitudes=magnitudes,
            phases=phases,
            power_spectrum=power_spectrum,
            dominant_frequencies=dominant_frequencies,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff
        )

    def compute_spectrogram(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using STFT.

        Args:
            data: Input audio data.

        Returns:
            Tuple of (frequencies, times, spectrogram).
        """
        f, t, Zxx = stft(
            data,
            fs=self._sample_rate,
            window=self._config.window_type,
            nperseg=self._config.fft_size,
            noverlap=self._config.fft_size - self._config.hop_size
        )
        return f, t, np.abs(Zxx)

    def compute_mfcc(
        self,
        data: np.ndarray,
        num_mfcc: int = 13,
        num_filters: int = 40
    ) -> np.ndarray:
        """
        Compute Mel-frequency cepstral coefficients.

        Args:
            data: Input audio data.
            num_mfcc: Number of MFCCs to compute.
            num_filters: Number of Mel filters.

        Returns:
            MFCC feature array.
        """
        # Compute power spectrum
        spectrum = np.abs(rfft(data * self._window[:len(data)])) ** 2

        # Create Mel filterbank
        mel_filters = self._create_mel_filterbank(len(spectrum), num_filters)

        # Apply filterbank
        mel_spectrum = np.dot(mel_filters, spectrum)

        # Log compression
        mel_spectrum = np.log(mel_spectrum + 1e-12)

        # DCT to get MFCCs
        mfcc = self._dct(mel_spectrum, num_mfcc)

        return mfcc

    def _create_mel_filterbank(
        self,
        num_bins: int,
        num_filters: int
    ) -> np.ndarray:
        """Create Mel filterbank matrix."""
        mel_low = self._hz_to_mel(0)
        mel_high = self._hz_to_mel(self._sample_rate / 2)

        mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor(
            (num_bins * 2) * hz_points / self._sample_rate
        ).astype(int)

        filters = np.zeros((num_filters, num_bins))

        for i in range(num_filters):
            for j in range(int(bin_points[i]), int(bin_points[i + 1])):
                filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(int(bin_points[i + 1]), int(bin_points[i + 2])):
                filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

        return filters

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert Hz to Mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        """Convert Mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    @staticmethod
    def _dct(x: np.ndarray, num_coeffs: int) -> np.ndarray:
        """Compute DCT-II."""
        n = len(x)
        result = np.zeros(num_coeffs)
        for k in range(num_coeffs):
            result[k] = np.sum(
                x * np.cos(np.pi * k * (2 * np.arange(n) + 1) / (2 * n))
            )
        return result * np.sqrt(2 / n)


class GCCProcessor:
    """
    Generalized Cross-Correlation Processor.

    Implements GCC-PHAT for time delay estimation between microphone pairs.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 2048):
        """
        Initialize GCC processor.

        Args:
            sample_rate: Sample rate in Hz.
            fft_size: FFT size for correlation.
        """
        self._sample_rate = sample_rate
        self._fft_size = fft_size

    def gcc_phat(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        max_delay_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute GCC-PHAT between two signals.

        Args:
            signal1: First signal.
            signal2: Second signal.
            max_delay_samples: Maximum delay to consider.

        Returns:
            Tuple of (correlation, delay_samples, peak_value).
        """
        # Pad signals to FFT size
        n = max(len(signal1), len(signal2), self._fft_size)
        n = int(2 ** np.ceil(np.log2(n)))

        sig1_padded = np.zeros(n)
        sig2_padded = np.zeros(n)
        sig1_padded[:len(signal1)] = signal1
        sig2_padded[:len(signal2)] = signal2

        # Compute FFTs
        fft1 = fft(sig1_padded)
        fft2 = fft(sig2_padded)

        # Cross-spectrum with PHAT weighting
        cross_spectrum = fft1 * np.conj(fft2)
        magnitude = np.abs(cross_spectrum)
        cross_spectrum = cross_spectrum / (magnitude + 1e-12)

        # Inverse FFT to get correlation
        correlation = np.real(ifft(cross_spectrum))

        # Shift for proper delay representation
        correlation = np.fft.fftshift(correlation)

        # Find peak
        if max_delay_samples:
            center = len(correlation) // 2
            search_range = slice(
                center - max_delay_samples,
                center + max_delay_samples + 1
            )
            peak_idx = np.argmax(np.abs(correlation[search_range]))
            peak_idx = peak_idx + center - max_delay_samples
        else:
            peak_idx = np.argmax(np.abs(correlation))

        delay_samples = peak_idx - len(correlation) // 2
        peak_value = correlation[peak_idx]

        return correlation, delay_samples, peak_value

    def estimate_tdoa(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        max_delay_seconds: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate Time Difference of Arrival.

        Args:
            signal1: First signal.
            signal2: Second signal.
            max_delay_seconds: Maximum delay to consider.

        Returns:
            Tuple of (delay_seconds, confidence).
        """
        max_delay_samples = None
        if max_delay_seconds:
            max_delay_samples = int(max_delay_seconds * self._sample_rate)

        correlation, delay_samples, peak_value = self.gcc_phat(
            signal1, signal2, max_delay_samples
        )

        # Sub-sample interpolation for better precision
        delay_refined = self._subsample_interpolation(correlation, delay_samples)

        delay_seconds = delay_refined / self._sample_rate
        confidence = abs(peak_value)

        return delay_seconds, confidence

    def _subsample_interpolation(
        self,
        correlation: np.ndarray,
        peak_idx: int
    ) -> float:
        """
        Perform parabolic interpolation for sub-sample accuracy.

        Args:
            correlation: Correlation array.
            peak_idx: Index of peak.

        Returns:
            Refined peak position.
        """
        if peak_idx <= 0 or peak_idx >= len(correlation) - 1:
            return float(peak_idx - len(correlation) // 2)

        y0 = abs(correlation[peak_idx - 1])
        y1 = abs(correlation[peak_idx])
        y2 = abs(correlation[peak_idx + 1])

        # Parabolic interpolation
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) < 1e-12:
            offset = 0.0
        else:
            offset = (y0 - y2) / denom

        refined_idx = peak_idx + offset - len(correlation) // 2
        return refined_idx


class SignalProcessor:
    """
    Main signal processing pipeline.

    Combines filtering, spectral analysis, and feature extraction
    into a unified processing pipeline.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        filter_config: Optional[FilterConfig] = None,
        spectral_config: Optional[SpectralConfig] = None
    ):
        """
        Initialize signal processor.

        Args:
            sample_rate: Sample rate in Hz.
            filter_config: Filter configuration.
            spectral_config: Spectral analysis configuration.
        """
        self._sample_rate = sample_rate
        self._filter_bank = FilterBank(sample_rate, filter_config)
        self._spectral_analyzer = SpectralAnalyzer(sample_rate, spectral_config)
        self._gcc_processor = GCCProcessor(sample_rate)
        self._noise_gate = NoiseGate(sample_rate=sample_rate)
        self._agc = AGC(sample_rate=sample_rate)

    def preprocess(
        self,
        data: np.ndarray,
        apply_agc: bool = True,
        apply_noise_gate: bool = False
    ) -> np.ndarray:
        """
        Preprocess audio data.

        Args:
            data: Input audio data (channels x samples or 1D).
            apply_agc: Whether to apply AGC.
            apply_noise_gate: Whether to apply noise gate.

        Returns:
            Preprocessed audio data.
        """
        # Apply filters
        result = self._filter_bank.apply_all(data)

        # Per-channel processing for multi-channel data
        if result.ndim > 1:
            for i in range(result.shape[0]):
                if apply_noise_gate:
                    result[i] = self._noise_gate.process(result[i])
                if apply_agc:
                    result[i] = self._agc.process(result[i])
        else:
            if apply_noise_gate:
                result = self._noise_gate.process(result)
            if apply_agc:
                result = self._agc.process(result)

        return result

    def analyze_spectrum(self, data: np.ndarray) -> SpectralFeatures:
        """
        Analyze spectral content.

        Args:
            data: Input audio data (1D).

        Returns:
            SpectralFeatures object.
        """
        return self._spectral_analyzer.analyze(data)

    def compute_spectrogram(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram.

        Args:
            data: Input audio data.

        Returns:
            Tuple of (frequencies, times, spectrogram).
        """
        return self._spectral_analyzer.compute_spectrogram(data)

    def extract_mfcc(self, data: np.ndarray, num_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features.

        Args:
            data: Input audio data.
            num_mfcc: Number of MFCCs.

        Returns:
            MFCC feature array.
        """
        return self._spectral_analyzer.compute_mfcc(data, num_mfcc)

    def compute_gcc_phat(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute GCC-PHAT between two signals.

        Args:
            signal1: First signal.
            signal2: Second signal.

        Returns:
            Tuple of (delay_seconds, confidence).
        """
        return self._gcc_processor.estimate_tdoa(signal1, signal2)

    def compute_all_tdoas(
        self,
        multi_channel_data: np.ndarray,
        reference_channel: int = 0
    ) -> List[Tuple[float, float]]:
        """
        Compute TDOAs between reference channel and all others.

        Args:
            multi_channel_data: Multi-channel audio data (channels x samples).
            reference_channel: Reference channel index.

        Returns:
            List of (delay_seconds, confidence) for each channel pair.
        """
        tdoas = []
        ref_signal = multi_channel_data[reference_channel]

        for i in range(multi_channel_data.shape[0]):
            if i == reference_channel:
                tdoas.append((0.0, 1.0))
            else:
                delay, conf = self._gcc_processor.estimate_tdoa(
                    ref_signal, multi_channel_data[i]
                )
                tdoas.append((delay, conf))

        return tdoas

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate
