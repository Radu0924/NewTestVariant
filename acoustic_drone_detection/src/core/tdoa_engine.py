"""
Time Difference of Arrival (TDOA) Engine

Provides high-precision TDOA estimation with:
- Sub-sample interpolation
- Multiple algorithm support (GCC-PHAT, SRP-PHAT)
- Multi-pair TDOA computation
- Consistency checking
"""

import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
from scipy.signal import correlate, correlation_lags
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class TDOAResult:
    """Result of TDOA estimation for a microphone pair."""
    channel_i: int
    channel_j: int
    delay_seconds: float
    delay_samples: float
    confidence: float
    snr_estimate: float = 0.0


@dataclass
class TDOAConfig:
    """TDOA engine configuration."""
    method: str = "gcc_phat"  # gcc_phat, gcc_scot, gcc_roth, srp_phat
    fft_size: int = 4096
    max_delay_meters: float = 1.0
    sound_speed: float = 343.0
    interpolation: str = "parabolic"  # none, parabolic, sinc
    consistency_threshold: float = 0.1  # seconds


class InterpolationMethods:
    """Sub-sample interpolation methods for TDOA refinement."""

    @staticmethod
    def parabolic(correlation: np.ndarray, peak_idx: int) -> float:
        """
        Parabolic (quadratic) interpolation.

        Args:
            correlation: Correlation array.
            peak_idx: Index of the peak.

        Returns:
            Refined peak position with sub-sample accuracy.
        """
        if peak_idx <= 0 or peak_idx >= len(correlation) - 1:
            return float(peak_idx)

        y0 = correlation[peak_idx - 1]
        y1 = correlation[peak_idx]
        y2 = correlation[peak_idx + 1]

        # Handle complex values
        if np.iscomplexobj(correlation):
            y0, y1, y2 = np.abs(y0), np.abs(y1), np.abs(y2)

        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) < 1e-12:
            return float(peak_idx)

        offset = (y0 - y2) / denom
        return peak_idx + offset

    @staticmethod
    def sinc(correlation: np.ndarray, peak_idx: int, oversample: int = 16) -> float:
        """
        Sinc interpolation for higher accuracy.

        Args:
            correlation: Correlation array.
            peak_idx: Index of the peak.
            oversample: Oversampling factor.

        Returns:
            Refined peak position.
        """
        # Extract region around peak
        half_width = 4
        start = max(0, peak_idx - half_width)
        end = min(len(correlation), peak_idx + half_width + 1)
        region = correlation[start:end]

        if len(region) < 3:
            return float(peak_idx)

        # Interpolate
        n = len(region)
        x_original = np.arange(n)
        x_interp = np.linspace(0, n - 1, n * oversample)

        # Sinc interpolation kernel
        interp_values = np.zeros(len(x_interp))
        for i, x in enumerate(x_interp):
            sinc_values = np.sinc(x_original - x)
            interp_values[i] = np.abs(np.sum(region * sinc_values))

        # Find refined peak
        refined_idx = np.argmax(interp_values)
        refined_pos = start + x_interp[refined_idx]

        return refined_pos


class TDOAEngine:
    """
    High-precision TDOA estimation engine.

    Provides multiple algorithms for time delay estimation
    with sub-sample accuracy using interpolation.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        config: Optional[TDOAConfig] = None
    ):
        """
        Initialize TDOA engine.

        Args:
            sample_rate: Sample rate in Hz.
            config: TDOA configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or TDOAConfig()
        self._interpolation = InterpolationMethods()
        self._lock = threading.Lock()

        self._max_delay_samples = int(
            self._config.max_delay_meters / self._config.sound_speed * sample_rate
        )

    def estimate_tdoa(
        self,
        signal_i: np.ndarray,
        signal_j: np.ndarray,
        channel_i: int = 0,
        channel_j: int = 1
    ) -> TDOAResult:
        """
        Estimate TDOA between two signals.

        Args:
            signal_i: First signal (reference).
            signal_j: Second signal.
            channel_i: First channel index.
            channel_j: Second channel index.

        Returns:
            TDOAResult with delay and confidence.
        """
        method = self._config.method.lower()

        if method == "gcc_phat":
            correlation, delay_samples, confidence = self._gcc_phat(signal_i, signal_j)
        elif method == "gcc_scot":
            correlation, delay_samples, confidence = self._gcc_scot(signal_i, signal_j)
        elif method == "gcc_roth":
            correlation, delay_samples, confidence = self._gcc_roth(signal_i, signal_j)
        else:
            correlation, delay_samples, confidence = self._gcc_phat(signal_i, signal_j)

        # Sub-sample interpolation
        if self._config.interpolation == "parabolic":
            center = len(correlation) // 2
            peak_idx = center + int(delay_samples)
            if 0 < peak_idx < len(correlation) - 1:
                refined = self._interpolation.parabolic(correlation, peak_idx)
                delay_samples = refined - center
        elif self._config.interpolation == "sinc":
            center = len(correlation) // 2
            peak_idx = center + int(delay_samples)
            refined = self._interpolation.sinc(correlation, peak_idx)
            delay_samples = refined - center

        delay_seconds = delay_samples / self._sample_rate

        # Estimate SNR
        snr = self._estimate_snr(signal_i, signal_j)

        return TDOAResult(
            channel_i=channel_i,
            channel_j=channel_j,
            delay_seconds=delay_seconds,
            delay_samples=delay_samples,
            confidence=confidence,
            snr_estimate=snr
        )

    def _gcc_phat(
        self,
        signal_i: np.ndarray,
        signal_j: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        GCC-PHAT (Phase Transform) algorithm.

        Args:
            signal_i: First signal.
            signal_j: Second signal.

        Returns:
            Tuple of (correlation, delay_samples, confidence).
        """
        # Pad to FFT size
        n = max(len(signal_i), len(signal_j), self._config.fft_size)
        n = int(2 ** np.ceil(np.log2(n)))

        sig_i = np.zeros(n)
        sig_j = np.zeros(n)
        sig_i[:len(signal_i)] = signal_i
        sig_j[:len(signal_j)] = signal_j

        # FFT
        fft_i = fft(sig_i)
        fft_j = fft(sig_j)

        # Cross-spectrum with PHAT weighting
        cross_spectrum = fft_i * np.conj(fft_j)
        magnitude = np.abs(cross_spectrum) + 1e-12
        weighted_spectrum = cross_spectrum / magnitude

        # IFFT
        correlation = np.real(ifft(weighted_spectrum))
        correlation = np.fft.fftshift(correlation)

        # Find peak within max delay
        center = len(correlation) // 2
        search_start = max(0, center - self._max_delay_samples)
        search_end = min(len(correlation), center + self._max_delay_samples + 1)

        search_region = np.abs(correlation[search_start:search_end])
        peak_local_idx = np.argmax(search_region)
        peak_idx = search_start + peak_local_idx

        delay_samples = peak_idx - center
        confidence = np.abs(correlation[peak_idx])

        return correlation, delay_samples, confidence

    def _gcc_scot(
        self,
        signal_i: np.ndarray,
        signal_j: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        GCC-SCOT (Smoothed Coherence Transform) algorithm.

        Args:
            signal_i: First signal.
            signal_j: Second signal.

        Returns:
            Tuple of (correlation, delay_samples, confidence).
        """
        n = max(len(signal_i), len(signal_j), self._config.fft_size)
        n = int(2 ** np.ceil(np.log2(n)))

        sig_i = np.zeros(n)
        sig_j = np.zeros(n)
        sig_i[:len(signal_i)] = signal_i
        sig_j[:len(signal_j)] = signal_j

        fft_i = fft(sig_i)
        fft_j = fft(sig_j)

        # Auto-spectra
        Gii = np.abs(fft_i) ** 2 + 1e-12
        Gjj = np.abs(fft_j) ** 2 + 1e-12

        # Cross-spectrum with SCOT weighting
        cross_spectrum = fft_i * np.conj(fft_j)
        weighting = np.sqrt(Gii * Gjj)
        weighted_spectrum = cross_spectrum / weighting

        correlation = np.real(ifft(weighted_spectrum))
        correlation = np.fft.fftshift(correlation)

        center = len(correlation) // 2
        search_start = max(0, center - self._max_delay_samples)
        search_end = min(len(correlation), center + self._max_delay_samples + 1)

        search_region = np.abs(correlation[search_start:search_end])
        peak_local_idx = np.argmax(search_region)
        peak_idx = search_start + peak_local_idx

        delay_samples = peak_idx - center
        confidence = np.abs(correlation[peak_idx])

        return correlation, delay_samples, confidence

    def _gcc_roth(
        self,
        signal_i: np.ndarray,
        signal_j: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        GCC-ROTH algorithm.

        Args:
            signal_i: First signal.
            signal_j: Second signal.

        Returns:
            Tuple of (correlation, delay_samples, confidence).
        """
        n = max(len(signal_i), len(signal_j), self._config.fft_size)
        n = int(2 ** np.ceil(np.log2(n)))

        sig_i = np.zeros(n)
        sig_j = np.zeros(n)
        sig_i[:len(signal_i)] = signal_i
        sig_j[:len(signal_j)] = signal_j

        fft_i = fft(sig_i)
        fft_j = fft(sig_j)

        # Roth weighting
        Gii = np.abs(fft_i) ** 2 + 1e-12

        cross_spectrum = fft_i * np.conj(fft_j)
        weighted_spectrum = cross_spectrum / Gii

        correlation = np.real(ifft(weighted_spectrum))
        correlation = np.fft.fftshift(correlation)

        center = len(correlation) // 2
        search_start = max(0, center - self._max_delay_samples)
        search_end = min(len(correlation), center + self._max_delay_samples + 1)

        search_region = np.abs(correlation[search_start:search_end])
        peak_local_idx = np.argmax(search_region)
        peak_idx = search_start + peak_local_idx

        delay_samples = peak_idx - center
        confidence = np.abs(correlation[peak_idx])

        return correlation, delay_samples, confidence

    def _estimate_snr(
        self,
        signal_i: np.ndarray,
        signal_j: np.ndarray
    ) -> float:
        """Estimate SNR based on signal correlation."""
        correlation = correlate(signal_i, signal_j, mode='full')
        peak = np.max(np.abs(correlation))
        noise_floor = np.percentile(np.abs(correlation), 10)

        if noise_floor > 0:
            snr = 20 * np.log10(peak / noise_floor)
        else:
            snr = 60.0

        return snr

    def compute_all_tdoas(
        self,
        multi_channel_data: np.ndarray,
        reference_channel: int = 0
    ) -> List[TDOAResult]:
        """
        Compute TDOAs from reference channel to all others.

        Args:
            multi_channel_data: Multi-channel data (channels x samples).
            reference_channel: Reference channel index.

        Returns:
            List of TDOAResult objects.
        """
        results = []
        ref_signal = multi_channel_data[reference_channel]

        for i in range(multi_channel_data.shape[0]):
            if i == reference_channel:
                results.append(TDOAResult(
                    channel_i=reference_channel,
                    channel_j=i,
                    delay_seconds=0.0,
                    delay_samples=0.0,
                    confidence=1.0,
                    snr_estimate=60.0
                ))
            else:
                result = self.estimate_tdoa(
                    ref_signal,
                    multi_channel_data[i],
                    reference_channel,
                    i
                )
                results.append(result)

        return results

    def compute_all_pairs(
        self,
        multi_channel_data: np.ndarray
    ) -> List[TDOAResult]:
        """
        Compute TDOAs for all unique microphone pairs.

        Args:
            multi_channel_data: Multi-channel data (channels x samples).

        Returns:
            List of TDOAResult objects for all pairs.
        """
        n_channels = multi_channel_data.shape[0]
        results = []

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                result = self.estimate_tdoa(
                    multi_channel_data[i],
                    multi_channel_data[j],
                    i, j
                )
                results.append(result)

        return results

    def check_consistency(self, tdoas: List[TDOAResult]) -> Dict[str, Any]:
        """
        Check TDOA consistency using closure relations.

        For three microphones i, j, k:
        tau_ij + tau_jk + tau_ki should equal 0

        Args:
            tdoas: List of TDOA results.

        Returns:
            Dictionary with consistency metrics.
        """
        # Build TDOA matrix
        channels = set()
        for tdoa in tdoas:
            channels.add(tdoa.channel_i)
            channels.add(tdoa.channel_j)

        n = max(channels) + 1
        tdoa_matrix = np.zeros((n, n))

        for tdoa in tdoas:
            tdoa_matrix[tdoa.channel_i, tdoa.channel_j] = tdoa.delay_seconds
            tdoa_matrix[tdoa.channel_j, tdoa.channel_i] = -tdoa.delay_seconds

        # Check closure relations
        errors = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    closure_error = (
                        tdoa_matrix[i, j] +
                        tdoa_matrix[j, k] +
                        tdoa_matrix[k, i]
                    )
                    errors.append(abs(closure_error))

        if errors:
            return {
                'is_consistent': max(errors) < self._config.consistency_threshold,
                'max_error': max(errors),
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'num_triplets': len(errors)
            }
        else:
            return {
                'is_consistent': True,
                'max_error': 0.0,
                'mean_error': 0.0,
                'std_error': 0.0,
                'num_triplets': 0
            }

    def tdoas_to_range_differences(
        self,
        tdoas: List[TDOAResult]
    ) -> np.ndarray:
        """
        Convert TDOAs to range differences.

        Args:
            tdoas: List of TDOA results.

        Returns:
            Array of range differences in meters.
        """
        range_diffs = []
        for tdoa in tdoas:
            range_diff = tdoa.delay_seconds * self._config.sound_speed
            range_diffs.append(range_diff)

        return np.array(range_diffs)

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def max_delay_samples(self) -> int:
        """Get maximum delay in samples."""
        return self._max_delay_samples

    def update_config(self, config: TDOAConfig) -> None:
        """Update TDOA configuration."""
        with self._lock:
            self._config = config
            self._max_delay_samples = int(
                config.max_delay_meters / config.sound_speed * self._sample_rate
            )
