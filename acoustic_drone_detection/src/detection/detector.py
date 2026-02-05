"""
Drone Detection Module

Provides drone presence detection from acoustic signals:
- Harmonic detection for brushless motors
- Energy-based detection
- Spectral pattern matching
- Multi-stage detection pipeline
"""

import numpy as np
from scipy.signal import find_peaks, spectrogram
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading


class DetectionStatus(Enum):
    """Detection status."""
    NO_DETECTION = "no_detection"
    POSSIBLE = "possible"
    PROBABLE = "probable"
    CONFIRMED = "confirmed"


@dataclass
class DetectionResult:
    """Result of drone detection."""
    status: DetectionStatus
    confidence: float
    snr: float
    dominant_frequencies: np.ndarray
    harmonic_score: float
    energy_score: float
    timestamp: float
    metadata: Dict[str, Any] = None


@dataclass
class DetectorConfig:
    """Detector configuration."""
    energy_threshold_db: float = -40.0
    min_confidence: float = 0.3
    harmonic_threshold: float = 0.5
    min_harmonics: int = 2
    fundamental_range: Tuple[float, float] = (100.0, 500.0)
    detection_band: Tuple[float, float] = (80.0, 8000.0)
    smoothing_window: int = 5


class EnergyDetector:
    """
    Energy-based detection.

    Detects acoustic events based on signal energy levels.
    """

    def __init__(self, sample_rate: int = 48000, config: Optional[DetectorConfig] = None):
        """
        Initialize energy detector.

        Args:
            sample_rate: Sample rate in Hz.
            config: Detector configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or DetectorConfig()
        self._background_energy = None
        self._adaptation_rate = 0.01

    def detect(self, data: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect based on energy level.

        Args:
            data: Audio signal.

        Returns:
            Tuple of (is_detected, energy_db, snr).
        """
        # Calculate signal energy
        energy = np.sqrt(np.mean(data ** 2))
        energy_db = 20 * np.log10(energy + 1e-12)

        # Update background estimate
        if self._background_energy is None:
            self._background_energy = energy
        else:
            if energy < self._background_energy * 1.5:
                self._background_energy = (
                    (1 - self._adaptation_rate) * self._background_energy +
                    self._adaptation_rate * energy
                )

        # Calculate SNR (avoid log10(0) or division by ~0)
        if energy <= 1e-12 or (self._background_energy is not None and self._background_energy <= 1e-12):
            snr = 0.0
        else:
            snr = 20 * np.log10(energy / (self._background_energy + 1e-12))

        # Detection decision
        is_detected = energy_db > self._config.energy_threshold_db

        return is_detected, energy_db, snr

    def reset_background(self) -> None:
        """Reset background energy estimate."""
        self._background_energy = None


class HarmonicDetector:
    """
    Harmonic pattern detection.

    Detects harmonic series characteristic of brushless motors.
    """

    def __init__(self, sample_rate: int = 48000, config: Optional[DetectorConfig] = None):
        """
        Initialize harmonic detector.

        Args:
            sample_rate: Sample rate in Hz.
            config: Detector configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or DetectorConfig()

    def detect(
        self,
        data: np.ndarray,
        n_fft: int = 2048
    ) -> Tuple[bool, float, List[float], float]:
        """
        Detect harmonic patterns.

        Args:
            data: Audio signal.
            n_fft: FFT size.

        Returns:
            Tuple of (is_detected, fundamental_freq, harmonics, score).
        """
        # Compute spectrum
        window = np.hanning(min(len(data), n_fft))
        spectrum = np.abs(np.fft.rfft(data[:len(window)] * window))
        frequencies = np.fft.rfftfreq(len(window), 1 / self._sample_rate)

        # Limit to detection band
        band_mask = (frequencies >= self._config.detection_band[0]) & \
                    (frequencies <= self._config.detection_band[1])
        spectrum_band = spectrum[band_mask]
        freqs_band = frequencies[band_mask]

        if len(spectrum_band) == 0:
            return False, 0.0, [], 0.0

        # Find peaks
        threshold = np.mean(spectrum_band) + 2 * np.std(spectrum_band)
        peaks, properties = find_peaks(spectrum_band, height=threshold, distance=5)

        if len(peaks) < self._config.min_harmonics:
            return False, 0.0, [], 0.0

        peak_freqs = freqs_band[peaks]
        peak_heights = spectrum_band[peaks]

        # Search for harmonic series
        best_fundamental = 0.0
        best_harmonics = []
        best_score = 0.0

        for f0 in peak_freqs:
            if not (self._config.fundamental_range[0] <= f0 <= self._config.fundamental_range[1]):
                continue

            harmonics = []
            score = 0.0

            # Check for harmonics
            for n in range(1, 10):
                expected_freq = f0 * n
                if expected_freq > self._config.detection_band[1]:
                    break

                # Find closest peak
                freq_diffs = np.abs(peak_freqs - expected_freq)
                min_idx = np.argmin(freq_diffs)

                if freq_diffs[min_idx] < f0 * 0.05:  # 5% tolerance
                    harmonics.append(peak_freqs[min_idx])
                    score += peak_heights[min_idx]

            if len(harmonics) > len(best_harmonics):
                best_fundamental = f0
                best_harmonics = harmonics
                best_score = score

        # Normalize score
        if best_score > 0:
            best_score = best_score / np.sum(spectrum_band)

        is_detected = (
            len(best_harmonics) >= self._config.min_harmonics and
            best_score >= self._config.harmonic_threshold
        )

        return is_detected, best_fundamental, best_harmonics, best_score


class SpectralPatternMatcher:
    """
    Spectral pattern matching for drone signatures.

    Matches observed spectra against known drone signatures.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize pattern matcher.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate
        self._patterns: Dict[str, np.ndarray] = {}

    def add_pattern(self, name: str, spectrum: np.ndarray) -> None:
        """
        Add a reference pattern.

        Args:
            name: Pattern name.
            spectrum: Reference spectrum.
        """
        # Normalize pattern
        normalized = spectrum / (np.linalg.norm(spectrum) + 1e-12)
        self._patterns[name] = normalized

    def match(self, spectrum: np.ndarray) -> Tuple[str, float]:
        """
        Match spectrum against known patterns.

        Args:
            spectrum: Input spectrum.

        Returns:
            Tuple of (best_match_name, similarity_score).
        """
        if not self._patterns:
            return "unknown", 0.0

        # Normalize input
        normalized = spectrum / (np.linalg.norm(spectrum) + 1e-12)

        best_match = "unknown"
        best_score = 0.0

        for name, pattern in self._patterns.items():
            # Resize if necessary
            if len(pattern) != len(normalized):
                pattern_resized = np.interp(
                    np.linspace(0, 1, len(normalized)),
                    np.linspace(0, 1, len(pattern)),
                    pattern
                )
            else:
                pattern_resized = pattern

            # Correlation score
            score = np.abs(np.correlate(normalized, pattern_resized, mode='valid')[0])

            if score > best_score:
                best_score = score
                best_match = name

        return best_match, best_score


class DroneDetector:
    """
    Main drone detection engine.

    Combines multiple detection methods for robust drone presence detection.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        config: Optional[DetectorConfig] = None
    ):
        """
        Initialize drone detector.

        Args:
            sample_rate: Sample rate in Hz.
            config: Detector configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or DetectorConfig()

        self._energy_detector = EnergyDetector(sample_rate, config)
        self._harmonic_detector = HarmonicDetector(sample_rate, config)
        self._pattern_matcher = SpectralPatternMatcher(sample_rate)

        self._detection_history: List[DetectionResult] = []
        self._lock = threading.Lock()

    def detect(self, data: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """
        Perform drone detection.

        Args:
            data: Audio signal (mono).
            timestamp: Detection timestamp.

        Returns:
            DetectionResult object.
        """
        if data.ndim > 1:
            data = data.mean(axis=0)

        # Energy detection
        energy_detected, energy_db, snr = self._energy_detector.detect(data)

        # Harmonic detection
        harmonic_detected, fundamental, harmonics, harmonic_score = \
            self._harmonic_detector.detect(data)

        # Calculate overall confidence
        confidence = 0.0
        if energy_detected:
            confidence += 0.3
        if harmonic_detected:
            confidence += 0.5
            confidence += min(0.2, harmonic_score)

        # Determine status
        if confidence >= 0.7:
            status = DetectionStatus.CONFIRMED
        elif confidence >= 0.5:
            status = DetectionStatus.PROBABLE
        elif confidence >= 0.3:
            status = DetectionStatus.POSSIBLE
        else:
            status = DetectionStatus.NO_DETECTION

        result = DetectionResult(
            status=status,
            confidence=confidence,
            snr=snr,
            dominant_frequencies=np.array(harmonics) if harmonics else np.array([]),
            harmonic_score=harmonic_score,
            energy_score=energy_db,
            timestamp=timestamp,
            metadata={
                'fundamental': fundamental,
                'energy_detected': energy_detected,
                'harmonic_detected': harmonic_detected
            }
        )

        with self._lock:
            self._detection_history.append(result)
            if len(self._detection_history) > 100:
                self._detection_history.pop(0)

        return result

    def get_smoothed_detection(
        self,
        window_size: int = 5
    ) -> Optional[DetectionResult]:
        """
        Get temporally smoothed detection result.

        Args:
            window_size: Number of recent detections to consider.

        Returns:
            Smoothed DetectionResult or None.
        """
        with self._lock:
            if len(self._detection_history) < window_size:
                return self._detection_history[-1] if self._detection_history else None

            recent = self._detection_history[-window_size:]

        # Average confidence
        avg_confidence = np.mean([r.confidence for r in recent])

        # Majority vote for status
        confirmed_count = sum(1 for r in recent if r.status == DetectionStatus.CONFIRMED)
        probable_count = sum(1 for r in recent if r.status == DetectionStatus.PROBABLE)

        if confirmed_count >= window_size // 2:
            status = DetectionStatus.CONFIRMED
        elif confirmed_count + probable_count >= window_size // 2:
            status = DetectionStatus.PROBABLE
        elif avg_confidence >= 0.3:
            status = DetectionStatus.POSSIBLE
        else:
            status = DetectionStatus.NO_DETECTION

        # Average other metrics
        avg_snr = np.mean([r.snr for r in recent])

        # Collect all dominant frequencies
        all_freqs = []
        for r in recent:
            all_freqs.extend(r.dominant_frequencies.tolist())

        return DetectionResult(
            status=status,
            confidence=avg_confidence,
            snr=avg_snr,
            dominant_frequencies=np.array(sorted(set(all_freqs))[:10]),
            harmonic_score=np.mean([r.harmonic_score for r in recent]),
            energy_score=np.mean([r.energy_score for r in recent]),
            timestamp=recent[-1].timestamp
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._energy_detector.reset_background()
        with self._lock:
            self._detection_history.clear()

    def update_config(self, config: DetectorConfig) -> None:
        """Update detector configuration."""
        self._config = config
        self._energy_detector = EnergyDetector(self._sample_rate, config)
        self._harmonic_detector = HarmonicDetector(self._sample_rate, config)
