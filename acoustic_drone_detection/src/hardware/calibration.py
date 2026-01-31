"""
System Calibration Module

Provides calibration routines for the acoustic detection system:
- Microphone sensitivity calibration
- Phase alignment calibration
- Position verification
- Noise floor measurement
"""

import numpy as np
from scipy.signal import correlate
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml
from pathlib import Path


@dataclass
class MicrophoneCalibration:
    """Calibration data for a single microphone."""
    mic_id: int
    sensitivity_db: float = 0.0  # Relative sensitivity
    phase_offset_samples: float = 0.0  # Phase offset in samples
    noise_floor_db: float = -60.0  # Noise floor level
    is_active: bool = True
    last_calibrated: Optional[str] = None


@dataclass
class ArrayCalibration:
    """Complete array calibration data."""
    array_name: str
    sample_rate: int
    calibration_date: str
    microphones: List[MicrophoneCalibration]
    reference_mic: int = 0
    temperature_celsius: Optional[float] = None
    sound_speed: float = 343.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensitivityCalibrator:
    """
    Calibrates microphone sensitivities.

    Uses a known test signal to measure relative sensitivity differences.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize sensitivity calibrator.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate

    def calibrate(
        self,
        multi_channel_data: np.ndarray,
        reference_channel: int = 0
    ) -> List[float]:
        """
        Calibrate sensitivities from recorded data.

        Args:
            multi_channel_data: Multi-channel audio data (channels x samples).
            reference_channel: Reference microphone index.

        Returns:
            List of sensitivity offsets in dB.
        """
        n_channels = multi_channel_data.shape[0]
        sensitivities = []

        # Calculate RMS for reference channel
        ref_rms = np.sqrt(np.mean(multi_channel_data[reference_channel] ** 2))
        ref_db = 20 * np.log10(ref_rms + 1e-12)

        for ch in range(n_channels):
            ch_rms = np.sqrt(np.mean(multi_channel_data[ch] ** 2))
            ch_db = 20 * np.log10(ch_rms + 1e-12)

            # Relative sensitivity
            sensitivity_offset = ch_db - ref_db
            sensitivities.append(sensitivity_offset)

        return sensitivities

    def generate_compensation(
        self,
        sensitivities: List[float]
    ) -> np.ndarray:
        """
        Generate compensation gains from sensitivities.

        Args:
            sensitivities: List of sensitivity offsets in dB.

        Returns:
            Array of linear gain factors.
        """
        # Normalize to mean
        mean_sensitivity = np.mean(sensitivities)
        normalized = [s - mean_sensitivity for s in sensitivities]

        # Convert to linear gain (invert to compensate)
        gains = [10 ** (-s / 20) for s in normalized]

        return np.array(gains)


class PhaseCalibrator:
    """
    Calibrates phase/timing alignment between channels.

    Measures and corrects inter-channel delays.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize phase calibrator.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate

    def calibrate(
        self,
        multi_channel_data: np.ndarray,
        reference_channel: int = 0
    ) -> List[float]:
        """
        Measure phase offsets between channels.

        Args:
            multi_channel_data: Multi-channel audio data.
            reference_channel: Reference microphone index.

        Returns:
            List of phase offsets in samples.
        """
        n_channels = multi_channel_data.shape[0]
        phase_offsets = []

        ref_signal = multi_channel_data[reference_channel]

        for ch in range(n_channels):
            if ch == reference_channel:
                phase_offsets.append(0.0)
                continue

            ch_signal = multi_channel_data[ch]

            # Cross-correlation
            correlation = correlate(ref_signal, ch_signal, mode='full')
            lags = np.arange(-len(ch_signal) + 1, len(ref_signal))

            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            peak_lag = lags[peak_idx]

            # Sub-sample interpolation
            if 0 < peak_idx < len(correlation) - 1:
                y0 = correlation[peak_idx - 1]
                y1 = correlation[peak_idx]
                y2 = correlation[peak_idx + 1]

                denom = 2 * (2 * y1 - y0 - y2)
                if abs(denom) > 1e-12:
                    offset = (y0 - y2) / denom
                    peak_lag += offset

            phase_offsets.append(float(peak_lag))

        return phase_offsets

    def apply_correction(
        self,
        data: np.ndarray,
        phase_offsets: List[float]
    ) -> np.ndarray:
        """
        Apply phase correction to multi-channel data.

        Args:
            data: Multi-channel audio data.
            phase_offsets: Phase offsets to correct.

        Returns:
            Phase-corrected data.
        """
        n_channels, n_samples = data.shape
        corrected = np.zeros_like(data)

        for ch in range(n_channels):
            offset = phase_offsets[ch]

            # Integer and fractional parts
            int_offset = int(np.round(offset))
            frac_offset = offset - int_offset

            # Apply integer shift
            if int_offset > 0:
                corrected[ch, int_offset:] = data[ch, :-int_offset]
            elif int_offset < 0:
                corrected[ch, :int_offset] = data[ch, -int_offset:]
            else:
                corrected[ch] = data[ch]

            # Apply fractional shift using linear interpolation
            if abs(frac_offset) > 0.01:
                # Simple linear interpolation
                alpha = abs(frac_offset)
                if frac_offset > 0:
                    corrected[ch, 1:] = (1 - alpha) * corrected[ch, 1:] + alpha * corrected[ch, :-1]
                else:
                    corrected[ch, :-1] = (1 - alpha) * corrected[ch, :-1] + alpha * corrected[ch, 1:]

        return corrected


class NoiseFloorAnalyzer:
    """
    Analyzes noise floor for each channel.

    Measures ambient noise levels for threshold calibration.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize noise floor analyzer.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate

    def measure(
        self,
        multi_channel_data: np.ndarray,
        window_seconds: float = 1.0
    ) -> List[float]:
        """
        Measure noise floor for each channel.

        Args:
            multi_channel_data: Multi-channel audio data.
            window_seconds: Analysis window in seconds.

        Returns:
            List of noise floor levels in dB.
        """
        n_channels = multi_channel_data.shape[0]
        noise_floors = []

        window_samples = int(window_seconds * self._sample_rate)

        for ch in range(n_channels):
            ch_data = multi_channel_data[ch]

            # Analyze in windows, take minimum (assumes signal presence)
            min_rms = float('inf')

            for start in range(0, len(ch_data) - window_samples, window_samples // 2):
                window = ch_data[start:start + window_samples]
                rms = np.sqrt(np.mean(window ** 2))
                min_rms = min(min_rms, rms)

            noise_db = 20 * np.log10(min_rms + 1e-12)
            noise_floors.append(noise_db)

        return noise_floors

    def get_spectral_noise(
        self,
        multi_channel_data: np.ndarray,
        n_fft: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spectral noise profile.

        Args:
            multi_channel_data: Multi-channel audio data.
            n_fft: FFT size.

        Returns:
            Tuple of (frequencies, noise_spectrum per channel).
        """
        n_channels = multi_channel_data.shape[0]

        frequencies = np.fft.rfftfreq(n_fft, 1 / self._sample_rate)
        noise_spectra = np.zeros((n_channels, len(frequencies)))

        for ch in range(n_channels):
            # Compute average spectrum
            n_windows = len(multi_channel_data[ch]) // n_fft
            spectrum = np.zeros(len(frequencies))

            for i in range(n_windows):
                window = multi_channel_data[ch, i * n_fft:(i + 1) * n_fft]
                window = window * np.hanning(n_fft)
                spectrum += np.abs(np.fft.rfft(window)) ** 2

            spectrum /= n_windows
            noise_spectra[ch] = 10 * np.log10(spectrum + 1e-12)

        return frequencies, noise_spectra


class CalibrationManager:
    """
    Manages complete system calibration.

    Coordinates all calibration routines and stores results.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize calibration manager.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate
        self._sensitivity_cal = SensitivityCalibrator(sample_rate)
        self._phase_cal = PhaseCalibrator(sample_rate)
        self._noise_analyzer = NoiseFloorAnalyzer(sample_rate)

        self._current_calibration: Optional[ArrayCalibration] = None

    def run_full_calibration(
        self,
        multi_channel_data: np.ndarray,
        array_name: str,
        reference_channel: int = 0,
        temperature: Optional[float] = None
    ) -> ArrayCalibration:
        """
        Run complete calibration sequence.

        Args:
            multi_channel_data: Calibration recording (channels x samples).
            array_name: Name of the microphone array.
            reference_channel: Reference microphone index.
            temperature: Ambient temperature in Celsius.

        Returns:
            ArrayCalibration object.
        """
        n_channels = multi_channel_data.shape[0]

        # Run calibrations
        sensitivities = self._sensitivity_cal.calibrate(
            multi_channel_data, reference_channel
        )
        phase_offsets = self._phase_cal.calibrate(
            multi_channel_data, reference_channel
        )
        noise_floors = self._noise_analyzer.measure(multi_channel_data)

        # Create calibration objects
        mic_cals = []
        now = datetime.now().isoformat()

        for ch in range(n_channels):
            mic_cal = MicrophoneCalibration(
                mic_id=ch,
                sensitivity_db=sensitivities[ch],
                phase_offset_samples=phase_offsets[ch],
                noise_floor_db=noise_floors[ch],
                is_active=True,
                last_calibrated=now
            )
            mic_cals.append(mic_cal)

        # Calculate sound speed from temperature if provided
        sound_speed = 343.0
        if temperature is not None:
            sound_speed = 331.3 + 0.606 * temperature

        calibration = ArrayCalibration(
            array_name=array_name,
            sample_rate=self._sample_rate,
            calibration_date=now,
            microphones=mic_cals,
            reference_mic=reference_channel,
            temperature_celsius=temperature,
            sound_speed=sound_speed
        )

        self._current_calibration = calibration
        return calibration

    def apply_calibration(
        self,
        data: np.ndarray,
        calibration: Optional[ArrayCalibration] = None
    ) -> np.ndarray:
        """
        Apply calibration to multi-channel data.

        Args:
            data: Multi-channel audio data.
            calibration: Calibration to apply (uses current if not specified).

        Returns:
            Calibrated data.
        """
        cal = calibration or self._current_calibration
        if cal is None:
            return data

        result = data.copy()

        # Apply sensitivity compensation
        sensitivities = [m.sensitivity_db for m in cal.microphones]
        gains = self._sensitivity_cal.generate_compensation(sensitivities)

        for ch in range(min(len(gains), data.shape[0])):
            result[ch] *= gains[ch]

        # Apply phase correction
        phase_offsets = [m.phase_offset_samples for m in cal.microphones]
        result = self._phase_cal.apply_correction(result, phase_offsets)

        return result

    def save_calibration(
        self,
        filepath: str,
        calibration: Optional[ArrayCalibration] = None
    ) -> None:
        """
        Save calibration to file.

        Args:
            filepath: Output file path.
            calibration: Calibration to save.
        """
        cal = calibration or self._current_calibration
        if cal is None:
            raise ValueError("No calibration to save")

        data = {
            'array_name': cal.array_name,
            'sample_rate': cal.sample_rate,
            'calibration_date': cal.calibration_date,
            'reference_mic': cal.reference_mic,
            'temperature_celsius': cal.temperature_celsius,
            'sound_speed': cal.sound_speed,
            'microphones': [
                {
                    'mic_id': m.mic_id,
                    'sensitivity_db': m.sensitivity_db,
                    'phase_offset_samples': m.phase_offset_samples,
                    'noise_floor_db': m.noise_floor_db,
                    'is_active': m.is_active,
                    'last_calibrated': m.last_calibrated
                }
                for m in cal.microphones
            ],
            'metadata': cal.metadata
        }

        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def load_calibration(self, filepath: str) -> ArrayCalibration:
        """
        Load calibration from file.

        Args:
            filepath: Input file path.

        Returns:
            ArrayCalibration object.
        """
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

        mic_cals = [
            MicrophoneCalibration(
                mic_id=m['mic_id'],
                sensitivity_db=m['sensitivity_db'],
                phase_offset_samples=m['phase_offset_samples'],
                noise_floor_db=m['noise_floor_db'],
                is_active=m.get('is_active', True),
                last_calibrated=m.get('last_calibrated')
            )
            for m in data['microphones']
        ]

        calibration = ArrayCalibration(
            array_name=data['array_name'],
            sample_rate=data['sample_rate'],
            calibration_date=data['calibration_date'],
            microphones=mic_cals,
            reference_mic=data.get('reference_mic', 0),
            temperature_celsius=data.get('temperature_celsius'),
            sound_speed=data.get('sound_speed', 343.0),
            metadata=data.get('metadata', {})
        )

        self._current_calibration = calibration
        return calibration

    @property
    def current_calibration(self) -> Optional[ArrayCalibration]:
        """Get current calibration."""
        return self._current_calibration
