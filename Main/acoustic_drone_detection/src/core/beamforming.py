"""
Beamforming Module

Provides Direction of Arrival (DOA) estimation algorithms:
- Delay-and-Sum Beamformer
- MVDR (Minimum Variance Distortionless Response)
- MUSIC (Multiple Signal Classification)
- ESPRIT
- SRP-PHAT (Steered Response Power)
"""

import numpy as np
from scipy.linalg import eigh, inv, pinv
from scipy.fft import fft, rfft
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class DOAResult:
    """Result of DOA estimation."""
    azimuth: float  # degrees, 0-360
    elevation: float  # degrees, -90 to +90
    power: float  # relative power/confidence
    spectrum: Optional[np.ndarray] = None


@dataclass
class BeamformingConfig:
    """Beamforming configuration."""
    method: str = "music"  # das, mvdr, music, esprit, srp_phat
    num_azimuth: int = 360
    num_elevation: int = 91
    azimuth_range: Tuple[float, float] = (0.0, 360.0)
    elevation_range: Tuple[float, float] = (-90.0, 90.0)
    num_sources: int = 1
    frequency_range: Tuple[float, float] = (100.0, 8000.0)
    diagonal_loading: float = 1e-6


class SteeringVectorGenerator:
    """
    Generates steering vectors for microphone arrays.

    Computes phase delays for different look directions.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 48000,
        sound_speed: float = 343.0
    ):
        """
        Initialize steering vector generator.

        Args:
            mic_positions: Microphone positions (N x 3) in meters.
            sample_rate: Sample rate in Hz.
            sound_speed: Speed of sound in m/s.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sample_rate = sample_rate
        self._sound_speed = sound_speed
        self._n_mics = len(mic_positions)

    def compute_steering_vector(
        self,
        azimuth: float,
        elevation: float,
        frequency: float
    ) -> np.ndarray:
        """
        Compute steering vector for a given direction and frequency.

        Args:
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.
            frequency: Frequency in Hz.

        Returns:
            Complex steering vector.
        """
        # Direction vector (unit vector pointing towards source)
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        direction = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        # Time delays
        delays = self._mic_positions @ direction / self._sound_speed

        # Phase shifts
        steering_vector = np.exp(-2j * np.pi * frequency * delays)

        return steering_vector

    def compute_steering_matrix(
        self,
        azimuths: np.ndarray,
        elevations: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """
        Compute steering matrix for multiple directions.

        Args:
            azimuths: Array of azimuth angles.
            elevations: Array of elevation angles.
            frequency: Frequency in Hz.

        Returns:
            Steering matrix (n_mics x n_directions).
        """
        n_directions = len(azimuths) * len(elevations)
        steering_matrix = np.zeros((self._n_mics, n_directions), dtype=np.complex128)

        idx = 0
        for az in azimuths:
            for el in elevations:
                steering_matrix[:, idx] = self.compute_steering_vector(az, el, frequency)
                idx += 1

        return steering_matrix

    def precompute_steering_grid(
        self,
        n_azimuth: int,
        n_elevation: int,
        frequency: float,
        azimuth_range: Tuple[float, float] = (0.0, 360.0),
        elevation_range: Tuple[float, float] = (-90.0, 90.0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute steering vectors on a grid.

        Args:
            n_azimuth: Number of azimuth points.
            n_elevation: Number of elevation points.
            frequency: Frequency in Hz.
            azimuth_range: Azimuth range in degrees.
            elevation_range: Elevation range in degrees.

        Returns:
            Tuple of (azimuths, elevations, steering_vectors).
        """
        azimuths = np.linspace(*azimuth_range, n_azimuth)
        elevations = np.linspace(*elevation_range, n_elevation)

        steering_vectors = np.zeros(
            (n_azimuth, n_elevation, self._n_mics),
            dtype=np.complex128
        )

        for i, az in enumerate(azimuths):
            for j, el in enumerate(elevations):
                steering_vectors[i, j] = self.compute_steering_vector(az, el, frequency)

        return azimuths, elevations, steering_vectors


class CovarianceEstimator:
    """
    Spatial covariance matrix estimator.

    Provides various methods for estimating the covariance matrix
    from multi-channel audio data.
    """

    @staticmethod
    def sample_covariance(data: np.ndarray) -> np.ndarray:
        """
        Estimate covariance using sample covariance.

        Args:
            data: Multi-channel data (channels x samples).

        Returns:
            Covariance matrix.
        """
        # Center the data
        centered = data - data.mean(axis=1, keepdims=True)

        # Sample covariance
        n_samples = data.shape[1]
        cov = (centered @ centered.conj().T) / (n_samples - 1)

        return cov

    @staticmethod
    def frequency_domain_covariance(
        data: np.ndarray,
        sample_rate: int,
        freq_range: Tuple[float, float] = (100.0, 8000.0),
        n_fft: int = 2048
    ) -> np.ndarray:
        """
        Estimate covariance in frequency domain.

        Args:
            data: Multi-channel data (channels x samples).
            sample_rate: Sample rate in Hz.
            freq_range: Frequency range to consider.
            n_fft: FFT size.

        Returns:
            Covariance matrix.
        """
        n_channels = data.shape[0]

        # Compute FFT for each channel
        fft_data = rfft(data, n=n_fft, axis=1)
        freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

        # Frequency mask
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        fft_data = fft_data[:, freq_mask]

        # Covariance
        cov = (fft_data @ fft_data.conj().T) / fft_data.shape[1]

        return cov

    @staticmethod
    def diagonal_loading(cov: np.ndarray, loading: float = 1e-6) -> np.ndarray:
        """
        Apply diagonal loading for numerical stability.

        Args:
            cov: Covariance matrix.
            loading: Loading factor.

        Returns:
            Regularized covariance matrix.
        """
        n = cov.shape[0]
        return cov + loading * np.eye(n)


class DelayAndSumBeamformer:
    """
    Delay-and-Sum (DAS) Beamformer.

    Simple but robust beamforming method.
    """

    def __init__(
        self,
        steering_generator: SteeringVectorGenerator,
        config: BeamformingConfig
    ):
        """
        Initialize DAS beamformer.

        Args:
            steering_generator: Steering vector generator.
            config: Beamforming configuration.
        """
        self._steering = steering_generator
        self._config = config

    def compute_spectrum(
        self,
        cov: np.ndarray,
        frequency: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DAS spatial spectrum.

        Args:
            cov: Covariance matrix.
            frequency: Center frequency.

        Returns:
            Tuple of (azimuths, elevations, spectrum).
        """
        azimuths, elevations, steering_vectors = self._steering.precompute_steering_grid(
            self._config.num_azimuth,
            self._config.num_elevation,
            frequency,
            self._config.azimuth_range,
            self._config.elevation_range
        )

        spectrum = np.zeros((len(azimuths), len(elevations)))

        for i in range(len(azimuths)):
            for j in range(len(elevations)):
                sv = steering_vectors[i, j]
                spectrum[i, j] = np.abs(sv.conj() @ cov @ sv)

        return azimuths, elevations, spectrum


class MVDRBeamformer:
    """
    Minimum Variance Distortionless Response (MVDR) Beamformer.

    Also known as Capon beamformer.
    """

    def __init__(
        self,
        steering_generator: SteeringVectorGenerator,
        config: BeamformingConfig
    ):
        """
        Initialize MVDR beamformer.

        Args:
            steering_generator: Steering vector generator.
            config: Beamforming configuration.
        """
        self._steering = steering_generator
        self._config = config

    def compute_spectrum(
        self,
        cov: np.ndarray,
        frequency: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute MVDR spatial spectrum.

        Args:
            cov: Covariance matrix.
            frequency: Center frequency.

        Returns:
            Tuple of (azimuths, elevations, spectrum).
        """
        # Apply diagonal loading
        cov_reg = CovarianceEstimator.diagonal_loading(cov, self._config.diagonal_loading)

        # Inverse covariance
        try:
            cov_inv = inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = pinv(cov_reg)

        azimuths, elevations, steering_vectors = self._steering.precompute_steering_grid(
            self._config.num_azimuth,
            self._config.num_elevation,
            frequency,
            self._config.azimuth_range,
            self._config.elevation_range
        )

        spectrum = np.zeros((len(azimuths), len(elevations)))

        for i in range(len(azimuths)):
            for j in range(len(elevations)):
                sv = steering_vectors[i, j]
                denom = sv.conj() @ cov_inv @ sv
                spectrum[i, j] = 1.0 / (np.abs(denom) + 1e-12)

        return azimuths, elevations, spectrum


class MUSICBeamformer:
    """
    MUSIC (Multiple Signal Classification) Algorithm.

    High-resolution DOA estimation using subspace methods.
    """

    def __init__(
        self,
        steering_generator: SteeringVectorGenerator,
        config: BeamformingConfig
    ):
        """
        Initialize MUSIC beamformer.

        Args:
            steering_generator: Steering vector generator.
            config: Beamforming configuration.
        """
        self._steering = steering_generator
        self._config = config

    def compute_spectrum(
        self,
        cov: np.ndarray,
        frequency: float,
        num_sources: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute MUSIC spatial spectrum.

        Args:
            cov: Covariance matrix.
            frequency: Center frequency.
            num_sources: Number of sources (estimated if not provided).

        Returns:
            Tuple of (azimuths, elevations, spectrum).
        """
        n_sources = num_sources or self._config.num_sources

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace
        noise_subspace = eigenvectors[:, n_sources:]
        noise_proj = noise_subspace @ noise_subspace.conj().T

        azimuths, elevations, steering_vectors = self._steering.precompute_steering_grid(
            self._config.num_azimuth,
            self._config.num_elevation,
            frequency,
            self._config.azimuth_range,
            self._config.elevation_range
        )

        spectrum = np.zeros((len(azimuths), len(elevations)))

        for i in range(len(azimuths)):
            for j in range(len(elevations)):
                sv = steering_vectors[i, j]
                denom = sv.conj() @ noise_proj @ sv
                spectrum[i, j] = 1.0 / (np.abs(denom) + 1e-12)

        return azimuths, elevations, spectrum

    def estimate_num_sources(
        self,
        cov: np.ndarray,
        method: str = "mdl"
    ) -> int:
        """
        Estimate the number of sources.

        Args:
            cov: Covariance matrix.
            method: Estimation method ('mdl' or 'aic').

        Returns:
            Estimated number of sources.
        """
        eigenvalues, _ = eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        n = len(eigenvalues)
        n_samples = 100  # Assumed

        criterion = []
        for k in range(n - 1):
            noise_eigenvalues = eigenvalues[k + 1:]
            geometric_mean = np.exp(np.mean(np.log(noise_eigenvalues + 1e-12)))
            arithmetic_mean = np.mean(noise_eigenvalues)

            ratio = geometric_mean / (arithmetic_mean + 1e-12)

            if method == "mdl":
                # MDL criterion
                penalty = 0.5 * k * (2 * n - k) * np.log(n_samples)
            else:
                # AIC criterion
                penalty = k * (2 * n - k)

            criterion.append(-n_samples * (n - k) * np.log(ratio) + penalty)

        if criterion:
            return np.argmin(criterion)
        return 1


class ESPRITBeamformer:
    """
    ESPRIT (Estimation of Signal Parameters via Rotational Invariance) Algorithm.

    High-resolution DOA estimation for uniform arrays.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 48000,
        sound_speed: float = 343.0
    ):
        """
        Initialize ESPRIT.

        Args:
            mic_positions: Microphone positions.
            sample_rate: Sample rate in Hz.
            sound_speed: Speed of sound in m/s.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sample_rate = sample_rate
        self._sound_speed = sound_speed

    def estimate_doa(
        self,
        cov: np.ndarray,
        num_sources: int = 1,
        frequency: float = 1000.0
    ) -> List[Tuple[float, float]]:
        """
        Estimate DOA using ESPRIT.

        Args:
            cov: Covariance matrix.
            num_sources: Number of sources.
            frequency: Center frequency.

        Returns:
            List of (azimuth, elevation) tuples.
        """
        n_mics = cov.shape[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Signal subspace
        signal_subspace = eigenvectors[:, :num_sources]

        # Selection matrices for ESPRIT
        # This assumes a specific array geometry
        J1 = np.eye(n_mics - 1, n_mics)
        J2 = np.eye(n_mics - 1, n_mics, 1)

        E1 = J1 @ signal_subspace
        E2 = J2 @ signal_subspace

        # TLS-ESPRIT
        try:
            phi = pinv(E1) @ E2
            eigenvalues_phi = np.linalg.eigvals(phi)

            # Convert to angles
            wavelength = self._sound_speed / frequency
            spacing = np.linalg.norm(
                self._mic_positions[1] - self._mic_positions[0]
            )

            doas = []
            for ev in eigenvalues_phi:
                phase = np.angle(ev)
                sin_theta = phase * wavelength / (2 * np.pi * spacing)
                sin_theta = np.clip(sin_theta, -1, 1)
                theta = np.arcsin(sin_theta)
                azimuth = np.rad2deg(theta)
                doas.append((azimuth, 0.0))  # 2D estimation

            return doas

        except np.linalg.LinAlgError:
            return [(0.0, 0.0)]


class SRPPHATBeamformer:
    """
    Steered Response Power with Phase Transform (SRP-PHAT).

    Robust DOA estimation based on GCC-PHAT.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 48000,
        sound_speed: float = 343.0,
        config: Optional[BeamformingConfig] = None
    ):
        """
        Initialize SRP-PHAT.

        Args:
            mic_positions: Microphone positions.
            sample_rate: Sample rate in Hz.
            sound_speed: Speed of sound in m/s.
            config: Beamforming configuration.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sample_rate = sample_rate
        self._sound_speed = sound_speed
        self._config = config or BeamformingConfig()
        self._n_mics = len(mic_positions)

    def compute_spectrum(
        self,
        data: np.ndarray,
        n_fft: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SRP-PHAT spatial spectrum.

        Args:
            data: Multi-channel data (channels x samples).
            n_fft: FFT size.

        Returns:
            Tuple of (azimuths, elevations, spectrum).
        """
        azimuths = np.linspace(*self._config.azimuth_range, self._config.num_azimuth)
        elevations = np.linspace(*self._config.elevation_range, self._config.num_elevation)

        # Compute FFTs
        fft_data = fft(data, n=n_fft, axis=1)
        freqs = np.fft.fftfreq(n_fft, 1 / self._sample_rate)

        # Frequency mask
        freq_mask = (np.abs(freqs) >= self._config.frequency_range[0]) & \
                    (np.abs(freqs) <= self._config.frequency_range[1])

        spectrum = np.zeros((len(azimuths), len(elevations)))

        for i, az in enumerate(azimuths):
            for j, el in enumerate(elevations):
                power = self._compute_srp_point(
                    fft_data, freqs, freq_mask, az, el, n_fft
                )
                spectrum[i, j] = power

        return azimuths, elevations, spectrum

    def _compute_srp_point(
        self,
        fft_data: np.ndarray,
        freqs: np.ndarray,
        freq_mask: np.ndarray,
        azimuth: float,
        elevation: float,
        n_fft: int
    ) -> float:
        """Compute SRP for a single point."""
        # Direction vector
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        direction = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        # Time delays
        delays = self._mic_positions @ direction / self._sound_speed

        power = 0.0

        # Sum over microphone pairs
        for m1 in range(self._n_mics):
            for m2 in range(m1 + 1, self._n_mics):
                tau = delays[m1] - delays[m2]

                # GCC-PHAT for this pair
                cross_spectrum = fft_data[m1] * np.conj(fft_data[m2])
                magnitude = np.abs(cross_spectrum) + 1e-12
                phat_spectrum = cross_spectrum / magnitude

                # Phase shift for steering
                phase_shift = np.exp(2j * np.pi * freqs * tau)

                # Steered response
                steered = phat_spectrum * phase_shift
                steered = steered[freq_mask]

                power += np.abs(np.sum(steered))

        return power


class BeamformingEngine:
    """
    Unified beamforming engine.

    Provides a unified interface for all beamforming algorithms.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 48000,
        config: Optional[BeamformingConfig] = None
    ):
        """
        Initialize beamforming engine.

        Args:
            mic_positions: Microphone positions (N x 3).
            sample_rate: Sample rate in Hz.
            config: Beamforming configuration.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sample_rate = sample_rate
        self._config = config or BeamformingConfig()

        self._steering = SteeringVectorGenerator(
            mic_positions, sample_rate
        )

        self._das = DelayAndSumBeamformer(self._steering, self._config)
        self._mvdr = MVDRBeamformer(self._steering, self._config)
        self._music = MUSICBeamformer(self._steering, self._config)
        self._esprit = ESPRITBeamformer(mic_positions, sample_rate)
        self._srp_phat = SRPPHATBeamformer(mic_positions, sample_rate, config=self._config)

        self._lock = threading.Lock()

    def estimate_doa(
        self,
        data: np.ndarray,
        method: Optional[str] = None,
        frequency: float = 1000.0
    ) -> DOAResult:
        """
        Estimate Direction of Arrival.

        Args:
            data: Multi-channel data (channels x samples).
            method: Beamforming method (uses config default if not specified).
            frequency: Center frequency for analysis.

        Returns:
            DOAResult with estimated direction.
        """
        method = method or self._config.method

        if method == "srp_phat":
            azimuths, elevations, spectrum = self._srp_phat.compute_spectrum(data)
        else:
            # Compute covariance
            cov = CovarianceEstimator.frequency_domain_covariance(
                data, self._sample_rate, self._config.frequency_range
            )

            if method == "das":
                azimuths, elevations, spectrum = self._das.compute_spectrum(cov, frequency)
            elif method == "mvdr":
                azimuths, elevations, spectrum = self._mvdr.compute_spectrum(cov, frequency)
            elif method == "music":
                azimuths, elevations, spectrum = self._music.compute_spectrum(cov, frequency)
            else:
                azimuths, elevations, spectrum = self._music.compute_spectrum(cov, frequency)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        azimuth = azimuths[peak_idx[0]]
        elevation = elevations[peak_idx[1]]
        power = spectrum[peak_idx]

        return DOAResult(
            azimuth=azimuth,
            elevation=elevation,
            power=power,
            spectrum=spectrum
        )

    def estimate_multiple_sources(
        self,
        data: np.ndarray,
        num_sources: int = 2,
        frequency: float = 1000.0
    ) -> List[DOAResult]:
        """
        Estimate multiple source directions.

        Args:
            data: Multi-channel data.
            num_sources: Number of sources to estimate.
            frequency: Center frequency.

        Returns:
            List of DOAResult objects.
        """
        cov = CovarianceEstimator.frequency_domain_covariance(
            data, self._sample_rate, self._config.frequency_range
        )

        azimuths, elevations, spectrum = self._music.compute_spectrum(
            cov, frequency, num_sources
        )

        # Find multiple peaks
        results = []

        for _ in range(num_sources):
            peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            azimuth = azimuths[peak_idx[0]]
            elevation = elevations[peak_idx[1]]
            power = spectrum[peak_idx]

            results.append(DOAResult(
                azimuth=azimuth,
                elevation=elevation,
                power=power
            ))

            # Suppress peak for next iteration
            spectrum[peak_idx[0], peak_idx[1]] = 0

        return results

    def update_config(self, config: BeamformingConfig) -> None:
        """Update beamforming configuration."""
        with self._lock:
            self._config = config
            self._das = DelayAndSumBeamformer(self._steering, config)
            self._mvdr = MVDRBeamformer(self._steering, config)
            self._music = MUSICBeamformer(self._steering, config)
            self._srp_phat = SRPPHATBeamformer(
                self._mic_positions, self._sample_rate, config=config
            )
