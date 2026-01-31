"""
3D Localization Module

Provides 3D position estimation from TDOA and DOA measurements:
- Spherical to Cartesian conversion
- Multilateration algorithms
- Least squares optimization
- Distance estimation
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class Position3D:
    """3D position in Cartesian coordinates."""
    x: float  # meters
    y: float  # meters
    z: float  # meters

    def to_spherical(self) -> Tuple[float, float, float]:
        """Convert to spherical coordinates (azimuth, elevation, distance)."""
        distance = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        azimuth = np.rad2deg(np.arctan2(self.y, self.x))
        if azimuth < 0:
            azimuth += 360
        elevation = np.rad2deg(np.arcsin(self.z / (distance + 1e-12)))
        return azimuth, elevation, distance

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_spherical(cls, azimuth: float, elevation: float, distance: float) -> 'Position3D':
        """Create from spherical coordinates."""
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)

        return cls(x=x, y=y, z=z)


@dataclass
class LocalizationResult:
    """Result of localization computation."""
    position: Position3D
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float  # meters
    confidence: float  # 0-1
    residual: float  # fitting residual
    method: str


@dataclass
class LocalizationConfig:
    """Localization configuration."""
    method: str = "spherical_intersection"  # spherical_intersection, multilateration, hybrid
    sound_speed: float = 343.0
    max_distance: float = 500.0
    min_distance: float = 1.0
    optimization_method: str = "trf"  # trf, lm, dogbox
    max_iterations: int = 100


class SphericalIntersection:
    """
    Spherical Intersection method for 3D localization.

    Uses DOA (azimuth, elevation) from multiple sensors to find
    the intersection point.
    """

    def __init__(self, sound_speed: float = 343.0):
        """
        Initialize spherical intersection solver.

        Args:
            sound_speed: Speed of sound in m/s.
        """
        self._sound_speed = sound_speed

    def localize(
        self,
        azimuth: float,
        elevation: float,
        distance_estimate: float,
        sensor_position: np.ndarray = None
    ) -> Position3D:
        """
        Convert DOA to 3D position.

        Args:
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.
            distance_estimate: Estimated distance in meters.
            sensor_position: Sensor position (default origin).

        Returns:
            Position3D object.
        """
        sensor_pos = sensor_position if sensor_position is not None else np.zeros(3)

        pos = Position3D.from_spherical(azimuth, elevation, distance_estimate)

        # Offset by sensor position
        return Position3D(
            x=pos.x + sensor_pos[0],
            y=pos.y + sensor_pos[1],
            z=pos.z + sensor_pos[2]
        )


class MultilaterationSolver:
    """
    Multilateration solver for 3D localization from TDOAs.

    Uses time difference of arrival measurements between
    multiple microphone pairs to estimate source position.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sound_speed: float = 343.0,
        config: Optional[LocalizationConfig] = None
    ):
        """
        Initialize multilateration solver.

        Args:
            mic_positions: Microphone positions (N x 3) in meters.
            sound_speed: Speed of sound in m/s.
            config: Localization configuration.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sound_speed = sound_speed
        self._config = config or LocalizationConfig()
        self._n_mics = len(mic_positions)

    def localize(
        self,
        tdoas: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> LocalizationResult:
        """
        Estimate source position from TDOAs.

        Args:
            tdoas: TDOA values in seconds (relative to reference mic).
            initial_guess: Initial position estimate.

        Returns:
            LocalizationResult object.
        """
        # Convert TDOAs to range differences
        range_diffs = tdoas * self._sound_speed

        # Initial guess
        if initial_guess is None:
            initial_guess = self._estimate_initial_position(range_diffs)

        # Optimization
        result = least_squares(
            self._residual_function,
            initial_guess,
            args=(range_diffs,),
            method=self._config.optimization_method,
            max_nfev=self._config.max_iterations,
            bounds=(
                [-self._config.max_distance] * 3,
                [self._config.max_distance] * 3
            )
        )

        position = Position3D(x=result.x[0], y=result.x[1], z=result.x[2])
        azimuth, elevation, distance = position.to_spherical()

        # Compute confidence based on residual
        residual = np.sqrt(np.mean(result.fun ** 2))
        confidence = np.exp(-residual / 10)  # Decay with residual

        return LocalizationResult(
            position=position,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            confidence=confidence,
            residual=residual,
            method="multilateration"
        )

    def _residual_function(
        self,
        position: np.ndarray,
        range_diffs: np.ndarray
    ) -> np.ndarray:
        """
        Compute residuals for optimization.

        Args:
            position: Estimated position [x, y, z].
            range_diffs: Measured range differences.

        Returns:
            Array of residuals.
        """
        # Compute distances from position to each microphone
        distances = np.linalg.norm(self._mic_positions - position, axis=1)

        # Compute expected range differences (relative to mic 0)
        expected_diffs = distances - distances[0]

        # Residuals
        residuals = expected_diffs[1:] - range_diffs[1:]

        return residuals

    def _estimate_initial_position(self, range_diffs: np.ndarray) -> np.ndarray:
        """
        Estimate initial position using closed-form solution.

        Uses the spherical intersection method for initial estimate.
        """
        # Use first three microphone pairs for initial estimate
        # This is a simplified approach

        # Place initial guess in the direction of minimum range difference
        ref_pos = self._mic_positions[0]

        # Find the microphone with smallest range difference
        min_idx = np.argmin(np.abs(range_diffs[1:])) + 1
        direction = self._mic_positions[min_idx] - ref_pos
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        # Initial distance estimate based on array size
        array_size = np.max(np.linalg.norm(
            self._mic_positions - self._mic_positions.mean(axis=0), axis=1
        ))
        initial_distance = array_size * 10  # Start at 10x array size

        return ref_pos + direction * initial_distance

    def localize_linear(self, tdoas: np.ndarray) -> LocalizationResult:
        """
        Linear least squares localization (Chan algorithm).

        Faster but may be less accurate than nonlinear optimization.

        Args:
            tdoas: TDOA values in seconds.

        Returns:
            LocalizationResult object.
        """
        range_diffs = tdoas * self._sound_speed

        n = self._n_mics - 1
        A = np.zeros((n, 4))
        b = np.zeros(n)

        ref_mic = self._mic_positions[0]

        for i in range(n):
            mic_i = self._mic_positions[i + 1]
            rd = range_diffs[i + 1]

            A[i, :3] = 2 * (mic_i - ref_mic)
            A[i, 3] = -2 * rd

            b[i] = (np.sum(mic_i**2) - np.sum(ref_mic**2) - rd**2)

        # Solve linear system
        try:
            x, residuals, rank, s = lstsq(A, b)
            position = Position3D(x=x[0], y=x[1], z=x[2])

            azimuth, elevation, distance = position.to_spherical()

            residual = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
            confidence = np.exp(-residual / 10)

            return LocalizationResult(
                position=position,
                azimuth=azimuth,
                elevation=elevation,
                distance=distance,
                confidence=confidence,
                residual=residual,
                method="linear_multilateration"
            )

        except np.linalg.LinAlgError:
            # Fall back to default position
            return LocalizationResult(
                position=Position3D(0, 0, 0),
                azimuth=0,
                elevation=0,
                distance=0,
                confidence=0,
                residual=float('inf'),
                method="linear_multilateration"
            )


class DistanceEstimator:
    """
    Distance estimation from acoustic intensity.

    Estimates distance based on signal strength assuming
    free-field propagation.
    """

    def __init__(
        self,
        reference_spl: float = 80.0,
        reference_distance: float = 1.0
    ):
        """
        Initialize distance estimator.

        Args:
            reference_spl: Reference SPL at reference distance (dB).
            reference_distance: Reference distance in meters.
        """
        self._reference_spl = reference_spl
        self._reference_distance = reference_distance

    def estimate(
        self,
        signal: np.ndarray,
        calibration_factor: float = 94.0
    ) -> Tuple[float, float]:
        """
        Estimate distance from signal amplitude.

        Args:
            signal: Audio signal.
            calibration_factor: Microphone calibration factor (dB).

        Returns:
            Tuple of (estimated_distance, confidence).
        """
        # RMS level
        rms = np.sqrt(np.mean(signal ** 2))

        # Convert to SPL
        measured_spl = 20 * np.log10(rms + 1e-12) + calibration_factor

        # Inverse square law
        # SPL = SPL_ref - 20*log10(d/d_ref)
        # d = d_ref * 10^((SPL_ref - SPL) / 20)
        spl_diff = self._reference_spl - measured_spl
        distance = self._reference_distance * (10 ** (spl_diff / 20))

        # Confidence based on signal level
        confidence = min(1.0, rms * 10)  # Higher signal = higher confidence

        return distance, confidence

    def estimate_from_rms(self, rms_level: float) -> float:
        """
        Estimate distance from RMS level directly.

        Args:
            rms_level: RMS signal level (0-1).

        Returns:
            Estimated distance in meters.
        """
        # Simple inverse relationship
        if rms_level > 0:
            distance = self._reference_distance / rms_level
        else:
            distance = float('inf')

        return min(distance, 1000.0)  # Cap at 1km


class HybridLocalizer:
    """
    Hybrid localization combining DOA and TDOA.

    Uses multiple methods and combines results for improved accuracy.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sound_speed: float = 343.0,
        config: Optional[LocalizationConfig] = None
    ):
        """
        Initialize hybrid localizer.

        Args:
            mic_positions: Microphone positions.
            sound_speed: Speed of sound in m/s.
            config: Localization configuration.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sound_speed = sound_speed
        self._config = config or LocalizationConfig()

        self._spherical = SphericalIntersection(sound_speed)
        self._multilateration = MultilaterationSolver(
            mic_positions, sound_speed, config
        )
        self._distance_estimator = DistanceEstimator()

    def localize(
        self,
        azimuth: float,
        elevation: float,
        tdoas: Optional[np.ndarray] = None,
        signal_rms: Optional[float] = None
    ) -> LocalizationResult:
        """
        Perform hybrid localization.

        Args:
            azimuth: DOA azimuth in degrees.
            elevation: DOA elevation in degrees.
            tdoas: TDOA values (optional).
            signal_rms: Signal RMS level (optional).

        Returns:
            LocalizationResult object.
        """
        results = []
        weights = []

        # Method 1: Spherical with distance from amplitude
        if signal_rms is not None:
            distance_amplitude = self._distance_estimator.estimate_from_rms(signal_rms)
            pos_amplitude = Position3D.from_spherical(azimuth, elevation, distance_amplitude)
            results.append(pos_amplitude)
            weights.append(0.3)

        # Method 2: Multilateration if TDOAs available
        if tdoas is not None and len(tdoas) >= 3:
            mlat_result = self._multilateration.localize(tdoas)
            results.append(mlat_result.position)
            weights.append(mlat_result.confidence)

        # Combine results
        if not results:
            # Default to DOA-only with estimated distance
            distance = 50.0  # Default distance
            position = Position3D.from_spherical(azimuth, elevation, distance)
            return LocalizationResult(
                position=position,
                azimuth=azimuth,
                elevation=elevation,
                distance=distance,
                confidence=0.3,
                residual=0.0,
                method="spherical_default"
            )

        # Weighted average of positions
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        combined_pos = np.zeros(3)
        for pos, weight in zip(results, weights):
            combined_pos += weight * pos.to_array()

        position = Position3D(x=combined_pos[0], y=combined_pos[1], z=combined_pos[2])
        az, el, dist = position.to_spherical()

        # Overall confidence
        confidence = np.mean(weights)

        return LocalizationResult(
            position=position,
            azimuth=az,
            elevation=el,
            distance=dist,
            confidence=confidence,
            residual=0.0,
            method="hybrid"
        )


class LocalizationEngine:
    """
    Main localization engine.

    Provides unified interface for all localization methods.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        sound_speed: float = 343.0,
        config: Optional[LocalizationConfig] = None
    ):
        """
        Initialize localization engine.

        Args:
            mic_positions: Microphone positions.
            sound_speed: Speed of sound in m/s.
            config: Localization configuration.
        """
        self._mic_positions = np.asarray(mic_positions)
        self._sound_speed = sound_speed
        self._config = config or LocalizationConfig()

        self._multilateration = MultilaterationSolver(
            mic_positions, sound_speed, config
        )
        self._hybrid = HybridLocalizer(mic_positions, sound_speed, config)
        self._distance_estimator = DistanceEstimator()

        self._lock = threading.Lock()

    def localize_from_doa(
        self,
        azimuth: float,
        elevation: float,
        signal_rms: Optional[float] = None
    ) -> LocalizationResult:
        """
        Localize from DOA only.

        Args:
            azimuth: Azimuth in degrees.
            elevation: Elevation in degrees.
            signal_rms: Signal RMS level.

        Returns:
            LocalizationResult object.
        """
        if signal_rms is not None:
            distance = self._distance_estimator.estimate_from_rms(signal_rms)
            distance = np.clip(distance, self._config.min_distance, self._config.max_distance)
        else:
            distance = 100.0  # Default

        position = Position3D.from_spherical(azimuth, elevation, distance)

        return LocalizationResult(
            position=position,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            confidence=0.5 if signal_rms else 0.3,
            residual=0.0,
            method="doa_only"
        )

    def localize_from_tdoa(
        self,
        tdoas: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> LocalizationResult:
        """
        Localize from TDOAs.

        Args:
            tdoas: TDOA values in seconds.
            initial_guess: Initial position estimate.

        Returns:
            LocalizationResult object.
        """
        return self._multilateration.localize(tdoas, initial_guess)

    def localize_hybrid(
        self,
        azimuth: float,
        elevation: float,
        tdoas: Optional[np.ndarray] = None,
        signal_rms: Optional[float] = None
    ) -> LocalizationResult:
        """
        Perform hybrid localization.

        Args:
            azimuth: DOA azimuth.
            elevation: DOA elevation.
            tdoas: TDOA values.
            signal_rms: Signal RMS level.

        Returns:
            LocalizationResult object.
        """
        return self._hybrid.localize(azimuth, elevation, tdoas, signal_rms)

    def localize(
        self,
        azimuth: float,
        elevation: float,
        tdoas: Optional[np.ndarray] = None,
        signal_rms: Optional[float] = None
    ) -> LocalizationResult:
        """
        Main localization method using configured approach.

        Args:
            azimuth: DOA azimuth.
            elevation: DOA elevation.
            tdoas: TDOA values.
            signal_rms: Signal RMS level.

        Returns:
            LocalizationResult object.
        """
        method = self._config.method

        if method == "multilateration" and tdoas is not None:
            return self.localize_from_tdoa(tdoas)
        elif method == "spherical_intersection":
            return self.localize_from_doa(azimuth, elevation, signal_rms)
        else:
            return self.localize_hybrid(azimuth, elevation, tdoas, signal_rms)

    def update_config(self, config: LocalizationConfig) -> None:
        """Update localization configuration."""
        with self._lock:
            self._config = config
            self._multilateration = MultilaterationSolver(
                self._mic_positions, self._sound_speed, config
            )
            self._hybrid = HybridLocalizer(
                self._mic_positions, self._sound_speed, config
            )
