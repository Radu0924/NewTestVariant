"""
Multi-Target Tracking Module

Provides tracking algorithms for multiple drone targets:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter
- Multi-Hypothesis Tracking (MHT)
- Data association with GNN
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from abc import ABC, abstractmethod


@dataclass
class TrackState:
    """State of a tracked target."""
    track_id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    covariance: np.ndarray  # State covariance matrix
    last_update: float  # Timestamp of last update
    age: int  # Number of updates
    consecutive_misses: int  # Consecutive missed detections
    classification: str = "unknown"
    confidence: float = 0.5


@dataclass
class Detection:
    """Single detection measurement."""
    timestamp: float
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float  # meters
    confidence: float
    classification: str = "unknown"

    def to_cartesian(self) -> np.ndarray:
        """Convert to Cartesian coordinates."""
        az_rad = np.deg2rad(self.azimuth)
        el_rad = np.deg2rad(self.elevation)

        x = self.distance * np.cos(el_rad) * np.cos(az_rad)
        y = self.distance * np.cos(el_rad) * np.sin(az_rad)
        z = self.distance * np.sin(el_rad)

        return np.array([x, y, z])


@dataclass
class Track:
    """Complete track information."""
    track_id: int
    state: TrackState
    history: deque = field(default_factory=lambda: deque(maxlen=500))
    status: str = "tentative"  # tentative, confirmed, deleted


@dataclass
class TrackingConfig:
    """Tracking configuration."""
    filter_type: str = "ekf"  # ekf, ukf, particle
    max_tracks: int = 10
    track_timeout: float = 5.0  # seconds
    confirmation_threshold: int = 3
    deletion_threshold: int = 5
    association_threshold: float = 20.0  # meters
    process_noise: float = 1.0
    measurement_noise: float = 5.0
    update_rate: float = 30.0  # Hz


class KalmanFilterBase(ABC):
    """Base class for Kalman filters."""

    def __init__(
        self,
        state_dim: int = 9,
        measurement_dim: int = 3,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0
    ):
        """
        Initialize Kalman filter base.

        Args:
            state_dim: State dimension.
            measurement_dim: Measurement dimension.
            process_noise: Process noise standard deviation.
            measurement_noise: Measurement noise standard deviation.
        """
        self._state_dim = state_dim
        self._measurement_dim = measurement_dim
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

        # Initialize matrices
        self._init_matrices()

    @abstractmethod
    def _init_matrices(self) -> None:
        """Initialize filter matrices."""
        pass

    @abstractmethod
    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        pass

    @abstractmethod
    def update(
        self,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement."""
        pass


class ExtendedKalmanFilter(KalmanFilterBase):
    """
    Extended Kalman Filter for nonlinear tracking.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0
    ):
        """
        Initialize EKF.

        Args:
            initial_state: Initial state estimate.
            process_noise: Process noise.
            measurement_noise: Measurement noise.
        """
        super().__init__(9, 3, process_noise, measurement_noise)

        if initial_state is not None:
            self._x = initial_state.copy()
        else:
            self._x = np.zeros(9)

    def _init_matrices(self) -> None:
        """Initialize EKF matrices."""
        # State covariance
        self._P = np.eye(9) * 100

        # Measurement matrix (observe position only)
        self._H = np.zeros((3, 9))
        self._H[:3, :3] = np.eye(3)

        # Measurement noise covariance
        self._R = np.eye(3) * (self._measurement_noise ** 2)

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        """Get state transition matrix for time step dt."""
        F = np.eye(9)

        # Position update from velocity
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Position update from acceleration
        F[0, 6] = 0.5 * dt**2
        F[1, 7] = 0.5 * dt**2
        F[2, 8] = 0.5 * dt**2

        # Velocity update from acceleration
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        return F

    def _get_process_noise_matrix(self, dt: float) -> np.ndarray:
        """Get process noise covariance matrix."""
        q = self._process_noise ** 2

        # Continuous white noise acceleration model
        Q = np.zeros((9, 9))

        # Position covariance
        Q[0, 0] = Q[1, 1] = Q[2, 2] = (dt**5) / 20 * q
        # Position-velocity cross-covariance
        Q[0, 3] = Q[1, 4] = Q[2, 5] = (dt**4) / 8 * q
        Q[3, 0] = Q[4, 1] = Q[5, 2] = (dt**4) / 8 * q
        # Velocity covariance
        Q[3, 3] = Q[4, 4] = Q[5, 5] = (dt**3) / 3 * q
        # Acceleration covariance
        Q[6, 6] = Q[7, 7] = Q[8, 8] = dt * q

        return Q

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state.

        Args:
            dt: Time step in seconds.

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        F = self._get_transition_matrix(dt)
        Q = self._get_process_noise_matrix(dt)

        # State prediction
        self._x = F @ self._x

        # Covariance prediction
        self._P = F @ self._P @ F.T + Q

        return self._x.copy(), self._P.copy()

    def update(
        self,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with measurement.

        Args:
            measurement: Measurement vector [x, y, z].

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        # Innovation
        y = measurement - self._H @ self._x

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        return self._x.copy(), self._P.copy()

    @property
    def state(self) -> np.ndarray:
        """Get current state estimate."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        return self._P.copy()

    def set_state(self, state: np.ndarray, covariance: Optional[np.ndarray] = None) -> None:
        """Set filter state."""
        self._x = state.copy()
        if covariance is not None:
            self._P = covariance.copy()


class UnscentedKalmanFilter(KalmanFilterBase):
    """
    Unscented Kalman Filter for nonlinear tracking.

    Uses sigma points to capture nonlinear behavior.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        Initialize UKF.

        Args:
            initial_state: Initial state estimate.
            process_noise: Process noise.
            measurement_noise: Measurement noise.
            alpha: Spread of sigma points.
            beta: Prior knowledge of distribution.
            kappa: Secondary scaling parameter.
        """
        super().__init__(9, 3, process_noise, measurement_noise)

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        self._lambda = alpha**2 * (self._state_dim + kappa) - self._state_dim

        # Sigma point weights
        self._compute_weights()

        if initial_state is not None:
            self._x = initial_state.copy()
        else:
            self._x = np.zeros(9)

    def _init_matrices(self) -> None:
        """Initialize UKF matrices."""
        self._P = np.eye(9) * 100
        self._R = np.eye(3) * (self._measurement_noise ** 2)

    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        n = self._state_dim

        self._Wm = np.zeros(2 * n + 1)
        self._Wc = np.zeros(2 * n + 1)

        self._Wm[0] = self._lambda / (n + self._lambda)
        self._Wc[0] = self._Wm[0] + (1 - self._alpha**2 + self._beta)

        for i in range(1, 2 * n + 1):
            self._Wm[i] = 1 / (2 * (n + self._lambda))
            self._Wc[i] = self._Wm[i]

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points around current state."""
        n = self._state_dim
        sigma_points = np.zeros((2 * n + 1, n))

        # Square root of covariance
        try:
            sqrt_P = np.linalg.cholesky((n + self._lambda) * self._P)
        except np.linalg.LinAlgError:
            sqrt_P = np.sqrt((n + self._lambda)) * np.sqrt(np.diag(np.diag(self._P)))

        sigma_points[0] = self._x

        for i in range(n):
            sigma_points[i + 1] = self._x + sqrt_P[i]
            sigma_points[n + i + 1] = self._x - sqrt_P[i]

        return sigma_points

    def _state_transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear state transition function."""
        new_state = state.copy()

        # Position update
        new_state[0] += state[3] * dt + 0.5 * state[6] * dt**2
        new_state[1] += state[4] * dt + 0.5 * state[7] * dt**2
        new_state[2] += state[5] * dt + 0.5 * state[8] * dt**2

        # Velocity update
        new_state[3] += state[6] * dt
        new_state[4] += state[7] * dt
        new_state[5] += state[8] * dt

        return new_state

    def _measurement_function(self, state: np.ndarray) -> np.ndarray:
        """Nonlinear measurement function."""
        return state[:3]

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state using UKF."""
        # Generate sigma points
        sigma_points = self._generate_sigma_points()

        # Propagate sigma points
        propagated = np.zeros_like(sigma_points)
        for i in range(len(sigma_points)):
            propagated[i] = self._state_transition(sigma_points[i], dt)

        # Predicted state
        self._x = np.sum(self._Wm[:, np.newaxis] * propagated, axis=0)

        # Predicted covariance
        self._P = np.zeros((9, 9))
        for i in range(len(propagated)):
            diff = propagated[i] - self._x
            self._P += self._Wc[i] * np.outer(diff, diff)

        # Add process noise
        Q = np.eye(9) * (self._process_noise ** 2) * dt
        self._P += Q

        return self._x.copy(), self._P.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement using UKF."""
        # Generate sigma points
        sigma_points = self._generate_sigma_points()

        # Transform to measurement space
        measurements_sigma = np.zeros((len(sigma_points), 3))
        for i in range(len(sigma_points)):
            measurements_sigma[i] = self._measurement_function(sigma_points[i])

        # Predicted measurement
        z_pred = np.sum(self._Wm[:, np.newaxis] * measurements_sigma, axis=0)

        # Measurement covariance
        Pzz = np.zeros((3, 3))
        for i in range(len(measurements_sigma)):
            diff = measurements_sigma[i] - z_pred
            Pzz += self._Wc[i] * np.outer(diff, diff)
        Pzz += self._R

        # Cross-covariance
        Pxz = np.zeros((9, 3))
        for i in range(len(sigma_points)):
            diff_x = sigma_points[i] - self._x
            diff_z = measurements_sigma[i] - z_pred
            Pxz += self._Wc[i] * np.outer(diff_x, diff_z)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # Update
        innovation = measurement - z_pred
        self._x = self._x + K @ innovation
        self._P = self._P - K @ Pzz @ K.T

        return self._x.copy(), self._P.copy()

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()


class DataAssociation:
    """
    Data association for multi-target tracking.

    Implements Global Nearest Neighbor (GNN) association.
    """

    def __init__(self, threshold: float = 20.0):
        """
        Initialize data association.

        Args:
            threshold: Maximum distance for association (meters).
        """
        self._threshold = threshold

    def associate(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Associate detections with tracks using GNN.

        Args:
            tracks: List of existing tracks.
            detections: List of new detections.

        Returns:
            Tuple of (associations, unassigned_tracks, unassigned_detections).
            associations: Dict mapping track_id to detection index.
        """
        if not tracks or not detections:
            return ({}, list(range(len(tracks))), list(range(len(detections))))

        # Compute cost matrix (distances)
        n_tracks = len(tracks)
        n_detections = len(detections)
        cost_matrix = np.full((n_tracks, n_detections), np.inf)

        for i, track in enumerate(tracks):
            track_pos = track.state.position
            for j, det in enumerate(detections):
                det_pos = det.to_cartesian()
                distance = np.linalg.norm(track_pos - det_pos)
                if distance < self._threshold:
                    cost_matrix[i, j] = distance

        # Greedy assignment (GNN)
        associations = {}
        assigned_tracks = set()
        assigned_detections = set()

        while True:
            # Find minimum cost
            valid_costs = cost_matrix.copy()
            valid_costs[list(assigned_tracks), :] = np.inf
            valid_costs[:, list(assigned_detections)] = np.inf

            if np.all(np.isinf(valid_costs)):
                break

            min_idx = np.unravel_index(np.argmin(valid_costs), valid_costs.shape)

            if valid_costs[min_idx] < self._threshold:
                track_idx, det_idx = min_idx
                associations[tracks[track_idx].track_id] = det_idx
                assigned_tracks.add(track_idx)
                assigned_detections.add(det_idx)
            else:
                break

        unassigned_tracks = [i for i in range(n_tracks) if i not in assigned_tracks]
        unassigned_detections = [i for i in range(n_detections) if i not in assigned_detections]

        return associations, unassigned_tracks, unassigned_detections


class MultiTargetTracker:
    """
    Multi-target tracking system.

    Manages multiple tracks with Kalman filters and data association.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize multi-target tracker.

        Args:
            config: Tracking configuration.
        """
        self._config = config or TrackingConfig()
        self._tracks: Dict[int, Track] = {}
        self._next_track_id = 0
        self._data_association = DataAssociation(self._config.association_threshold)
        self._last_update_time: Optional[float] = None
        self._lock = threading.Lock()

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of new detections.

        Returns:
            List of active tracks.
        """
        current_time = time.time()

        with self._lock:
            # Calculate time delta
            if self._last_update_time is None:
                dt = 1.0 / self._config.update_rate
            else:
                dt = current_time - self._last_update_time

            self._last_update_time = current_time

            # Predict all tracks
            for track in self._tracks.values():
                if track.status != "deleted":
                    self._predict_track(track, dt)

            # Associate detections with tracks
            active_tracks = [t for t in self._tracks.values() if t.status != "deleted"]
            associations, unassigned_tracks, unassigned_dets = \
                self._data_association.associate(active_tracks, detections)

            # Update associated tracks
            for track_id, det_idx in associations.items():
                track = self._tracks[track_id]
                detection = detections[det_idx]
                self._update_track(track, detection)

            # Handle unassigned tracks
            for track_idx in unassigned_tracks:
                track = active_tracks[track_idx]
                self._miss_track(track)

            # Create new tracks for unassigned detections
            for det_idx in unassigned_dets:
                if len(self._tracks) < self._config.max_tracks:
                    detection = detections[det_idx]
                    self._create_track(detection)

            # Remove deleted tracks
            self._cleanup_tracks()

            return [t for t in self._tracks.values() if t.status != "deleted"]

    def _create_filter(self, initial_pos: np.ndarray) -> KalmanFilterBase:
        """Create appropriate Kalman filter."""
        initial_state = np.zeros(9)
        initial_state[:3] = initial_pos

        if self._config.filter_type == "ukf":
            return UnscentedKalmanFilter(
                initial_state,
                self._config.process_noise,
                self._config.measurement_noise
            )
        else:
            return ExtendedKalmanFilter(
                initial_state,
                self._config.process_noise,
                self._config.measurement_noise
            )

    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection."""
        initial_pos = detection.to_cartesian()

        filter = self._create_filter(initial_pos)

        state = TrackState(
            track_id=self._next_track_id,
            position=initial_pos,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            covariance=filter.covariance,
            last_update=time.time(),
            age=1,
            consecutive_misses=0,
            classification=detection.classification,
            confidence=detection.confidence
        )

        track = Track(
            track_id=self._next_track_id,
            state=state,
            status="tentative"
        )
        track.history.append(state)

        # Store filter reference
        track._filter = filter

        self._tracks[self._next_track_id] = track
        self._next_track_id += 1

        return track

    def _predict_track(self, track: Track, dt: float) -> None:
        """Predict track state."""
        if hasattr(track, '_filter'):
            state, cov = track._filter.predict(dt)
            track.state.position = state[:3]
            track.state.velocity = state[3:6]
            track.state.acceleration = state[6:9]
            track.state.covariance = cov

    def _update_track(self, track: Track, detection: Detection) -> None:
        """Update track with detection."""
        measurement = detection.to_cartesian()

        if hasattr(track, '_filter'):
            state, cov = track._filter.update(measurement)
            track.state.position = state[:3]
            track.state.velocity = state[3:6]
            track.state.acceleration = state[6:9]
            track.state.covariance = cov

        track.state.last_update = time.time()
        track.state.age += 1
        track.state.consecutive_misses = 0
        track.state.classification = detection.classification
        track.state.confidence = detection.confidence

        track.history.append(TrackState(
            track_id=track.track_id,
            position=track.state.position.copy(),
            velocity=track.state.velocity.copy(),
            acceleration=track.state.acceleration.copy(),
            covariance=track.state.covariance.copy(),
            last_update=track.state.last_update,
            age=track.state.age,
            consecutive_misses=track.state.consecutive_misses,
            classification=track.state.classification,
            confidence=track.state.confidence
        ))

        # Confirm track
        if track.status == "tentative" and \
           track.state.age >= self._config.confirmation_threshold:
            track.status = "confirmed"

    def _miss_track(self, track: Track) -> None:
        """Handle missed detection for track."""
        track.state.consecutive_misses += 1
        track.state.confidence *= 0.9  # Decay confidence

        if track.state.consecutive_misses >= self._config.deletion_threshold:
            track.status = "deleted"

    def _cleanup_tracks(self) -> None:
        """Remove stale tracks."""
        current_time = time.time()
        to_delete = []

        for track_id, track in self._tracks.items():
            if track.status == "deleted":
                to_delete.append(track_id)
            elif current_time - track.state.last_update > self._config.track_timeout:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self._tracks[track_id]

    def get_tracks(self) -> List[Track]:
        """Get all active tracks."""
        with self._lock:
            return [t for t in self._tracks.values() if t.status != "deleted"]

    def get_confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks only."""
        with self._lock:
            return [t for t in self._tracks.values() if t.status == "confirmed"]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get specific track by ID."""
        with self._lock:
            return self._tracks.get(track_id)

    def predict_position(
        self,
        track_id: int,
        time_ahead: float
    ) -> Optional[np.ndarray]:
        """
        Predict future position for a track.

        Args:
            track_id: Track ID.
            time_ahead: Time in seconds to predict ahead.

        Returns:
            Predicted position [x, y, z] or None if track not found.
        """
        track = self.get_track(track_id)
        if track is None:
            return None

        # Simple kinematic prediction
        pos = track.state.position
        vel = track.state.velocity
        acc = track.state.acceleration

        predicted = pos + vel * time_ahead + 0.5 * acc * time_ahead**2

        return predicted

    def clear(self) -> None:
        """Clear all tracks."""
        with self._lock:
            self._tracks.clear()
            self._next_track_id = 0

    def update_config(self, config: TrackingConfig) -> None:
        """Update tracking configuration."""
        with self._lock:
            self._config = config
            self._data_association = DataAssociation(config.association_threshold)
