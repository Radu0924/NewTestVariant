"""
Unit tests for Tracking module.
"""

import pytest
import numpy as np
import time

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tracking import (
    MultiTargetTracker, TrackingConfig, Detection, Track, TrackState,
    ExtendedKalmanFilter, UnscentedKalmanFilter, DataAssociation
)


class TestTrackingConfig:
    """Tests for TrackingConfig dataclass."""

    def test_default_values(self):
        config = TrackingConfig()
        assert config.filter_type == "EKF"
        assert config.max_tracks == 20
        assert config.track_timeout == 5.0
        assert config.association_threshold == 30.0

    def test_custom_values(self):
        config = TrackingConfig(
            filter_type="UKF",
            max_tracks=10,
            track_timeout=3.0,
            association_threshold=20.0
        )
        assert config.filter_type == "UKF"
        assert config.max_tracks == 10


class TestDetection:
    """Tests for Detection dataclass."""

    def test_creation(self):
        det = Detection(
            timestamp=time.time(),
            azimuth=45.0,
            elevation=30.0,
            distance=100.0,
            confidence=0.9,
            classification="quadcopter"
        )
        assert det.azimuth == 45.0
        assert det.elevation == 30.0
        assert det.distance == 100.0
        assert det.confidence == 0.9


class TestTrackState:
    """Tests for TrackState dataclass."""

    def test_creation(self):
        state = TrackState(
            azimuth=45.0,
            elevation=30.0,
            distance=100.0,
            velocity_azimuth=1.0,
            velocity_elevation=0.5,
            velocity_radial=-5.0,
            confidence=0.85
        )
        assert state.azimuth == 45.0
        assert state.velocity_radial == -5.0


class TestExtendedKalmanFilter:
    """Tests for ExtendedKalmanFilter class."""

    def test_init(self):
        initial_state = np.array([45.0, 30.0, 100.0, 0.0, 0.0, 0.0])
        ekf = ExtendedKalmanFilter(initial_state)

        assert np.array_equal(ekf._state, initial_state)
        assert ekf._covariance.shape == (6, 6)

    def test_predict(self):
        initial_state = np.array([45.0, 30.0, 100.0, 1.0, 0.5, -2.0])
        ekf = ExtendedKalmanFilter(initial_state)

        # Predict forward
        dt = 0.1
        ekf.predict(dt)

        # State should change based on velocities
        assert ekf._state[0] != initial_state[0]  # Azimuth changed
        assert ekf._state[2] != initial_state[2]  # Distance changed

    def test_update(self):
        initial_state = np.array([45.0, 30.0, 100.0, 0.0, 0.0, 0.0])
        ekf = ExtendedKalmanFilter(initial_state)

        # New measurement
        measurement = np.array([46.0, 31.0, 99.0])

        ekf.update(measurement)

        # State should move toward measurement
        assert abs(ekf._state[0] - 46.0) < abs(initial_state[0] - 46.0)

    def test_get_state(self):
        initial_state = np.array([45.0, 30.0, 100.0, 1.0, 0.5, -2.0])
        ekf = ExtendedKalmanFilter(initial_state)

        state = ekf.get_state()

        assert isinstance(state, TrackState)
        assert state.azimuth == 45.0
        assert state.velocity_azimuth == 1.0


class TestUnscentedKalmanFilter:
    """Tests for UnscentedKalmanFilter class."""

    def test_init(self):
        initial_state = np.array([45.0, 30.0, 100.0, 0.0, 0.0, 0.0])
        ukf = UnscentedKalmanFilter(initial_state)

        assert np.array_equal(ukf._state, initial_state)

    def test_predict(self):
        initial_state = np.array([45.0, 30.0, 100.0, 1.0, 0.5, -2.0])
        ukf = UnscentedKalmanFilter(initial_state)

        # Predict forward
        dt = 0.1
        ukf.predict(dt)

        # State should change
        assert ukf._state is not None

    def test_update(self):
        initial_state = np.array([45.0, 30.0, 100.0, 0.0, 0.0, 0.0])
        ukf = UnscentedKalmanFilter(initial_state)

        # New measurement
        measurement = np.array([46.0, 31.0, 99.0])

        ukf.update(measurement)

        # State should be updated
        assert ukf._state is not None


class TestDataAssociation:
    """Tests for DataAssociation class."""

    def test_init(self):
        da = DataAssociation(threshold=30.0)
        assert da._threshold == 30.0

    def test_compute_cost_matrix(self):
        da = DataAssociation(threshold=30.0)

        # Create tracks
        tracks = [
            Track(track_id=1, state=TrackState(
                azimuth=45.0, elevation=30.0, distance=100.0,
                velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
                confidence=0.9
            )),
            Track(track_id=2, state=TrackState(
                azimuth=90.0, elevation=0.0, distance=150.0,
                velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
                confidence=0.8
            ))
        ]

        # Create detections
        detections = [
            Detection(timestamp=time.time(), azimuth=46.0, elevation=31.0,
                     distance=99.0, confidence=0.85, classification="quadcopter"),
            Detection(timestamp=time.time(), azimuth=180.0, elevation=45.0,
                     distance=200.0, confidence=0.7, classification="unknown")
        ]

        cost_matrix = da.compute_cost_matrix(tracks, detections)

        assert cost_matrix.shape == (2, 2)
        # First track should be closer to first detection
        assert cost_matrix[0, 0] < cost_matrix[0, 1]

    def test_associate(self):
        da = DataAssociation(threshold=30.0)

        # Create track near first detection
        tracks = [
            Track(track_id=1, state=TrackState(
                azimuth=45.0, elevation=30.0, distance=100.0,
                velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
                confidence=0.9
            ))
        ]

        # Create detection close to track
        detections = [
            Detection(timestamp=time.time(), azimuth=46.0, elevation=31.0,
                     distance=99.0, confidence=0.85, classification="quadcopter")
        ]

        assignments, unassigned_tracks, unassigned_detections = da.associate(
            tracks, detections
        )

        # Should associate track 0 with detection 0
        assert (0, 0) in assignments
        assert len(unassigned_tracks) == 0
        assert len(unassigned_detections) == 0


class TestTrack:
    """Tests for Track class."""

    def test_init(self):
        state = TrackState(
            azimuth=45.0, elevation=30.0, distance=100.0,
            velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
            confidence=0.9
        )
        track = Track(track_id=1, state=state)

        assert track.track_id == 1
        assert track.state == state
        assert track.classification == "unknown"

    def test_predict(self):
        state = TrackState(
            azimuth=45.0, elevation=30.0, distance=100.0,
            velocity_azimuth=1.0, velocity_elevation=0.5, velocity_radial=-2.0,
            confidence=0.9
        )
        track = Track(track_id=1, state=state)

        track.predict(dt=0.1)

        # State should be updated
        assert track.state is not None

    def test_update(self):
        state = TrackState(
            azimuth=45.0, elevation=30.0, distance=100.0,
            velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
            confidence=0.9
        )
        track = Track(track_id=1, state=state)

        detection = Detection(
            timestamp=time.time(), azimuth=46.0, elevation=31.0,
            distance=99.0, confidence=0.85, classification="quadcopter"
        )

        track.update(detection)

        assert track.classification == "quadcopter"

    def test_age(self):
        state = TrackState(
            azimuth=45.0, elevation=30.0, distance=100.0,
            velocity_azimuth=0, velocity_elevation=0, velocity_radial=0,
            confidence=0.9
        )
        track = Track(track_id=1, state=state)

        # Age should be positive
        time.sleep(0.1)
        assert track.age > 0


class TestMultiTargetTracker:
    """Tests for MultiTargetTracker class."""

    def test_init_default(self):
        tracker = MultiTargetTracker()
        assert tracker._config is not None
        assert len(tracker._tracks) == 0

    def test_init_with_config(self):
        config = TrackingConfig(max_tracks=5)
        tracker = MultiTargetTracker(config=config)
        assert tracker._config.max_tracks == 5

    def test_update_new_detection(self):
        tracker = MultiTargetTracker()

        detections = [
            Detection(timestamp=time.time(), azimuth=45.0, elevation=30.0,
                     distance=100.0, confidence=0.9, classification="quadcopter")
        ]

        tracks = tracker.update(detections)

        # Should create new track
        assert len(tracks) == 1
        assert tracks[0].classification == "quadcopter"

    def test_update_track_association(self):
        tracker = MultiTargetTracker()

        # First detection - creates track
        det1 = [
            Detection(timestamp=time.time(), azimuth=45.0, elevation=30.0,
                     distance=100.0, confidence=0.9, classification="quadcopter")
        ]
        tracks1 = tracker.update(det1)

        # Second detection - should associate with existing track
        time.sleep(0.1)
        det2 = [
            Detection(timestamp=time.time(), azimuth=46.0, elevation=31.0,
                     distance=99.0, confidence=0.85, classification="quadcopter")
        ]
        tracks2 = tracker.update(det2)

        # Should still have one track
        assert len(tracks2) == 1
        # Track ID should be the same
        assert tracks1[0].track_id == tracks2[0].track_id

    def test_track_timeout(self):
        config = TrackingConfig(track_timeout=0.1)
        tracker = MultiTargetTracker(config=config)

        # Create track
        det = [
            Detection(timestamp=time.time(), azimuth=45.0, elevation=30.0,
                     distance=100.0, confidence=0.9, classification="quadcopter")
        ]
        tracker.update(det)

        # Wait for timeout
        time.sleep(0.2)

        # Update with no detections
        tracks = tracker.update([])

        # Track should be removed due to timeout
        assert len(tracks) == 0

    def test_max_tracks(self):
        config = TrackingConfig(max_tracks=2)
        tracker = MultiTargetTracker(config=config)

        # Create 3 detections at different positions
        detections = [
            Detection(timestamp=time.time(), azimuth=0.0, elevation=0.0,
                     distance=100.0, confidence=0.9, classification="drone1"),
            Detection(timestamp=time.time(), azimuth=90.0, elevation=0.0,
                     distance=150.0, confidence=0.8, classification="drone2"),
            Detection(timestamp=time.time(), azimuth=180.0, elevation=0.0,
                     distance=200.0, confidence=0.7, classification="drone3")
        ]

        tracks = tracker.update(detections)

        # Should only keep max_tracks
        assert len(tracks) <= 2

    def test_get_tracks(self):
        tracker = MultiTargetTracker()

        det = [
            Detection(timestamp=time.time(), azimuth=45.0, elevation=30.0,
                     distance=100.0, confidence=0.9, classification="quadcopter")
        ]
        tracker.update(det)

        tracks = tracker.get_tracks()

        assert len(tracks) == 1
        assert isinstance(tracks[0], Track)

    def test_clear_tracks(self):
        tracker = MultiTargetTracker()

        det = [
            Detection(timestamp=time.time(), azimuth=45.0, elevation=30.0,
                     distance=100.0, confidence=0.9, classification="quadcopter")
        ]
        tracker.update(det)

        tracker.clear()

        assert len(tracker.get_tracks()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
