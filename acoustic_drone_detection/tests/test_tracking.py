"""
Unit tests for Tracking module (current API).
"""

import time
import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tracking import (
    MultiTargetTracker, TrackingConfig, Detection, ExtendedKalmanFilter
)


def test_tracking_config_defaults():
    config = TrackingConfig()
    assert config.filter_type == "ekf"
    assert config.max_tracks == 10
    assert config.track_timeout == 5.0
    assert config.association_threshold == 20.0


def test_detection_to_cartesian():
    det = Detection(
        timestamp=time.time(),
        azimuth=0.0,
        elevation=0.0,
        distance=100.0,
        confidence=0.9,
        classification="unknown"
    )
    xyz = det.to_cartesian()
    assert xyz.shape == (3,)
    assert np.isclose(xyz[0], 100.0, atol=1e-3)


def test_multitarget_tracker_creates_track():
    tracker = MultiTargetTracker(config=TrackingConfig(max_tracks=5))
    det = Detection(
        timestamp=time.time(),
        azimuth=45.0,
        elevation=0.0,
        distance=50.0,
        confidence=0.9,
        classification="unknown"
    )
    tracks = tracker.update([det])
    assert len(tracks) == 1


def test_ekf_predict_update():
    initial_state = np.zeros(9)
    ekf = ExtendedKalmanFilter(initial_state=initial_state)
    state_pred, _ = ekf.predict(0.1)
    assert state_pred.shape == (9,)
    measurement = np.array([1.0, 0.0, 0.0])
    state_upd, _ = ekf.update(measurement)
    assert state_upd.shape == (9,)
