"""
Basic multilateration test with confidence weights.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.localization import MultilaterationSolver, LocalizationConfig


def test_multilateration_with_weights():
    # Simple non-collinear geometry (tetra-like)
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    source = np.array([2.0, 2.0, 1.0])
    c = 343.0
    distances = np.linalg.norm(mic_positions - source, axis=1)
    tdoas = (distances - distances[0]) / c

    config = LocalizationConfig(max_distance=50.0)
    solver = MultilaterationSolver(mic_positions, sound_speed=c, config=config)

    weights = np.array([1.0, 0.9, 0.9, 0.9])
    result = solver.localize(tdoas, weights=weights)

    estimated = result.position.to_array()
    error = np.linalg.norm(estimated - source)

    assert error < 1.0
    if result.covariance is not None:
        assert result.covariance.shape == (3, 3)
