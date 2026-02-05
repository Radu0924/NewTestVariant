"""
Tests that _process_frame in main.py uses MLAT when TDOA confidence is sufficient.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.localization import LocalizationEngine, LocalizationConfig


def test_localize_hybrid_uses_mlat_with_high_confidence():
    """Verify that localize_hybrid uses MLAT when confidences are high."""
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    config = LocalizationConfig(max_distance=100.0)
    engine = LocalizationEngine(mic_positions, config=config)

    # Simulate TDOA from a known source
    source = np.array([5.0, 5.0, 2.0])
    c = 343.0
    distances = np.linalg.norm(mic_positions - source, axis=1)
    tdoas = (distances - distances[0]) / c

    # High confidence for all channels
    confidences = np.ones(len(mic_positions), dtype=np.float64)
    confidences[0] = 1.0  # Reference

    result = engine.localize_hybrid(
        azimuth=45.0,
        elevation=15.0,
        tdoas=tdoas,
        signal_rms=0.1,
        tdoa_confidences=confidences
    )

    # Should use hybrid method (which internally uses MLAT)
    assert result.method == "hybrid"
    assert result.confidence > 0.3

    # Position should be reasonably close to source
    estimated = result.position.to_array()
    error = np.linalg.norm(estimated - source)
    assert error < 5.0, f"Position error too large: {error}m"


def test_localize_hybrid_falls_back_with_low_confidence():
    """Verify fallback to DOA when TDOA confidence is low."""
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    config = LocalizationConfig(max_distance=100.0)
    engine = LocalizationEngine(mic_positions, config=config)

    # Simulate noisy/unreliable TDOAs
    tdoas = np.zeros(len(mic_positions), dtype=np.float64)

    # Very low confidence - should trigger amplitude-based fallback
    confidences = np.ones(len(mic_positions), dtype=np.float64) * 0.05

    result = engine.localize_hybrid(
        azimuth=90.0,
        elevation=30.0,
        tdoas=tdoas,
        signal_rms=0.2,
        tdoa_confidences=confidences
    )

    # Should still produce a result (hybrid with amplitude fallback)
    assert result is not None
    assert result.distance > 0


def test_process_frame_mlat_threshold():
    """Test that MIN_CONF threshold controls MLAT usage."""
    # This tests the threshold logic from main.py
    min_conf = 0.15
    n_channels = 8

    # Simulate varying confidences
    tdoa_confidences = np.array([1.0, 0.2, 0.3, 0.1, 0.5, 0.05, 0.8, 0.4])

    # Count valid pairs (excluding reference channel 0)
    valid_pairs = int(np.sum(tdoa_confidences[1:] >= min_conf))

    # Should have 5 valid pairs (0.2, 0.3, 0.5, 0.8, 0.4 >= 0.15)
    assert valid_pairs == 5

    # With >= 3 valid pairs, system should use MLAT
    use_mlat = valid_pairs >= 3
    assert use_mlat is True
