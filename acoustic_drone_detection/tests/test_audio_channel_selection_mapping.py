"""
Tests for enabled channel selection/mapping in AudioCapture.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_capture import AudioCapture


class _DummyStatus:
    input_overflow = False


def test_audio_capture_channel_selection():
    capture = AudioCapture(
        num_channels=12,
        sample_rate=48000,
        buffer_size=8,
        requested_channels=12,
        enabled_channels=[0, 2, 4, 6, 8, 10, 11, 1]
    )

    frames = 8
    indata = np.zeros((frames, 12), dtype=np.float32)
    for ch in range(12):
        indata[:, ch] = ch + 1  # channel marker

    capture._audio_callback(indata, frames, None, _DummyStatus())

    out = capture.read(frames, timeout=0.01)
    assert out is not None
    assert out.shape == (8, frames)

    # Expect selected channels in the configured order
    expected = np.array([1, 3, 5, 7, 9, 11, 12, 2], dtype=np.float32)
    assert np.allclose(out[:, 0], expected)
