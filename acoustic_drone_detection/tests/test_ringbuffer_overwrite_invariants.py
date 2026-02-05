"""
Tests for RingBuffer overwrite behavior and invariants.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_capture import RingBuffer


def test_ringbuffer_overwrite_keeps_tail():
    # Buffer size = 10 samples
    rb = RingBuffer(channels=1, buffer_seconds=1.0, sample_rate=10)

    data = np.arange(25, dtype=np.float32).reshape(1, -1)
    written = rb.write(data)

    assert written == 10
    assert rb.available == 10

    out = rb.read(10, timeout=0.01)
    assert out is not None
    assert out.shape == (1, 10)
    assert np.allclose(out[0], np.arange(15, 25, dtype=np.float32))
