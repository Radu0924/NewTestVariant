"""
Tests that streaming notch filtering produces consistent results
when processing data in chunks vs as a single block.
"""

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.signal_processor import FilterBank, FilterConfig


def test_notch_streaming_chunk_consistency():
    """Verify that processing in chunks matches processing all at once."""
    sample_rate = 48000
    config = FilterConfig(notch_freqs=[50.0, 60.0])

    # Create test signal with notch frequencies
    duration = 0.2  # 200ms
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Mix of target and notch frequencies
    signal = (
        np.sin(2 * np.pi * 500 * t) +  # target frequency
        np.sin(2 * np.pi * 50 * t) +   # notch frequency 1
        np.sin(2 * np.pi * 60 * t)     # notch frequency 2
    ).astype(np.float32)

    # Process as single block
    fb_single = FilterBank(sample_rate, config)
    fb_single._ensure_state(1)
    result_single = fb_single.apply_notch(signal.copy())

    # Process in chunks
    fb_chunked = FilterBank(sample_rate, config)
    fb_chunked._ensure_state(1)
    chunk_size = 512
    chunks = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        processed_chunk = fb_chunked.apply_notch(chunk.copy())
        chunks.append(processed_chunk)
    result_chunked = np.concatenate(chunks)

    # Compare results - should be identical since streaming state is preserved
    assert result_single.shape == result_chunked.shape
    assert np.allclose(result_single, result_chunked, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(result_single - result_chunked))}"


def test_notch_streaming_multichannel():
    """Verify multichannel streaming notch consistency."""
    sample_rate = 48000
    config = FilterConfig(notch_freqs=[50.0])
    num_channels = 4
    samples = 2048

    # Create test data
    np.random.seed(42)
    data = np.random.randn(num_channels, samples).astype(np.float32)

    # Process as single block
    fb_single = FilterBank(sample_rate, config)
    result_single = fb_single.apply_notch(data.copy())

    # Process in two chunks
    fb_chunked = FilterBank(sample_rate, config)
    half = samples // 2
    result1 = fb_chunked.apply_notch(data[:, :half].copy())
    result2 = fb_chunked.apply_notch(data[:, half:].copy())
    result_chunked = np.concatenate([result1, result2], axis=1)

    assert np.allclose(result_single, result_chunked, rtol=1e-5, atol=1e-6)
