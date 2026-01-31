"""
Multi-Channel Audio Capture Module

Provides real-time multi-channel audio capture with:
- Support for 8-20 microphones
- Configurable sample rates (44100, 48000, 96000, 192000 Hz)
- Thread-safe ring buffer
- Dropout detection and recovery
- Signal level monitoring
"""

import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import queue


@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    supported_sample_rates: List[int] = field(default_factory=list)


@dataclass
class ChannelStatus:
    """Status information for a single channel."""
    channel_id: int
    level_db: float = -100.0
    peak_db: float = -100.0
    is_clipping: bool = False
    is_active: bool = True


class RingBuffer:
    """
    Thread-safe ring buffer for audio data.

    Provides efficient circular buffer storage for continuous audio streams.
    """

    def __init__(self, channels: int, buffer_seconds: float, sample_rate: int):
        """
        Initialize the ring buffer.

        Args:
            channels: Number of audio channels.
            buffer_seconds: Buffer duration in seconds.
            sample_rate: Sample rate in Hz.
        """
        self._channels = channels
        self._sample_rate = sample_rate
        self._buffer_size = int(buffer_seconds * sample_rate)

        self._buffer = np.zeros((channels, self._buffer_size), dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def write(self, data: np.ndarray) -> int:
        """
        Write data to the buffer.

        Args:
            data: Audio data array (channels x samples).

        Returns:
            Number of samples written.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.shape[0] > data.shape[1]:
            data = data.T

        samples = data.shape[1]

        with self._not_full:
            # Wait if buffer is full
            while self._available >= self._buffer_size:
                self._not_full.wait(timeout=0.1)
                if self._available >= self._buffer_size:
                    # Overwrite oldest data
                    self._read_pos = (self._read_pos + samples) % self._buffer_size
                    self._available -= samples

            # Write data
            space_at_end = self._buffer_size - self._write_pos
            if samples <= space_at_end:
                self._buffer[:, self._write_pos:self._write_pos + samples] = data
            else:
                self._buffer[:, self._write_pos:] = data[:, :space_at_end]
                self._buffer[:, :samples - space_at_end] = data[:, space_at_end:]

            self._write_pos = (self._write_pos + samples) % self._buffer_size
            self._available += samples

            self._not_empty.notify_all()

        return samples

    def read(self, samples: int, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read data from the buffer.

        Args:
            samples: Number of samples to read.
            timeout: Timeout in seconds.

        Returns:
            Audio data array or None if timeout.
        """
        with self._not_empty:
            # Wait for data
            start_time = time.time()
            while self._available < samples:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            # Read data
            data = np.zeros((self._channels, samples), dtype=np.float32)

            space_at_end = self._buffer_size - self._read_pos
            if samples <= space_at_end:
                data[:] = self._buffer[:, self._read_pos:self._read_pos + samples]
            else:
                data[:, :space_at_end] = self._buffer[:, self._read_pos:]
                data[:, space_at_end:] = self._buffer[:, :samples - space_at_end]

            self._read_pos = (self._read_pos + samples) % self._buffer_size
            self._available -= samples

            self._not_full.notify_all()

        return data

    def peek(self, samples: int) -> Optional[np.ndarray]:
        """
        Peek at data without removing it from buffer.

        Args:
            samples: Number of samples to peek.

        Returns:
            Audio data array or None if not enough data.
        """
        with self._lock:
            if self._available < samples:
                return None

            data = np.zeros((self._channels, samples), dtype=np.float32)

            space_at_end = self._buffer_size - self._read_pos
            if samples <= space_at_end:
                data[:] = self._buffer[:, self._read_pos:self._read_pos + samples]
            else:
                data[:, :space_at_end] = self._buffer[:, self._read_pos:]
                data[:, space_at_end:] = self._buffer[:, :samples - space_at_end]

        return data

    @property
    def available(self) -> int:
        """Get number of available samples."""
        with self._lock:
            return self._available

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._available = 0


class DropoutDetector:
    """
    Detects audio dropouts and buffer underruns.

    Monitors the audio stream for discontinuities and reports issues.
    """

    def __init__(self, threshold_ms: float = 10.0, window_size: int = 100):
        """
        Initialize dropout detector.

        Args:
            threshold_ms: Threshold for dropout detection in ms.
            window_size: Number of samples for detection window.
        """
        self._threshold_ms = threshold_ms
        self._window_size = window_size

        self._last_time: Optional[float] = None
        self._dropout_count = 0
        self._dropout_times: deque = deque(maxlen=window_size)
        self._callback: Optional[Callable] = None

    def check(self, expected_samples: int, sample_rate: int) -> bool:
        """
        Check for dropout.

        Args:
            expected_samples: Expected number of samples.
            sample_rate: Sample rate in Hz.

        Returns:
            True if dropout detected.
        """
        current_time = time.time()

        if self._last_time is not None:
            expected_interval = expected_samples / sample_rate
            actual_interval = current_time - self._last_time
            gap_ms = (actual_interval - expected_interval) * 1000

            if gap_ms > self._threshold_ms:
                self._dropout_count += 1
                self._dropout_times.append(current_time)

                if self._callback:
                    self._callback(gap_ms)

                self._last_time = current_time
                return True

        self._last_time = current_time
        return False

    def set_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for dropout events."""
        self._callback = callback

    @property
    def dropout_count(self) -> int:
        """Get total dropout count."""
        return self._dropout_count

    @property
    def recent_dropouts(self) -> int:
        """Get number of recent dropouts."""
        cutoff = time.time() - 60  # Last minute
        return sum(1 for t in self._dropout_times if t > cutoff)

    def reset(self) -> None:
        """Reset the detector."""
        self._last_time = None
        self._dropout_count = 0
        self._dropout_times.clear()


class AudioCapture:
    """
    Multi-channel audio capture system.

    Provides real-time capture from multi-channel audio interfaces
    with monitoring, dropout detection, and thread-safe buffering.
    """

    SUPPORTED_SAMPLE_RATES = [44100, 48000, 96000, 192000]
    SUPPORTED_BIT_DEPTHS = [16, 24, 32]

    def __init__(
        self,
        num_channels: int = 12,
        sample_rate: int = 48000,
        buffer_size: int = 512,
        device_index: Optional[int] = None,
        buffer_seconds: float = 5.0
    ):
        """
        Initialize audio capture.

        Args:
            num_channels: Number of input channels (8-20).
            sample_rate: Sample rate in Hz.
            buffer_size: Buffer size in samples.
            device_index: Audio device index (None for default).
            buffer_seconds: Ring buffer duration in seconds.
        """
        self._num_channels = max(8, min(20, num_channels))
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._device_index = device_index
        self._buffer_seconds = buffer_seconds

        self._ring_buffer = RingBuffer(
            self._num_channels, buffer_seconds, sample_rate
        )
        self._dropout_detector = DropoutDetector()

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

        self._channel_status: List[ChannelStatus] = [
            ChannelStatus(i) for i in range(self._num_channels)
        ]

        self._callbacks: List[Callable] = []
        self._error_callback: Optional[Callable] = None

        self._total_samples = 0
        self._start_time: Optional[float] = None

    @staticmethod
    def get_available_devices() -> List[AudioDeviceInfo]:
        """
        Get list of available audio input devices.

        Returns:
            List of AudioDeviceInfo objects.
        """
        devices = []

        for idx, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                info = AudioDeviceInfo(
                    index=idx,
                    name=device['name'],
                    max_input_channels=device['max_input_channels'],
                    max_output_channels=device['max_output_channels'],
                    default_sample_rate=device['default_samplerate']
                )

                # Test supported sample rates
                for rate in AudioCapture.SUPPORTED_SAMPLE_RATES:
                    try:
                        sd.check_input_settings(
                            device=idx,
                            channels=min(device['max_input_channels'], 4),
                            samplerate=rate
                        )
                        info.supported_sample_rates.append(rate)
                    except Exception:
                        pass

                devices.append(info)

        return devices

    @staticmethod
    def get_default_device() -> Optional[int]:
        """Get the default input device index."""
        try:
            return sd.default.device[0]
        except Exception:
            return None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags
    ) -> None:
        """
        Callback for audio input stream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Time information.
            status: Status flags.
        """
        if status:
            if status.input_overflow:
                self._dropout_detector.check(frames, self._sample_rate)

        # Convert to float32 and transpose
        data = indata.T.astype(np.float32)

        # Update channel status
        self._update_channel_status(data)

        # Write to ring buffer
        self._ring_buffer.write(data)

        # Update statistics
        self._total_samples += frames

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)

    def _update_channel_status(self, data: np.ndarray) -> None:
        """Update channel status with current levels."""
        for i in range(min(data.shape[0], len(self._channel_status))):
            channel_data = data[i]
            rms = np.sqrt(np.mean(channel_data ** 2))
            peak = np.max(np.abs(channel_data))

            level_db = 20 * np.log10(rms + 1e-12)
            peak_db = 20 * np.log10(peak + 1e-12)

            self._channel_status[i].level_db = level_db
            self._channel_status[i].peak_db = peak_db
            self._channel_status[i].is_clipping = peak > 0.99

    def start(self) -> bool:
        """
        Start audio capture.

        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._running:
                return True

            try:
                # Determine actual channel count
                device_info = sd.query_devices(self._device_index)
                actual_channels = min(
                    self._num_channels,
                    device_info['max_input_channels']
                )

                self._stream = sd.InputStream(
                    device=self._device_index,
                    channels=actual_channels,
                    samplerate=self._sample_rate,
                    blocksize=self._buffer_size,
                    dtype=np.float32,
                    callback=self._audio_callback
                )

                self._stream.start()
                self._running = True
                self._start_time = time.time()
                self._dropout_detector.reset()

                return True

            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)
                return False

    def stop(self) -> None:
        """Stop audio capture."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def read(self, samples: int, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read audio data from buffer.

        Args:
            samples: Number of samples to read.
            timeout: Timeout in seconds.

        Returns:
            Audio data (channels x samples) or None.
        """
        return self._ring_buffer.read(samples, timeout)

    def read_latest(self, samples: int) -> Optional[np.ndarray]:
        """
        Read the latest samples (non-blocking).

        Args:
            samples: Number of samples to read.

        Returns:
            Audio data or None.
        """
        return self._ring_buffer.peek(samples)

    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add a callback for new audio data."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set error callback."""
        self._error_callback = callback

    def set_dropout_callback(self, callback: Callable[[float], None]) -> None:
        """Set dropout detection callback."""
        self._dropout_detector.set_callback(callback)

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def channel_status(self) -> List[ChannelStatus]:
        """Get status of all channels."""
        return self._channel_status.copy()

    @property
    def buffer_available(self) -> int:
        """Get available samples in buffer."""
        return self._ring_buffer.available

    @property
    def dropout_count(self) -> int:
        """Get total dropout count."""
        return self._dropout_detector.dropout_count

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get capture statistics."""
        duration = time.time() - self._start_time if self._start_time else 0

        return {
            'running': self._running,
            'total_samples': self._total_samples,
            'duration_seconds': duration,
            'sample_rate': self._sample_rate,
            'channels': self._num_channels,
            'buffer_size': self._buffer_size,
            'buffer_available': self.buffer_available,
            'dropout_count': self._dropout_detector.dropout_count,
            'recent_dropouts': self._dropout_detector.recent_dropouts
        }

    def set_device(self, device_index: int) -> None:
        """
        Set the audio device.

        Args:
            device_index: Device index.
        """
        was_running = self._running

        if was_running:
            self.stop()

        self._device_index = device_index

        if was_running:
            self.start()

    def set_sample_rate(self, sample_rate: int) -> None:
        """
        Set the sample rate.

        Args:
            sample_rate: Sample rate in Hz.
        """
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        was_running = self._running

        if was_running:
            self.stop()

        self._sample_rate = sample_rate
        self._ring_buffer = RingBuffer(
            self._num_channels, self._buffer_seconds, sample_rate
        )

        if was_running:
            self.start()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
