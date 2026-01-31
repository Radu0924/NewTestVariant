"""
Audio Interface Module

Provides hardware abstraction for audio devices:
- Device discovery and selection
- Multi-channel configuration
- Sample rate management
- Buffer management
"""

import sounddevice as sd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import threading
import time


@dataclass
class AudioDeviceCapabilities:
    """Capabilities of an audio device."""
    device_id: int
    name: str
    hostapi: int
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    supported_sample_rates: List[int] = field(default_factory=list)
    low_latency_input: float = 0.0
    high_latency_input: float = 0.0
    is_default_input: bool = False
    is_default_output: bool = False


@dataclass
class AudioStreamConfig:
    """Configuration for an audio stream."""
    device_id: Optional[int] = None
    sample_rate: int = 48000
    channels: int = 12
    buffer_size: int = 512
    dtype: str = "float32"
    latency: str = "low"  # low, high, or float in seconds


class AudioInterfaceManager:
    """
    Manages audio hardware interfaces.

    Provides device discovery, configuration, and stream management.
    """

    SUPPORTED_SAMPLE_RATES = [44100, 48000, 96000, 192000]
    SUPPORTED_DTYPES = ['int16', 'int24', 'int32', 'float32']

    def __init__(self):
        """Initialize audio interface manager."""
        self._devices: Dict[int, AudioDeviceCapabilities] = {}
        self._active_stream: Optional[sd.InputStream] = None
        self._stream_config: Optional[AudioStreamConfig] = None
        self._lock = threading.Lock()

        self.refresh_devices()

    def refresh_devices(self) -> List[AudioDeviceCapabilities]:
        """
        Refresh list of available audio devices.

        Returns:
            List of AudioDeviceCapabilities objects.
        """
        self._devices.clear()

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                hostapi_idx = device['hostapi']
                hostapi_name = hostapis[hostapi_idx]['name'] if hostapi_idx < len(hostapis) else "Unknown"

                capabilities = AudioDeviceCapabilities(
                    device_id=idx,
                    name=device['name'],
                    hostapi=hostapi_idx,
                    hostapi_name=hostapi_name,
                    max_input_channels=device['max_input_channels'],
                    max_output_channels=device['max_output_channels'],
                    default_sample_rate=device['default_samplerate'],
                    low_latency_input=device.get('default_low_input_latency', 0),
                    high_latency_input=device.get('default_high_input_latency', 0),
                    is_default_input=(idx == default_input),
                    is_default_output=(idx == default_output)
                )

                # Test supported sample rates
                for rate in self.SUPPORTED_SAMPLE_RATES:
                    if self._test_sample_rate(idx, device['max_input_channels'], rate):
                        capabilities.supported_sample_rates.append(rate)

                self._devices[idx] = capabilities

        return list(self._devices.values())

    def _test_sample_rate(
        self,
        device_id: int,
        channels: int,
        sample_rate: int
    ) -> bool:
        """Test if a sample rate is supported."""
        try:
            sd.check_input_settings(
                device=device_id,
                channels=min(channels, 2),  # Test with fewer channels
                samplerate=sample_rate,
                dtype='float32'
            )
            return True
        except Exception:
            return False

    def get_device(self, device_id: int) -> Optional[AudioDeviceCapabilities]:
        """Get capabilities for a specific device."""
        return self._devices.get(device_id)

    def get_default_input_device(self) -> Optional[AudioDeviceCapabilities]:
        """Get the default input device."""
        for device in self._devices.values():
            if device.is_default_input:
                return device
        return None

    def get_multi_channel_devices(self, min_channels: int = 8) -> List[AudioDeviceCapabilities]:
        """
        Get devices with at least the specified number of input channels.

        Args:
            min_channels: Minimum number of input channels required.

        Returns:
            List of compatible devices.
        """
        return [d for d in self._devices.values() if d.max_input_channels >= min_channels]

    def validate_config(self, config: AudioStreamConfig) -> Tuple[bool, str]:
        """
        Validate stream configuration.

        Args:
            config: Stream configuration to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        device_id = config.device_id
        if device_id is None:
            device_id = sd.default.device[0]

        device = self._devices.get(device_id)
        if device is None:
            return False, f"Device {device_id} not found"

        if config.channels > device.max_input_channels:
            return False, f"Device only supports {device.max_input_channels} channels"

        if config.sample_rate not in device.supported_sample_rates:
            if not device.supported_sample_rates:
                return False, "No supported sample rates detected"
            return False, f"Sample rate {config.sample_rate} not supported"

        if config.buffer_size < 64 or config.buffer_size > 8192:
            return False, "Buffer size must be between 64 and 8192"

        return True, ""

    def get_optimal_buffer_size(
        self,
        device_id: int,
        sample_rate: int,
        target_latency_ms: float = 10.0
    ) -> int:
        """
        Calculate optimal buffer size for target latency.

        Args:
            device_id: Audio device ID.
            sample_rate: Sample rate in Hz.
            target_latency_ms: Target latency in milliseconds.

        Returns:
            Recommended buffer size.
        """
        # Calculate samples for target latency
        samples = int(sample_rate * target_latency_ms / 1000)

        # Round to power of 2
        buffer_size = int(2 ** np.ceil(np.log2(samples)))

        # Clamp to reasonable range
        return max(64, min(buffer_size, 4096))

    def get_device_latency(
        self,
        device_id: int,
        buffer_size: int,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Calculate expected latency for configuration.

        Args:
            device_id: Audio device ID.
            buffer_size: Buffer size in samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Dictionary with latency information.
        """
        device = self._devices.get(device_id)
        if device is None:
            return {}

        buffer_latency_ms = buffer_size / sample_rate * 1000
        device_latency_ms = device.low_latency_input * 1000

        return {
            'buffer_latency_ms': buffer_latency_ms,
            'device_latency_ms': device_latency_ms,
            'total_latency_ms': buffer_latency_ms + device_latency_ms
        }

    @property
    def devices(self) -> List[AudioDeviceCapabilities]:
        """Get all available input devices."""
        return list(self._devices.values())


class AudioInputStream:
    """
    Managed audio input stream.

    Wraps sounddevice InputStream with additional management features.
    """

    def __init__(self, config: AudioStreamConfig):
        """
        Initialize audio input stream.

        Args:
            config: Stream configuration.
        """
        self._config = config
        self._stream: Optional[sd.InputStream] = None
        self._callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None
        self._running = False
        self._lock = threading.Lock()

        # Statistics
        self._total_frames = 0
        self._overflow_count = 0
        self._underflow_count = 0
        self._start_time: Optional[float] = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags
    ) -> None:
        """Internal audio callback."""
        if status.input_overflow:
            self._overflow_count += 1
        if status.input_underflow:
            self._underflow_count += 1

        self._total_frames += frames

        if self._callback:
            try:
                self._callback(indata.copy(), frames, time_info, status)
            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)

    def start(self, callback: Callable) -> bool:
        """
        Start the audio stream.

        Args:
            callback: Callback function for audio data.

        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._running:
                return True

            self._callback = callback

            try:
                self._stream = sd.InputStream(
                    device=self._config.device_id,
                    channels=self._config.channels,
                    samplerate=self._config.sample_rate,
                    blocksize=self._config.buffer_size,
                    dtype=self._config.dtype,
                    latency=self._config.latency,
                    callback=self._audio_callback
                )

                self._stream.start()
                self._running = True
                self._start_time = time.time()
                self._total_frames = 0
                self._overflow_count = 0
                self._underflow_count = 0

                return True

            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)
                return False

    def stop(self) -> None:
        """Stop the audio stream."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def set_error_callback(self, callback: Callable) -> None:
        """Set error callback."""
        self._error_callback = callback

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get stream statistics."""
        duration = time.time() - self._start_time if self._start_time else 0

        return {
            'running': self._running,
            'total_frames': self._total_frames,
            'duration_seconds': duration,
            'overflow_count': self._overflow_count,
            'underflow_count': self._underflow_count,
            'average_fps': self._total_frames / (duration + 1e-12) / self._config.buffer_size
        }

    @property
    def config(self) -> AudioStreamConfig:
        """Get stream configuration."""
        return self._config
