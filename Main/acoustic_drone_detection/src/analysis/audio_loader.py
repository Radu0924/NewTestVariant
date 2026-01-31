"""
Audio Loader Module

Loads audio files in various formats for offline analysis:
- WAV, FLAC, MP3, OGG support
- Multi-channel file handling
- Automatic format detection
- Metadata extraction
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Generator
from dataclasses import dataclass
import wave
import struct
import os


@dataclass
class AudioMetadata:
    """Metadata for an audio file."""
    filepath: str
    filename: str
    format: str
    sample_rate: int
    channels: int
    duration_seconds: float
    total_samples: int
    bit_depth: int
    file_size_bytes: int


@dataclass
class AudioSegment:
    """A segment of audio data."""
    data: np.ndarray
    sample_rate: int
    channels: int
    start_time: float
    duration: float
    metadata: Optional[AudioMetadata] = None


class AudioLoader:
    """
    Multi-format audio file loader.

    Supports loading WAV, FLAC, MP3, and OGG files with
    automatic format detection and multi-channel handling.
    """

    SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac']

    def __init__(self, target_sample_rate: Optional[int] = None):
        """
        Initialize audio loader.

        Args:
            target_sample_rate: Target sample rate for resampling (None = native).
        """
        self._target_sample_rate = target_sample_rate
        self._scipy_available = self._check_scipy()
        self._librosa_available = self._check_librosa()
        self._soundfile_available = self._check_soundfile()

    def _check_scipy(self) -> bool:
        try:
            from scipy.io import wavfile
            return True
        except ImportError:
            return False

    def _check_librosa(self) -> bool:
        try:
            import librosa
            return True
        except ImportError:
            return False

    def _check_soundfile(self) -> bool:
        try:
            import soundfile
            return True
        except ImportError:
            return False

    def load(self, filepath: str) -> AudioSegment:
        """
        Load an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            AudioSegment with loaded data.

        Raises:
            ValueError: If format is not supported.
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Load based on format and available libraries
        if ext == '.wav':
            data, sample_rate = self._load_wav(filepath)
        elif ext == '.flac':
            data, sample_rate = self._load_flac(filepath)
        elif ext in ['.mp3', '.ogg', '.m4a', '.aac']:
            data, sample_rate = self._load_compressed(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Ensure correct shape (channels x samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.shape[0] > data.shape[1]:
            data = data.T

        # Resample if needed
        if self._target_sample_rate and sample_rate != self._target_sample_rate:
            data = self._resample(data, sample_rate, self._target_sample_rate)
            sample_rate = self._target_sample_rate

        # Get metadata
        metadata = self.get_metadata(filepath)

        return AudioSegment(
            data=data,
            sample_rate=sample_rate,
            channels=data.shape[0],
            start_time=0.0,
            duration=data.shape[1] / sample_rate,
            metadata=metadata
        )

    def _load_wav(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load WAV file."""
        if self._scipy_available:
            from scipy.io import wavfile
            sample_rate, data = wavfile.read(filepath)

            # Convert to float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0

            return data, sample_rate

        elif self._soundfile_available:
            import soundfile as sf
            data, sample_rate = sf.read(filepath, dtype='float32')
            return data, sample_rate

        else:
            # Fallback to wave module (limited support)
            return self._load_wav_native(filepath)

    def _load_wav_native(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load WAV using native wave module."""
        with wave.open(filepath, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()

            raw_data = wf.readframes(n_frames)

            if sample_width == 1:
                dtype = np.uint8
                max_val = 128.0
                offset = 128
            elif sample_width == 2:
                dtype = np.int16
                max_val = 32768.0
                offset = 0
            elif sample_width == 4:
                dtype = np.int32
                max_val = 2147483648.0
                offset = 0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            data = np.frombuffer(raw_data, dtype=dtype)
            data = (data.astype(np.float32) - offset) / max_val

            if channels > 1:
                data = data.reshape(-1, channels)

            return data, sample_rate

    def _load_flac(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load FLAC file."""
        if self._soundfile_available:
            import soundfile as sf
            data, sample_rate = sf.read(filepath, dtype='float32')
            return data, sample_rate

        elif self._librosa_available:
            import librosa
            data, sample_rate = librosa.load(filepath, sr=None, mono=False)
            return data, sample_rate

        else:
            raise ImportError("soundfile or librosa required for FLAC support")

    def _load_compressed(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load compressed audio (MP3, OGG, etc.)."""
        if self._librosa_available:
            import librosa
            data, sample_rate = librosa.load(filepath, sr=None, mono=False)
            return data, sample_rate

        elif self._soundfile_available:
            import soundfile as sf
            data, sample_rate = sf.read(filepath, dtype='float32')
            return data, sample_rate

        else:
            raise ImportError("librosa or soundfile required for compressed audio")

    def _resample(
        self,
        data: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio data."""
        if self._librosa_available:
            import librosa
            if data.ndim == 1:
                return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
            else:
                resampled = []
                for ch in range(data.shape[0]):
                    resampled.append(
                        librosa.resample(data[ch], orig_sr=orig_sr, target_sr=target_sr)
                    )
                return np.array(resampled)

        elif self._scipy_available:
            from scipy import signal
            ratio = target_sr / orig_sr
            new_length = int(data.shape[-1] * ratio)

            if data.ndim == 1:
                return signal.resample(data, new_length)
            else:
                resampled = np.zeros((data.shape[0], new_length))
                for ch in range(data.shape[0]):
                    resampled[ch] = signal.resample(data[ch], new_length)
                return resampled

        else:
            raise ImportError("scipy or librosa required for resampling")

    def get_metadata(self, filepath: str) -> AudioMetadata:
        """
        Get metadata for an audio file.

        Args:
            filepath: Path to audio file.

        Returns:
            AudioMetadata object.
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        file_size = path.stat().st_size

        # Get audio info
        if ext == '.wav':
            with wave.open(filepath, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                total_samples = wf.getnframes()
                bit_depth = wf.getsampwidth() * 8
                duration = total_samples / sample_rate

        elif self._soundfile_available:
            import soundfile as sf
            info = sf.info(filepath)
            sample_rate = info.samplerate
            channels = info.channels
            total_samples = info.frames
            duration = info.duration
            bit_depth = 16  # Approximate

        elif self._librosa_available:
            import librosa
            duration = librosa.get_duration(path=filepath)
            # Load small portion to get info
            data, sample_rate = librosa.load(filepath, sr=None, mono=False, duration=0.1)
            channels = 1 if data.ndim == 1 else data.shape[0]
            total_samples = int(duration * sample_rate)
            bit_depth = 16

        else:
            # Minimal info
            sample_rate = 0
            channels = 0
            total_samples = 0
            duration = 0.0
            bit_depth = 0

        return AudioMetadata(
            filepath=str(path.absolute()),
            filename=path.name,
            format=ext[1:].upper(),
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration,
            total_samples=total_samples,
            bit_depth=bit_depth,
            file_size_bytes=file_size
        )

    def load_segment(
        self,
        filepath: str,
        start_time: float,
        duration: float
    ) -> AudioSegment:
        """
        Load a segment of an audio file.

        Args:
            filepath: Path to audio file.
            start_time: Start time in seconds.
            duration: Duration in seconds.

        Returns:
            AudioSegment with the specified portion.
        """
        if self._librosa_available:
            import librosa
            data, sample_rate = librosa.load(
                filepath,
                sr=self._target_sample_rate,
                mono=False,
                offset=start_time,
                duration=duration
            )

            if data.ndim == 1:
                data = data.reshape(1, -1)

            return AudioSegment(
                data=data,
                sample_rate=sample_rate,
                channels=data.shape[0],
                start_time=start_time,
                duration=data.shape[1] / sample_rate
            )

        else:
            # Load full file and slice
            segment = self.load(filepath)
            start_sample = int(start_time * segment.sample_rate)
            end_sample = int((start_time + duration) * segment.sample_rate)

            data = segment.data[:, start_sample:end_sample]

            return AudioSegment(
                data=data,
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                start_time=start_time,
                duration=data.shape[1] / segment.sample_rate
            )

    def iter_segments(
        self,
        filepath: str,
        segment_duration: float,
        overlap: float = 0.0
    ) -> Generator[AudioSegment, None, None]:
        """
        Iterate over audio file in segments.

        Args:
            filepath: Path to audio file.
            segment_duration: Duration of each segment in seconds.
            overlap: Overlap between segments in seconds.

        Yields:
            AudioSegment objects.
        """
        metadata = self.get_metadata(filepath)
        total_duration = metadata.duration_seconds

        step = segment_duration - overlap
        current_time = 0.0

        while current_time < total_duration:
            remaining = total_duration - current_time
            duration = min(segment_duration, remaining)

            yield self.load_segment(filepath, current_time, duration)

            current_time += step

    def scan_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> List[AudioMetadata]:
        """
        Scan directory for audio files.

        Args:
            directory: Directory path.
            recursive: Scan subdirectories.

        Returns:
            List of AudioMetadata for found files.
        """
        path = Path(directory)
        files = []

        pattern = '**/*' if recursive else '*'

        for ext in self.SUPPORTED_FORMATS:
            for file_path in path.glob(f"{pattern}{ext}"):
                try:
                    metadata = self.get_metadata(str(file_path))
                    files.append(metadata)
                except Exception:
                    pass

        return files
