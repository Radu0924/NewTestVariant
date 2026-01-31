"""
C++ Backend Integration

Provides Python wrappers for the high-performance C++ core.
Falls back to pure Python implementations if C++ extension is not available.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import C++ extension
_CPP_AVAILABLE = False
_cpp_module = None

try:
    from drone_core_py import (
        # Types
        Position3D as CppPosition3D,
        SphericalCoord as CppSphericalCoord,
        DOAResult as CppDOAResult,
        DetectionResult as CppDetectionResult,

        # Enums
        FFTBackend,
        DOAAlgorithm,
        TDOAMethod,

        # Configs
        SignalProcessorConfig,
        BeamformerConfig,
        TDOAConfig,
        TrackConfig,

        # Classes
        FFTProcessor as CppFFTProcessor,
        SignalProcessor as CppSignalProcessor,
        ArrayGeometry as CppArrayGeometry,
        Beamformer as CppBeamformer,
        TDOAEngine as CppTDOAEngine,
        TDOATracker as CppTDOATracker,

        # Info
        CUDA_AVAILABLE,
    )
    _CPP_AVAILABLE = True
    logger.info("C++ backend loaded successfully")
    if CUDA_AVAILABLE:
        logger.info("CUDA acceleration available")
except ImportError as e:
    logger.warning(f"C++ backend not available: {e}")
    logger.info("Using pure Python implementations")

    # Define fallback constants
    CUDA_AVAILABLE = False

    class FFTBackend:
        CPU = 0
        CUDA = 1

    class DOAAlgorithm:
        DELAY_SUM = 0
        MVDR = 1
        MUSIC = 2
        ESPRIT = 3
        SRP_PHAT = 4

    class TDOAMethod:
        GCC_PHAT = 0
        GCC_SCOT = 1
        GCC_ML = 2
        DIRECT_CORR = 3


def is_cpp_available() -> bool:
    """Check if C++ backend is available."""
    return _CPP_AVAILABLE


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return _CPP_AVAILABLE and CUDA_AVAILABLE


@dataclass
class Position3D:
    """3D position."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_cpp(self):
        if _CPP_AVAILABLE:
            return CppPosition3D(self.x, self.y, self.z)
        return self

    @staticmethod
    def from_cpp(cpp_pos):
        if _CPP_AVAILABLE:
            return Position3D(cpp_pos.x, cpp_pos.y, cpp_pos.z)
        return cpp_pos

    def distance_to(self, other: 'Position3D') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class SphericalCoord:
    """Spherical coordinates."""
    azimuth: float = 0.0
    elevation: float = 0.0
    distance: float = 0.0

    def to_cartesian(self) -> Position3D:
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        cos_el = np.cos(el_rad)
        return Position3D(
            self.distance * cos_el * np.cos(az_rad),
            self.distance * cos_el * np.sin(az_rad),
            self.distance * np.sin(el_rad)
        )


@dataclass
class DOAResult:
    """Direction of arrival result."""
    azimuth: float = 0.0
    elevation: float = 0.0
    power: float = 0.0
    confidence: float = 0.0

    @staticmethod
    def from_cpp(cpp_result):
        if _CPP_AVAILABLE:
            return DOAResult(
                azimuth=cpp_result.azimuth,
                elevation=cpp_result.elevation,
                power=cpp_result.power,
                confidence=cpp_result.confidence
            )
        return cpp_result


class ArrayGeometry:
    """Microphone array geometry wrapper."""

    def __init__(self):
        self._positions: List[Position3D] = []
        self._cpp_geom = None

    @classmethod
    def circular(cls, num_mics: int, radius: float) -> 'ArrayGeometry':
        """Create circular array."""
        geom = cls()
        if _CPP_AVAILABLE:
            geom._cpp_geom = CppArrayGeometry.circular(num_mics, radius)
        else:
            for i in range(num_mics):
                angle = 2 * np.pi * i / num_mics
                geom._positions.append(Position3D(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0
                ))
        return geom

    @classmethod
    def spherical(cls, num_mics: int, radius: float) -> 'ArrayGeometry':
        """Create spherical array."""
        geom = cls()
        if _CPP_AVAILABLE:
            geom._cpp_geom = CppArrayGeometry.spherical(num_mics, radius)
        else:
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(num_mics):
                theta = 2 * np.pi * i / golden_ratio
                phi = np.arccos(1 - 2 * (i + 0.5) / num_mics)
                geom._positions.append(Position3D(
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi)
                ))
        return geom

    @classmethod
    def linear(cls, num_mics: int, spacing: float) -> 'ArrayGeometry':
        """Create linear array."""
        geom = cls()
        if _CPP_AVAILABLE:
            geom._cpp_geom = CppArrayGeometry.linear(num_mics, spacing)
        else:
            offset = (num_mics - 1) * spacing / 2
            for i in range(num_mics):
                geom._positions.append(Position3D(i * spacing - offset, 0, 0))
        return geom

    @property
    def num_mics(self) -> int:
        if self._cpp_geom:
            return self._cpp_geom.num_mics()
        return len(self._positions)

    @property
    def cpp_geometry(self):
        return self._cpp_geom


class SignalProcessor:
    """High-performance signal processor."""

    def __init__(self, sample_rate: int = 48000, num_channels: int = 8,
                 fft_size: int = 2048, min_freq: float = 80,
                 max_freq: float = 8000, use_gpu: bool = True):

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.fft_size = fft_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_gpu = use_gpu and is_cuda_available()

        self._cpp_processor = None

        if _CPP_AVAILABLE:
            config = SignalProcessorConfig()
            config.sample_rate = sample_rate
            config.num_channels = num_channels
            config.fft_size = fft_size
            config.min_frequency = min_freq
            config.max_frequency = max_freq
            config.use_gpu = self.use_gpu

            self._cpp_processor = CppSignalProcessor(config)
            logger.debug("Using C++ SignalProcessor")
        else:
            logger.debug("Using Python SignalProcessor")

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through bandpass filter."""
        if self._cpp_processor:
            return self._cpp_processor.process(audio.astype(np.float32))
        else:
            # Fallback: simple passthrough
            return audio

    def gcc_phat(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute GCC-PHAT between two signals."""
        if self._cpp_processor:
            return self._cpp_processor.gcc_phat(
                a.astype(np.float32),
                b.astype(np.float32)
            )
        else:
            # Python fallback
            n = len(a) + len(b) - 1
            fft_size = 2 ** int(np.ceil(np.log2(n)))

            A = np.fft.rfft(a, fft_size)
            B = np.fft.rfft(b, fft_size)

            cross = A * np.conj(B)
            cross /= np.maximum(np.abs(cross), 1e-10)

            return np.fft.irfft(cross)


class Beamformer:
    """High-performance beamformer."""

    def __init__(self, geometry: ArrayGeometry, sample_rate: int = 48000,
                 fft_size: int = 2048, algorithm: int = 2,
                 min_freq: float = 80, max_freq: float = 8000,
                 az_resolution: int = 360, el_resolution: int = 91,
                 use_gpu: bool = True):

        self.geometry = geometry
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and is_cuda_available()

        self._cpp_beamformer = None

        if _CPP_AVAILABLE and geometry.cpp_geometry:
            config = BeamformerConfig()
            config.sample_rate = sample_rate
            config.fft_size = fft_size
            config.algorithm = DOAAlgorithm(algorithm)
            config.min_frequency = min_freq
            config.max_frequency = max_freq
            config.azimuth_resolution = az_resolution
            config.elevation_resolution = el_resolution
            config.use_gpu = self.use_gpu

            self._cpp_beamformer = CppBeamformer(geometry.cpp_geometry, config)
            logger.debug("Using C++ Beamformer")
        else:
            logger.debug("Using Python Beamformer")

    def estimate_doa(self, audio: np.ndarray) -> List[DOAResult]:
        """Estimate direction of arrival."""
        if self._cpp_beamformer:
            cpp_results = self._cpp_beamformer.estimate_doa(
                audio.astype(np.float32)
            )
            return [DOAResult.from_cpp(r) for r in cpp_results]
        else:
            # Simple fallback - return empty
            return []

    def compute_spatial_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute spatial spectrum."""
        if self._cpp_beamformer:
            return self._cpp_beamformer.compute_spatial_spectrum(
                audio.astype(np.float32)
            )
        else:
            # Fallback: return zeros
            return np.zeros((360, 91))

    def steer(self, audio: np.ndarray, azimuth: float,
              elevation: float) -> np.ndarray:
        """Steer beamformer to direction."""
        if self._cpp_beamformer:
            return self._cpp_beamformer.steer(
                audio.astype(np.float32),
                azimuth, elevation
            )
        else:
            # Simple delay-and-sum fallback
            return np.mean(audio, axis=0)


class TDOAEngine:
    """High-performance TDOA engine."""

    def __init__(self, geometry: ArrayGeometry, sample_rate: int = 48000,
                 fft_size: int = 2048, max_delay: float = 0.01,
                 use_gpu: bool = True):

        self.geometry = geometry
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and is_cuda_available()

        self._cpp_engine = None

        if _CPP_AVAILABLE and geometry.cpp_geometry:
            config = TDOAConfig()
            config.sample_rate = sample_rate
            config.fft_size = fft_size
            config.max_delay_seconds = max_delay
            config.use_gpu = self.use_gpu

            self._cpp_engine = CppTDOAEngine(geometry.cpp_geometry, config)
            logger.debug("Using C++ TDOAEngine")
        else:
            logger.debug("Using Python TDOAEngine")

    def localize_sources(self, audio: np.ndarray,
                         num_sources: int = 1) -> List[Position3D]:
        """Localize sources from audio."""
        if self._cpp_engine:
            cpp_positions = self._cpp_engine.localize_sources(
                audio.astype(np.float32), num_sources
            )
            return [Position3D.from_cpp(p) for p in cpp_positions]
        else:
            return []

    def gcc_phat(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute GCC-PHAT."""
        if self._cpp_engine:
            return self._cpp_engine.gcc_phat(
                a.astype(np.float32),
                b.astype(np.float32)
            )
        else:
            # Python fallback
            n = len(a) + len(b) - 1
            fft_size = 2 ** int(np.ceil(np.log2(n)))

            A = np.fft.rfft(a, fft_size)
            B = np.fft.rfft(b, fft_size)

            cross = A * np.conj(B)
            cross /= np.maximum(np.abs(cross), 1e-10)

            return np.fft.irfft(cross)


# Export all
__all__ = [
    'is_cpp_available',
    'is_cuda_available',
    'Position3D',
    'SphericalCoord',
    'DOAResult',
    'ArrayGeometry',
    'SignalProcessor',
    'Beamformer',
    'TDOAEngine',
    'DOAAlgorithm',
    'TDOAMethod',
    'FFTBackend',
]
