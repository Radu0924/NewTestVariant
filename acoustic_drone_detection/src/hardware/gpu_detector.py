"""
GPU Detection and Management Module

Provides GPU hardware detection and management:
- NVIDIA CUDA detection
- AMD ROCm detection (optional)
- GPU memory management
- Compute capability checking
"""

import os
import sys
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading


class GPUBackend(Enum):
    """Available GPU backends."""
    NONE = "none"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    backend: GPUBackend
    compute_capability: Tuple[int, int] = (0, 0)
    total_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    driver_version: str = ""
    cuda_version: str = ""
    is_available: bool = True


class CUDADetector:
    """
    NVIDIA CUDA GPU detector.

    Detects NVIDIA GPUs and their capabilities using pynvml.
    """

    def __init__(self):
        """Initialize CUDA detector."""
        self._nvml_available = False
        self._device_count = 0
        self._initialized = False
        self._pynvml = None

        self._try_init()

    def _try_init(self) -> None:
        """Try to initialize NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_available = True
            self._initialized = True
            self._pynvml = pynvml
        except ImportError:
            self._nvml_available = False
        except Exception as e:
            self._nvml_available = False

    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._nvml_available

    @property
    def device_count(self) -> int:
        """Get number of CUDA devices."""
        return self._device_count

    def get_device_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """
        Get information about a CUDA device.

        Args:
            device_id: GPU device index.

        Returns:
            GPUInfo object or None if not available.
        """
        if not self._nvml_available or device_id >= self._device_count:
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Device name
            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Memory info
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = memory.total / (1024 ** 2)
            free_mb = memory.free / (1024 ** 2)

            # Compute capability
            major = self._pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
            minor = self._pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]

            # Driver version
            driver_version = self._pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')

            # CUDA version
            cuda_version = self._pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            cuda_version_str = f"{cuda_major}.{cuda_minor}"

            return GPUInfo(
                device_id=device_id,
                name=name,
                backend=GPUBackend.CUDA,
                compute_capability=(major, minor),
                total_memory_mb=total_mb,
                free_memory_mb=free_mb,
                driver_version=driver_version,
                cuda_version=cuda_version_str,
                is_available=True
            )

        except Exception as e:
            return None

    def get_utilization(self, device_id: int = 0) -> Tuple[float, float]:
        """
        Get GPU utilization percentages.

        Args:
            device_id: GPU device index.

        Returns:
            Tuple of (gpu_percent, memory_percent).
        """
        if not self._nvml_available or device_id >= self._device_count:
            return (0.0, 0.0)

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_util = float(utilization.gpu)
            mem_util = (memory.used / memory.total) * 100

            return (gpu_util, mem_util)

        except Exception:
            return (0.0, 0.0)

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
                self._initialized = False
            except Exception:
                pass


class PyTorchGPUDetector:
    """
    GPU detector using PyTorch.

    Provides GPU detection using PyTorch's CUDA interface.
    """

    def __init__(self):
        """Initialize PyTorch GPU detector."""
        self._torch_available = False
        self._cuda_available = False
        self._device_count = 0
        self._torch = None

        self._try_init()

    def _try_init(self) -> None:
        """Try to initialize PyTorch."""
        try:
            import torch
            self._torch = torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()

            if self._cuda_available:
                self._device_count = torch.cuda.device_count()

        except ImportError:
            self._torch_available = False

    @property
    def is_available(self) -> bool:
        """Check if CUDA is available through PyTorch."""
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get number of CUDA devices."""
        return self._device_count

    def get_device_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """
        Get information about a CUDA device.

        Args:
            device_id: GPU device index.

        Returns:
            GPUInfo object or None.
        """
        if not self._cuda_available or device_id >= self._device_count:
            return None

        try:
            name = self._torch.cuda.get_device_name(device_id)
            props = self._torch.cuda.get_device_properties(device_id)

            total_memory = props.total_memory / (1024 ** 2)
            free_memory = self._torch.cuda.memory_reserved(device_id) / (1024 ** 2)

            return GPUInfo(
                device_id=device_id,
                name=name,
                backend=GPUBackend.CUDA,
                compute_capability=(props.major, props.minor),
                total_memory_mb=total_memory,
                free_memory_mb=total_memory - free_memory,
                is_available=True
            )

        except Exception:
            return None

    def get_current_memory_usage(self, device_id: int = 0) -> Dict[str, float]:
        """
        Get current memory usage.

        Args:
            device_id: GPU device index.

        Returns:
            Dictionary with memory usage info.
        """
        if not self._cuda_available:
            return {}

        try:
            self._torch.cuda.set_device(device_id)
            return {
                'allocated_mb': self._torch.cuda.memory_allocated(device_id) / (1024 ** 2),
                'reserved_mb': self._torch.cuda.memory_reserved(device_id) / (1024 ** 2),
                'max_allocated_mb': self._torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
            }
        except Exception:
            return {}


class CuPyDetector:
    """
    GPU detector using CuPy.

    Provides GPU detection for signal processing operations.
    """

    def __init__(self):
        """Initialize CuPy detector."""
        self._cupy_available = False
        self._device_count = 0
        self._cupy = None

        self._try_init()

    def _try_init(self) -> None:
        """Try to initialize CuPy."""
        try:
            import cupy as cp
            self._cupy = cp
            self._cupy_available = True
            self._device_count = cp.cuda.runtime.getDeviceCount()
        except ImportError:
            self._cupy_available = False
        except Exception:
            self._cupy_available = False

    @property
    def is_available(self) -> bool:
        """Check if CuPy is available."""
        return self._cupy_available

    @property
    def device_count(self) -> int:
        """Get number of CUDA devices."""
        return self._device_count

    def get_device_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get information about a CUDA device."""
        if not self._cupy_available or device_id >= self._device_count:
            return None

        try:
            with self._cupy.cuda.Device(device_id):
                props = self._cupy.cuda.runtime.getDeviceProperties(device_id)
                mem_info = self._cupy.cuda.runtime.memGetInfo()

            name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']

            return GPUInfo(
                device_id=device_id,
                name=name,
                backend=GPUBackend.CUDA,
                compute_capability=(props['major'], props['minor']),
                total_memory_mb=props['totalGlobalMem'] / (1024 ** 2),
                free_memory_mb=mem_info[0] / (1024 ** 2),
                is_available=True
            )

        except Exception:
            return None


class GPUManager:
    """
    Unified GPU management system.

    Provides a single interface for GPU detection and management
    across different backends.
    """

    def __init__(self, prefer_backend: Optional[GPUBackend] = None):
        """
        Initialize GPU manager.

        Args:
            prefer_backend: Preferred GPU backend.
        """
        self._prefer_backend = prefer_backend or GPUBackend.CUDA
        self._cuda_detector = CUDADetector()
        self._pytorch_detector = PyTorchGPUDetector()
        self._cupy_detector = CuPyDetector()

        self._current_device: int = 0
        self._mode: str = "auto"  # auto, force_gpu, force_cpu
        self._lock = threading.Lock()

        self._detect()

    def _detect(self) -> None:
        """Detect available GPUs."""
        self._devices: List[GPUInfo] = []

        # Try NVML first (most accurate for NVIDIA)
        if self._cuda_detector.is_available:
            for i in range(self._cuda_detector.device_count):
                info = self._cuda_detector.get_device_info(i)
                if info:
                    self._devices.append(info)

        # Fall back to PyTorch if NVML not available
        elif self._pytorch_detector.is_available:
            for i in range(self._pytorch_detector.device_count):
                info = self._pytorch_detector.get_device_info(i)
                if info:
                    self._devices.append(info)

    def refresh(self) -> None:
        """Refresh device list."""
        with self._lock:
            self._detect()

    @property
    def is_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        return len(self._devices) > 0

    @property
    def device_count(self) -> int:
        """Get number of available GPUs."""
        return len(self._devices)

    @property
    def devices(self) -> List[GPUInfo]:
        """Get list of available GPUs."""
        return self._devices.copy()

    def get_device(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get info for specific device."""
        if device_id < len(self._devices):
            return self._devices[device_id]
        return None

    def get_best_device(self) -> Optional[GPUInfo]:
        """
        Get the best available GPU.

        Selects based on compute capability and available memory.

        Returns:
            GPUInfo for best device or None.
        """
        if not self._devices:
            return None

        # Sort by compute capability, then by free memory
        sorted_devices = sorted(
            self._devices,
            key=lambda d: (
                d.compute_capability[0] * 10 + d.compute_capability[1],
                d.free_memory_mb
            ),
            reverse=True
        )

        return sorted_devices[0]

    def get_utilization(self, device_id: int = 0) -> Tuple[float, float]:
        """
        Get GPU utilization.

        Args:
            device_id: Device index.

        Returns:
            Tuple of (gpu_percent, memory_percent).
        """
        if self._cuda_detector.is_available:
            return self._cuda_detector.get_utilization(device_id)
        return (0.0, 0.0)

    def set_mode(self, mode: str) -> None:
        """
        Set GPU mode.

        Args:
            mode: One of 'auto', 'force_gpu', 'force_cpu'.
        """
        if mode not in ['auto', 'force_gpu', 'force_cpu']:
            raise ValueError(f"Invalid mode: {mode}")

        with self._lock:
            self._mode = mode

    @property
    def mode(self) -> str:
        """Get current GPU mode."""
        return self._mode

    def should_use_gpu(self) -> bool:
        """
        Determine if GPU should be used.

        Returns:
            True if GPU should be used based on mode and availability.
        """
        if self._mode == 'force_cpu':
            return False
        elif self._mode == 'force_gpu':
            if not self.is_gpu_available:
                raise RuntimeError("GPU forced but no GPU available")
            return True
        else:  # auto
            return self.is_gpu_available

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about available backends.

        Returns:
            Dictionary with backend availability info.
        """
        return {
            'nvml_available': self._cuda_detector.is_available,
            'pytorch_cuda_available': self._pytorch_detector.is_available,
            'cupy_available': self._cupy_detector.is_available,
            'device_count': self.device_count,
            'mode': self._mode,
            'should_use_gpu': self.should_use_gpu() if self._mode != 'force_gpu' or self.is_gpu_available else False
        }

    def shutdown(self) -> None:
        """Shutdown GPU management."""
        self._cuda_detector.shutdown()


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return get_gpu_manager().is_gpu_available


def should_use_gpu() -> bool:
    """Check if GPU should be used based on configuration."""
    return get_gpu_manager().should_use_gpu()
