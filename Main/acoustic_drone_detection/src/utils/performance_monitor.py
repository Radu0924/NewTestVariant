"""
Performance Monitoring Module

Provides real-time performance monitoring including:
- CPU usage tracking
- Memory usage tracking
- GPU usage (if available)
- Processing latency measurement
- FPS calculation
"""

import time
import threading
import psutil
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque
import os


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    processing_latency_ms: float = 0.0
    fps: float = 0.0
    active_threads: int = 0
    queue_size: int = 0


@dataclass
class LatencyTracker:
    """Tracks latency statistics for an operation."""
    name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency_ms)

    @property
    def average(self) -> float:
        """Get average latency."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def min(self) -> float:
        """Get minimum latency."""
        return min(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        """Get maximum latency."""
        return max(self.samples) if self.samples else 0.0

    @property
    def percentile_95(self) -> float:
        """Get 95th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]


class GPUMonitor:
    """
    GPU performance monitor.

    Supports NVIDIA GPUs via pynvml if available.
    """

    def __init__(self):
        """Initialize GPU monitor."""
        self._nvml_available = False
        self._device_count = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_available = True
            self._pynvml = pynvml
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._nvml_available

    @property
    def device_count(self) -> int:
        """Get number of available GPUs."""
        return self._device_count

    def get_gpu_info(self, device_id: int = 0) -> Dict:
        """
        Get GPU information.

        Args:
            device_id: GPU device ID.

        Returns:
            Dictionary with GPU information.
        """
        if not self._nvml_available or device_id >= self._device_count:
            return {}

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                'name': name,
                'memory_total_mb': memory.total / (1024 ** 2),
                'memory_used_mb': memory.used / (1024 ** 2),
                'memory_free_mb': memory.free / (1024 ** 2),
                'memory_percent': (memory.used / memory.total) * 100,
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory
            }

        except Exception as e:
            return {'error': str(e)}

    def get_utilization(self, device_id: int = 0) -> tuple[float, float]:
        """
        Get GPU utilization percentages.

        Args:
            device_id: GPU device ID.

        Returns:
            Tuple of (gpu_percent, memory_percent).
        """
        info = self.get_gpu_info(device_id)
        return (
            info.get('gpu_utilization', 0.0),
            info.get('memory_percent', 0.0)
        )

    def shutdown(self) -> None:
        """Shutdown GPU monitoring."""
        if self._nvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class PerformanceMonitor:
    """
    Comprehensive performance monitor for the drone detection system.

    Tracks CPU, memory, GPU usage, and processing latencies.
    """

    def __init__(
        self,
        update_interval: float = 1.0,
        history_size: int = 100
    ):
        """
        Initialize the performance monitor.

        Args:
            update_interval: Update interval in seconds.
            history_size: Number of historical samples to keep.
        """
        self._update_interval = update_interval
        self._history_size = history_size

        self._metrics = PerformanceMetrics()
        self._metrics_history: deque = deque(maxlen=history_size)
        self._latency_trackers: Dict[str, LatencyTracker] = {}

        self._process = psutil.Process(os.getpid())
        self._gpu_monitor = GPUMonitor()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._fps_samples: deque = deque(maxlen=30)
        self._last_frame_time: Optional[float] = None

        self._observers: List[Callable] = []

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._update_metrics()
                time.sleep(self._update_interval)
            except Exception as e:
                print(f"Performance monitor error: {e}")

    def _update_metrics(self) -> None:
        """Update all performance metrics."""
        with self._lock:
            # CPU usage
            self._metrics.cpu_percent = self._process.cpu_percent()

            # Memory usage
            memory_info = self._process.memory_info()
            self._metrics.memory_mb = memory_info.rss / (1024 ** 2)
            self._metrics.memory_percent = self._process.memory_percent()

            # GPU usage
            if self._gpu_monitor.available:
                gpu_util, gpu_mem = self._gpu_monitor.get_utilization()
                self._metrics.gpu_percent = gpu_util
                self._metrics.gpu_memory_percent = gpu_mem

            # Thread count
            self._metrics.active_threads = threading.active_count()

            # Store history
            self._metrics_history.append(PerformanceMetrics(
                cpu_percent=self._metrics.cpu_percent,
                memory_percent=self._metrics.memory_percent,
                memory_mb=self._metrics.memory_mb,
                gpu_percent=self._metrics.gpu_percent,
                gpu_memory_percent=self._metrics.gpu_memory_percent,
                processing_latency_ms=self._metrics.processing_latency_ms,
                fps=self._metrics.fps,
                active_threads=self._metrics.active_threads,
                queue_size=self._metrics.queue_size
            ))

        # Notify observers
        for observer in self._observers:
            try:
                observer(self._metrics)
            except Exception:
                pass

    def record_frame(self) -> None:
        """Record a frame for FPS calculation."""
        current_time = time.time()

        if self._last_frame_time is not None:
            frame_time = current_time - self._last_frame_time
            if frame_time > 0:
                self._fps_samples.append(1.0 / frame_time)

        self._last_frame_time = current_time

        with self._lock:
            if self._fps_samples:
                self._metrics.fps = sum(self._fps_samples) / len(self._fps_samples)

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Operation name.
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            if operation not in self._latency_trackers:
                self._latency_trackers[operation] = LatencyTracker(operation)

            self._latency_trackers[operation].add_sample(latency_ms)

            # Update main processing latency if this is the main operation
            if operation == "processing":
                self._metrics.processing_latency_ms = latency_ms

    def set_queue_size(self, size: int) -> None:
        """
        Set current queue size.

        Args:
            size: Queue size.
        """
        with self._lock:
            self._metrics.queue_size = size

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return PerformanceMetrics(
                cpu_percent=self._metrics.cpu_percent,
                memory_percent=self._metrics.memory_percent,
                memory_mb=self._metrics.memory_mb,
                gpu_percent=self._metrics.gpu_percent,
                gpu_memory_percent=self._metrics.gpu_memory_percent,
                processing_latency_ms=self._metrics.processing_latency_ms,
                fps=self._metrics.fps,
                active_threads=self._metrics.active_threads,
                queue_size=self._metrics.queue_size
            )

    def get_latency_stats(self, operation: str) -> Dict:
        """
        Get latency statistics for an operation.

        Args:
            operation: Operation name.

        Returns:
            Dictionary with latency statistics.
        """
        with self._lock:
            tracker = self._latency_trackers.get(operation)
            if not tracker:
                return {}

            return {
                'average_ms': tracker.average,
                'min_ms': tracker.min,
                'max_ms': tracker.max,
                'p95_ms': tracker.percentile_95,
                'samples': len(tracker.samples)
            }

    def get_all_latency_stats(self) -> Dict[str, Dict]:
        """
        Get latency statistics for all operations.

        Returns:
            Dictionary mapping operation to statistics.
        """
        with self._lock:
            return {
                name: {
                    'average_ms': tracker.average,
                    'min_ms': tracker.min,
                    'max_ms': tracker.max,
                    'p95_ms': tracker.percentile_95,
                    'samples': len(tracker.samples)
                }
                for name, tracker in self._latency_trackers.items()
            }

    def get_history(self) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        with self._lock:
            return list(self._metrics_history)

    def add_observer(self, callback: Callable) -> None:
        """Add an observer for metrics updates."""
        self._observers.append(callback)

    def remove_observer(self, callback: Callable) -> None:
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def get_system_info(self) -> Dict:
        """
        Get system information.

        Returns:
            Dictionary with system info.
        """
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'platform': os.name
        }

        if self._gpu_monitor.available:
            info['gpu'] = self._gpu_monitor.get_gpu_info()

        return info

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation: str
    ):
        """
        Initialize timing context.

        Args:
            monitor: Performance monitor instance.
            operation: Operation name.
        """
        self.monitor = monitor
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.monitor.record_latency(self.operation, elapsed_ms)
        return False


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def timing(operation: str):
    """
    Decorator for timing function execution.

    Args:
        operation: Operation name for the timing.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with TimingContext(monitor, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator
