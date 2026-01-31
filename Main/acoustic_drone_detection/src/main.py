"""
Acoustic Drone Detection System - Main Entry Point

This module provides the main entry point for the drone detection system,
integrating all components into a unified detection pipeline.
"""

import sys
import time
import threading
import argparse
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass

import numpy as np

# Import core components
from core.audio_capture import AudioCapture
from core.signal_processor import SignalProcessor, FilterConfig
from core.beamforming import BeamformingEngine, BeamformingConfig
from core.localization import LocalizationEngine, LocalizationConfig
from core.tracking import MultiTargetTracker, TrackingConfig, Detection

# Import detection components
from detection.detector import DroneDetector, DetectorConfig
from detection.classifier import DroneClassifier, ClassifierConfig

# Import hardware components
from hardware.array_geometry import ArrayGeometryManager
from hardware.gpu_detector import get_gpu_manager

# Import utilities
from utils.config_manager import ConfigManager, get_config
from utils.logger import setup_logging, get_logger
from utils.data_export import DataExportManager, create_detection_record
from utils.performance_monitor import PerformanceMonitor, TimingContext


@dataclass
class DetectionEvent:
    """Complete detection event with all information."""
    timestamp: float
    azimuth: float
    elevation: float
    distance: float
    confidence: float
    classification: str
    threat_level: str
    track_id: Optional[int] = None
    snr: float = 0.0
    dominant_frequencies: List[float] = None


class DroneDetectionSystem:
    """
    Main drone detection system class.

    Integrates audio capture, signal processing, DOA estimation,
    localization, tracking, and classification into a unified pipeline.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        array_config_path: Optional[str] = None
    ):
        """
        Initialize the drone detection system.

        Args:
            config_path: Path to main configuration file.
            array_config_path: Path to array geometry configuration.
        """
        # Load configuration
        self._config_manager = ConfigManager(config_path)
        config = self._config_manager.config

        # Setup logging
        self._logger = setup_logging(
            level=20,  # INFO
            log_dir=config.recording.save_location.replace('recordings', 'logs')
        )
        self._logger.info("Initializing Drone Detection System")

        # Initialize GPU manager
        self._gpu_manager = get_gpu_manager()
        if self._gpu_manager.is_gpu_available:
            gpu_info = self._gpu_manager.get_best_device()
            self._logger.info(f"GPU detected: {gpu_info.name if gpu_info else 'Unknown'}")
        else:
            self._logger.info("No GPU detected, using CPU")

        # Initialize array geometry
        self._geometry_manager = ArrayGeometryManager()
        if array_config_path:
            self._array = self._geometry_manager.load(array_config_path)
        else:
            self._array = self._geometry_manager.create_circular(
                num_mics=config.audio.num_channels,
                radius=config.array.mic_spacing
            )
        self._logger.info(f"Array geometry: {self._array.name}")

        # Get microphone positions
        mic_positions = self._array.get_positions_array()

        # Initialize audio capture
        self._audio_capture = AudioCapture(
            num_channels=config.audio.num_channels,
            sample_rate=config.audio.sample_rate,
            buffer_size=config.audio.buffer_size,
            device_index=config.audio.device_index
        )

        # Initialize signal processor
        filter_config = FilterConfig(
            bandpass_low=config.detection.freq_min,
            bandpass_high=config.detection.freq_max
        )
        self._signal_processor = SignalProcessor(
            sample_rate=config.audio.sample_rate,
            filter_config=filter_config
        )

        # Initialize beamforming engine
        beamforming_config = BeamformingConfig(
            num_azimuth=360,
            num_elevation=91
        )
        self._beamformer = BeamformingEngine(
            mic_positions=mic_positions,
            sample_rate=config.audio.sample_rate,
            config=beamforming_config
        )

        # Initialize localization engine
        localization_config = LocalizationConfig(
            max_distance=config.detection.max_detection_range
        )
        self._localizer = LocalizationEngine(
            mic_positions=mic_positions,
            config=localization_config
        )

        # Initialize tracker
        tracking_config = TrackingConfig(
            filter_type=config.tracking.algorithm,
            max_tracks=config.tracking.max_targets,
            track_timeout=config.tracking.track_timeout
        )
        self._tracker = MultiTargetTracker(config=tracking_config)

        # Initialize detector and classifier
        detector_config = DetectorConfig(
            min_confidence=config.detection.min_confidence,
            detection_band=(config.detection.freq_min, config.detection.freq_max)
        )
        self._detector = DroneDetector(
            sample_rate=config.audio.sample_rate,
            config=detector_config
        )
        self._classifier = DroneClassifier(
            sample_rate=config.audio.sample_rate
        )

        # Initialize data export
        self._export_manager = DataExportManager(
            export_dir=config.recording.save_location.replace('recordings', 'exports')
        )

        # Initialize performance monitor
        self._perf_monitor = PerformanceMonitor()

        # State
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[DetectionEvent], None]] = []
        self._lock = threading.Lock()

        self._logger.info("Drone Detection System initialized")

    def start(self) -> bool:
        """
        Start the detection system.

        Returns:
            True if started successfully.
        """
        with self._lock:
            if self._running:
                return True

            # Start audio capture
            if not self._audio_capture.start():
                self._logger.error("Failed to start audio capture")
                return False

            # Start performance monitoring
            self._perf_monitor.start()

            # Start detection thread
            self._running = True
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True
            )
            self._detection_thread.start()

            self._logger.info("Detection system started")
            return True

    def stop(self) -> None:
        """Stop the detection system."""
        with self._lock:
            if not self._running:
                return

            self._running = False

        # Wait for detection thread
        if self._detection_thread:
            self._detection_thread.join(timeout=2.0)

        # Stop components
        self._audio_capture.stop()
        self._perf_monitor.stop()

        self._logger.info("Detection system stopped")

    def _detection_loop(self) -> None:
        """Main detection processing loop."""
        config = self._config_manager.config
        buffer_samples = int(0.1 * config.audio.sample_rate)  # 100ms buffer

        while self._running:
            try:
                # Read audio data
                with TimingContext(self._perf_monitor, "audio_read"):
                    audio_data = self._audio_capture.read(
                        samples=buffer_samples,
                        timeout=0.5
                    )

                if audio_data is None:
                    continue

                # Record frame for FPS calculation
                self._perf_monitor.record_frame()

                # Process audio
                with TimingContext(self._perf_monitor, "processing"):
                    detection_event = self._process_frame(audio_data)

                # Notify callbacks
                if detection_event and detection_event.confidence >= config.detection.min_confidence:
                    self._notify_callbacks(detection_event)

                    # Export detection
                    record = create_detection_record(
                        timestamp=detection_event.timestamp,
                        azimuth=detection_event.azimuth,
                        elevation=detection_event.elevation,
                        distance=detection_event.distance,
                        confidence=detection_event.confidence,
                        snr=detection_event.snr,
                        classification=detection_event.classification,
                        threat_level=detection_event.threat_level,
                        dominant_frequencies=detection_event.dominant_frequencies or [],
                        track_id=detection_event.track_id
                    )
                    self._export_manager.add_detection(record)

            except Exception as e:
                self._logger.error(f"Detection loop error: {e}")

    def _process_frame(self, audio_data: np.ndarray) -> Optional[DetectionEvent]:
        """
        Process a single audio frame.

        Args:
            audio_data: Multi-channel audio data.

        Returns:
            DetectionEvent or None if no detection.
        """
        timestamp = time.time()

        # Preprocess audio
        filtered = self._signal_processor.preprocess(audio_data)

        # Mono signal for detection
        mono = filtered.mean(axis=0) if filtered.ndim > 1 else filtered

        # Detection
        detection_result = self._detector.detect(mono, timestamp)

        if detection_result.status.value == "no_detection":
            return None

        # DOA estimation
        doa_result = self._beamformer.estimate_doa(filtered)

        # Localization
        signal_rms = np.sqrt(np.mean(mono ** 2))
        loc_result = self._localizer.localize_from_doa(
            doa_result.azimuth,
            doa_result.elevation,
            signal_rms
        )

        # Classification
        class_result = self._classifier.classify(
            mono,
            detection_result.dominant_frequencies,
            timestamp
        )

        # Create tracking detection
        track_detection = Detection(
            timestamp=timestamp,
            azimuth=loc_result.azimuth,
            elevation=loc_result.elevation,
            distance=loc_result.distance,
            confidence=detection_result.confidence,
            classification=class_result.primary_class.value
        )

        # Update tracker
        tracks = self._tracker.update([track_detection])

        # Get track ID if associated
        track_id = None
        if tracks:
            for track in tracks:
                if track.state.confidence > 0.5:
                    track_id = track.track_id
                    break

        # Assess threat level
        threat_level = self._assess_threat(loc_result.distance, detection_result.confidence)

        # Log detection
        self._logger.detection(
            azimuth=loc_result.azimuth,
            elevation=loc_result.elevation,
            distance=loc_result.distance,
            confidence=detection_result.confidence,
            classification=class_result.primary_class.value,
            threat_level=threat_level
        )

        return DetectionEvent(
            timestamp=timestamp,
            azimuth=loc_result.azimuth,
            elevation=loc_result.elevation,
            distance=loc_result.distance,
            confidence=detection_result.confidence,
            classification=class_result.primary_class.value,
            threat_level=threat_level,
            track_id=track_id,
            snr=detection_result.snr,
            dominant_frequencies=detection_result.dominant_frequencies.tolist()
            if len(detection_result.dominant_frequencies) > 0 else []
        )

    def _assess_threat(self, distance: float, confidence: float) -> str:
        """Assess threat level based on distance and confidence."""
        if distance < 50 and confidence > 0.7:
            return "high"
        elif distance < 150 and confidence > 0.5:
            return "medium"
        return "low"

    def _notify_callbacks(self, event: DetectionEvent) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(f"Callback error: {e}")

    def add_callback(self, callback: Callable[[DetectionEvent], None]) -> None:
        """Add a detection callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a detection callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._running

    @property
    def performance_metrics(self):
        """Get current performance metrics."""
        return self._perf_monitor.metrics

    def get_tracks(self):
        """Get current tracks."""
        return self._tracker.get_tracks()

    def export_data(self, prefix: str = "detection") -> dict:
        """Export accumulated detection data."""
        return self._export_manager.export_all(prefix)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Acoustic Drone Detection System"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "-a", "--array",
        help="Path to array geometry file",
        default=None
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI interface"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.gui:
        # Launch GUI
        try:
            from gui.main_window import main as gui_main
            gui_main()
        except ImportError:
            print("GUI dependencies not installed. Install with: pip install acoustic-drone-detection[gui]")
            sys.exit(1)
    else:
        # Command-line mode
        print("Acoustic Drone Detection System")
        print("================================")

        system = DroneDetectionSystem(
            config_path=args.config,
            array_config_path=args.array
        )

        def detection_callback(event: DetectionEvent):
            print(f"[{event.threat_level.upper()}] {event.classification} detected at "
                  f"Az: {event.azimuth:.1f}deg, El: {event.elevation:.1f}deg, "
                  f"Dist: {event.distance:.1f}m, Conf: {event.confidence:.1%}")

        system.add_callback(detection_callback)

        print("Starting detection... Press Ctrl+C to stop.")

        try:
            system.start()

            while True:
                time.sleep(1)
                metrics = system.performance_metrics
                print(f"\rFPS: {metrics.fps:.1f} | CPU: {metrics.cpu_percent:.1f}% | "
                      f"Latency: {metrics.processing_latency_ms:.1f}ms", end="")

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            system.stop()
            exported = system.export_data()
            if exported:
                print(f"Data exported to: {exported}")


if __name__ == "__main__":
    main()
