"""
Batch Processor Module

Processes multiple audio files for offline analysis:
- Parallel processing support
- Progress tracking
- Result aggregation
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
from datetime import datetime

from .audio_loader import AudioLoader, AudioMetadata, AudioSegment


@dataclass
class DetectionEvent:
    """Detection event in a file."""
    timestamp: float
    file_time: float  # Time within the file
    azimuth: float
    elevation: float
    distance: float
    confidence: float
    classification: str
    threat_level: str
    dominant_frequencies: List[float] = field(default_factory=list)


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""
    filepath: str
    filename: str
    metadata: AudioMetadata
    detections: List[DetectionEvent]
    processing_time_seconds: float
    status: str  # success, error, no_detections
    error_message: Optional[str] = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


@dataclass
class BatchResult:
    """Result of batch processing."""
    files_processed: int
    files_with_detections: int
    total_detections: int
    total_processing_time: float
    results: List[FileAnalysisResult]
    start_time: str
    end_time: str

    def get_all_detections(self) -> List[tuple]:
        """Get all detections with file info."""
        all_detections = []
        for result in self.results:
            for det in result.detections:
                all_detections.append((result.filepath, det))
        return all_detections

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        classifications = {}
        threat_levels = {'high': 0, 'medium': 0, 'low': 0}

        for result in self.results:
            for det in result.detections:
                classifications[det.classification] = \
                    classifications.get(det.classification, 0) + 1
                if det.threat_level in threat_levels:
                    threat_levels[det.threat_level] += 1

        return {
            'files_processed': self.files_processed,
            'files_with_detections': self.files_with_detections,
            'total_detections': self.total_detections,
            'processing_time_seconds': self.total_processing_time,
            'classifications': classifications,
            'threat_levels': threat_levels,
            'detection_rate': self.files_with_detections / max(1, self.files_processed)
        }


class BatchProcessor:
    """
    Batch audio file processor.

    Processes multiple audio files for drone detection with
    parallel processing support.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        segment_duration: float = 1.0,
        overlap: float = 0.5,
        num_workers: int = 4,
        use_multiprocessing: bool = False
    ):
        """
        Initialize batch processor.

        Args:
            sample_rate: Target sample rate.
            segment_duration: Analysis segment duration in seconds.
            overlap: Segment overlap in seconds.
            num_workers: Number of parallel workers.
            use_multiprocessing: Use processes instead of threads.
        """
        self._sample_rate = sample_rate
        self._segment_duration = segment_duration
        self._overlap = overlap
        self._num_workers = num_workers
        self._use_multiprocessing = use_multiprocessing

        self._audio_loader = AudioLoader(target_sample_rate=sample_rate)
        self._detector = None
        self._classifier = None
        self._beamformer = None

        self._progress_callback: Optional[Callable] = None
        self._cancel_flag = threading.Event()

    def set_detection_pipeline(
        self,
        detector,
        classifier,
        beamformer=None
    ) -> None:
        """
        Set the detection pipeline components.

        Args:
            detector: DroneDetector instance.
            classifier: DroneClassifier instance.
            beamformer: Optional BeamformingEngine instance.
        """
        self._detector = detector
        self._classifier = classifier
        self._beamformer = beamformer

    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Set progress callback.

        Args:
            callback: Function(current, total, filename) called on progress.
        """
        self._progress_callback = callback

    def process_file(self, filepath: str) -> FileAnalysisResult:
        """
        Process a single audio file.

        Args:
            filepath: Path to audio file.

        Returns:
            FileAnalysisResult object.
        """
        start_time = time.time()
        detections = []

        try:
            # Load metadata
            metadata = self._audio_loader.get_metadata(filepath)

            # Process in segments
            for segment in self._audio_loader.iter_segments(
                filepath,
                self._segment_duration,
                self._overlap
            ):
                if self._cancel_flag.is_set():
                    break

                # Analyze segment
                segment_detections = self._analyze_segment(segment)
                detections.extend(segment_detections)

            processing_time = time.time() - start_time

            status = 'success' if detections else 'no_detections'

            return FileAnalysisResult(
                filepath=filepath,
                filename=Path(filepath).name,
                metadata=metadata,
                detections=detections,
                processing_time_seconds=processing_time,
                status=status
            )

        except Exception as e:
            return FileAnalysisResult(
                filepath=filepath,
                filename=Path(filepath).name,
                metadata=AudioMetadata(
                    filepath=filepath,
                    filename=Path(filepath).name,
                    format='',
                    sample_rate=0,
                    channels=0,
                    duration_seconds=0,
                    total_samples=0,
                    bit_depth=0,
                    file_size_bytes=0
                ),
                detections=[],
                processing_time_seconds=time.time() - start_time,
                status='error',
                error_message=str(e)
            )

    def _analyze_segment(self, segment: AudioSegment) -> List[DetectionEvent]:
        """Analyze a single audio segment."""
        if self._detector is None:
            return []

        detections = []

        # Get mono signal
        mono = segment.data.mean(axis=0) if segment.data.ndim > 1 else segment.data

        # Detection
        detection_result = self._detector.detect(mono, time.time())

        if detection_result.status.value == "no_detection":
            return []

        # DOA estimation
        azimuth, elevation = 0.0, 0.0
        if self._beamformer is not None:
            try:
                doa_result = self._beamformer.estimate_doa(segment.data)
                azimuth = doa_result.azimuth
                elevation = doa_result.elevation
            except Exception:
                pass

        # Classification
        classification = "unknown"
        if self._classifier is not None:
            try:
                class_result = self._classifier.classify(mono)
                classification = class_result.primary_class.value
            except Exception:
                pass

        # Distance estimation (simple)
        rms = np.sqrt(np.mean(mono ** 2))
        distance = max(1.0, min(500.0, 1.0 / (rms + 0.001)))

        # Threat assessment
        if distance < 50 and detection_result.confidence > 0.7:
            threat_level = "high"
        elif distance < 150 and detection_result.confidence > 0.5:
            threat_level = "medium"
        else:
            threat_level = "low"

        detections.append(DetectionEvent(
            timestamp=time.time(),
            file_time=segment.start_time,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            confidence=detection_result.confidence,
            classification=classification,
            threat_level=threat_level,
            dominant_frequencies=detection_result.dominant_frequencies.tolist()
            if len(detection_result.dominant_frequencies) > 0 else []
        ))

        return detections

    def process_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> BatchResult:
        """
        Process all audio files in a directory.

        Args:
            directory: Directory path.
            recursive: Process subdirectories.

        Returns:
            BatchResult with all results.
        """
        # Scan for files
        files = self._audio_loader.scan_directory(directory, recursive)
        filepaths = [f.filepath for f in files]

        return self.process_files(filepaths)

    def process_files(self, filepaths: List[str]) -> BatchResult:
        """
        Process a list of audio files.

        Args:
            filepaths: List of file paths.

        Returns:
            BatchResult with all results.
        """
        start_time = datetime.now()
        self._cancel_flag.clear()

        results = []
        total = len(filepaths)

        # Choose executor
        ExecutorClass = ProcessPoolExecutor if self._use_multiprocessing else ThreadPoolExecutor

        with ExecutorClass(max_workers=self._num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.process_file, fp): fp
                for fp in filepaths
            }

            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                if self._cancel_flag.is_set():
                    break

                filepath = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(FileAnalysisResult(
                        filepath=filepath,
                        filename=Path(filepath).name,
                        metadata=AudioMetadata(
                            filepath=filepath,
                            filename=Path(filepath).name,
                            format='',
                            sample_rate=0,
                            channels=0,
                            duration_seconds=0,
                            total_samples=0,
                            bit_depth=0,
                            file_size_bytes=0
                        ),
                        detections=[],
                        processing_time_seconds=0,
                        status='error',
                        error_message=str(e)
                    ))

                # Progress callback
                if self._progress_callback:
                    self._progress_callback(i + 1, total, Path(filepath).name)

        end_time = datetime.now()

        # Aggregate results
        files_with_detections = sum(1 for r in results if r.has_detections)
        total_detections = sum(r.detection_count for r in results)
        total_processing_time = sum(r.processing_time_seconds for r in results)

        return BatchResult(
            files_processed=len(results),
            files_with_detections=files_with_detections,
            total_detections=total_detections,
            total_processing_time=total_processing_time,
            results=results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )

    def cancel(self) -> None:
        """Cancel ongoing processing."""
        self._cancel_flag.set()

    def iter_process(
        self,
        filepaths: List[str]
    ) -> Generator[FileAnalysisResult, None, None]:
        """
        Process files yielding results as they complete.

        Args:
            filepaths: List of file paths.

        Yields:
            FileAnalysisResult for each file.
        """
        self._cancel_flag.clear()

        for filepath in filepaths:
            if self._cancel_flag.is_set():
                break

            yield self.process_file(filepath)


class AnalysisSession:
    """
    Analysis session for managing batch processing.

    Provides session management and result persistence.
    """

    def __init__(
        self,
        session_name: str,
        output_directory: str
    ):
        """
        Initialize analysis session.

        Args:
            session_name: Name for this session.
            output_directory: Directory for output files.
        """
        self.session_name = session_name
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.created_at = datetime.now()
        self.batch_results: List[BatchResult] = []

    def add_result(self, result: BatchResult) -> None:
        """Add a batch result to the session."""
        self.batch_results.append(result)

    def get_all_detections(self) -> List[tuple]:
        """Get all detections from all batches."""
        all_detections = []
        for batch in self.batch_results:
            all_detections.extend(batch.get_all_detections())
        return all_detections

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        total_files = sum(b.files_processed for b in self.batch_results)
        total_detections = sum(b.total_detections for b in self.batch_results)
        total_time = sum(b.total_processing_time for b in self.batch_results)

        return {
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat(),
            'total_batches': len(self.batch_results),
            'total_files': total_files,
            'total_detections': total_detections,
            'total_processing_time': total_time,
            'output_directory': str(self.output_directory)
        }

    def save(self) -> str:
        """
        Save session to file.

        Returns:
            Path to saved session file.
        """
        import json

        session_data = {
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat(),
            'batches': []
        }

        for batch in self.batch_results:
            batch_data = {
                'files_processed': batch.files_processed,
                'total_detections': batch.total_detections,
                'start_time': batch.start_time,
                'end_time': batch.end_time,
                'results': []
            }

            for result in batch.results:
                result_data = {
                    'filepath': result.filepath,
                    'status': result.status,
                    'detection_count': result.detection_count,
                    'detections': [
                        {
                            'file_time': d.file_time,
                            'azimuth': d.azimuth,
                            'elevation': d.elevation,
                            'distance': d.distance,
                            'confidence': d.confidence,
                            'classification': d.classification,
                            'threat_level': d.threat_level
                        }
                        for d in result.detections
                    ]
                }
                batch_data['results'].append(result_data)

            session_data['batches'].append(batch_data)

        output_path = self.output_directory / f"{self.session_name}_session.json"

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        return str(output_path)
