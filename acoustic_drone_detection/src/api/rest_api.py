"""
REST API Module

FastAPI-based REST API for the drone detection system:
- System control endpoints
- Detection data retrieval
- Configuration management
- Status monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import io


# Pydantic models for API
class SystemStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class DetectionResponse(BaseModel):
    """Detection event response."""
    timestamp: float
    datetime: str
    azimuth: float = Field(..., ge=-180, le=180)
    elevation: float = Field(..., ge=-90, le=90)
    distance: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    classification: str
    threat_level: str
    track_id: Optional[int] = None
    snr: float = 0.0
    dominant_frequencies: List[float] = []


class StatusResponse(BaseModel):
    """System status response."""
    status: SystemStatus
    uptime_seconds: float
    detections_total: int
    detections_last_hour: int
    active_tracks: int
    cpu_usage: float
    memory_usage: float
    gpu_available: bool
    fps: float
    latency_ms: float


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    section: str
    key: str
    value: Any


class StartRequest(BaseModel):
    """System start request."""
    config_path: Optional[str] = None
    array_config: Optional[str] = None


class FilterParams(BaseModel):
    """Detection filter parameters."""
    min_confidence: float = 0.0
    max_distance: Optional[float] = None
    classification: Optional[str] = None
    threat_level: Optional[str] = None
    since: Optional[datetime] = None
    limit: int = 100


class TrackResponse(BaseModel):
    """Track information response."""
    track_id: int
    azimuth: float
    elevation: float
    distance: float
    velocity_azimuth: float
    velocity_elevation: float
    velocity_radial: float
    confidence: float
    classification: str
    age_seconds: float
    last_update: str


class PerformanceMetrics(BaseModel):
    """Performance metrics response."""
    fps: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]
    gpu_memory_percent: Optional[float]
    processing_latency_ms: float
    audio_buffer_usage: float
    dropped_frames: int


# Create FastAPI app
app = FastAPI(
    title="Acoustic Drone Detection API",
    description="REST API for acoustic drone detection, localization, and classification system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (would be replaced with actual system reference)
_system = None
_start_time = None
_detections_history: List[DetectionResponse] = []
_max_history = 10000


def set_detection_system(system):
    """Set the detection system instance."""
    global _system
    _system = system


def _get_system():
    """Get the detection system or raise error."""
    if _system is None:
        raise HTTPException(
            status_code=503,
            detail="Detection system not initialized"
        )
    return _system


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# System status
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status."""
    try:
        system = _get_system()
        metrics = system.performance_metrics

        # Calculate uptime
        uptime = 0.0
        if _start_time:
            uptime = (datetime.now() - _start_time).total_seconds()

        # Count recent detections
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_detections = sum(
            1 for d in _detections_history
            if datetime.fromisoformat(d.datetime) > one_hour_ago
        )

        # Get active tracks
        tracks = system.get_tracks() if system.is_running else []

        status = SystemStatus.RUNNING if system.is_running else SystemStatus.STOPPED

        return StatusResponse(
            status=status,
            uptime_seconds=uptime,
            detections_total=len(_detections_history),
            detections_last_hour=recent_detections,
            active_tracks=len(tracks),
            cpu_usage=metrics.cpu_percent if metrics else 0.0,
            memory_usage=metrics.memory_percent if metrics else 0.0,
            gpu_available=metrics.gpu_percent is not None if metrics else False,
            fps=metrics.fps if metrics else 0.0,
            latency_ms=metrics.processing_latency_ms if metrics else 0.0
        )

    except HTTPException:
        raise
    except Exception as e:
        return StatusResponse(
            status=SystemStatus.ERROR,
            uptime_seconds=0,
            detections_total=0,
            detections_last_hour=0,
            active_tracks=0,
            cpu_usage=0,
            memory_usage=0,
            gpu_available=False,
            fps=0,
            latency_ms=0
        )


# Start system
@app.post("/start")
async def start_system(request: StartRequest, background_tasks: BackgroundTasks):
    """Start the detection system."""
    global _start_time

    system = _get_system()

    if system.is_running:
        return {"status": "already_running", "message": "System is already running"}

    try:
        success = system.start()
        if success:
            _start_time = datetime.now()
            return {"status": "started", "message": "Detection system started"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start system")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Stop system
@app.post("/stop")
async def stop_system():
    """Stop the detection system."""
    global _start_time

    system = _get_system()

    if not system.is_running:
        return {"status": "already_stopped", "message": "System is already stopped"}

    try:
        system.stop()
        _start_time = None
        return {"status": "stopped", "message": "Detection system stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get detections
@app.get("/detections", response_model=List[DetectionResponse])
async def get_detections(
    min_confidence: float = Query(0.0, ge=0, le=1),
    max_distance: Optional[float] = Query(None, ge=0),
    classification: Optional[str] = None,
    threat_level: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get detection history with optional filtering."""
    filtered = _detections_history.copy()

    # Apply filters
    if min_confidence > 0:
        filtered = [d for d in filtered if d.confidence >= min_confidence]

    if max_distance is not None:
        filtered = [d for d in filtered if d.distance <= max_distance]

    if classification:
        filtered = [d for d in filtered if d.classification == classification]

    if threat_level:
        filtered = [d for d in filtered if d.threat_level == threat_level]

    if since:
        filtered = [d for d in filtered if datetime.fromisoformat(d.datetime) >= since]

    # Sort by timestamp descending and limit
    filtered.sort(key=lambda x: x.timestamp, reverse=True)
    return filtered[:limit]


# Get latest detection
@app.get("/detections/latest", response_model=Optional[DetectionResponse])
async def get_latest_detection():
    """Get the most recent detection."""
    if not _detections_history:
        return None
    return max(_detections_history, key=lambda x: x.timestamp)


# Get tracks
@app.get("/tracks", response_model=List[TrackResponse])
async def get_tracks():
    """Get current active tracks."""
    system = _get_system()

    if not system.is_running:
        return []

    tracks = system.get_tracks()
    track_responses = []

    for track in tracks:
        state = track.state
        track_responses.append(TrackResponse(
            track_id=track.track_id,
            azimuth=state.azimuth,
            elevation=state.elevation,
            distance=state.distance,
            velocity_azimuth=state.velocity_azimuth,
            velocity_elevation=state.velocity_elevation,
            velocity_radial=state.velocity_radial,
            confidence=state.confidence,
            classification=track.classification,
            age_seconds=track.age,
            last_update=datetime.fromtimestamp(track.last_update).isoformat()
        ))

    return track_responses


# Get performance metrics
@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics():
    """Get current performance metrics."""
    system = _get_system()
    metrics = system.performance_metrics

    if metrics is None:
        return PerformanceMetrics(
            fps=0, cpu_percent=0, memory_percent=0,
            gpu_percent=None, gpu_memory_percent=None,
            processing_latency_ms=0, audio_buffer_usage=0, dropped_frames=0
        )

    return PerformanceMetrics(
        fps=metrics.fps,
        cpu_percent=metrics.cpu_percent,
        memory_percent=metrics.memory_percent,
        gpu_percent=metrics.gpu_percent,
        gpu_memory_percent=metrics.gpu_memory_percent,
        processing_latency_ms=metrics.processing_latency_ms,
        audio_buffer_usage=getattr(metrics, 'audio_buffer_usage', 0),
        dropped_frames=getattr(metrics, 'dropped_frames', 0)
    )


# Get configuration
@app.get("/config")
async def get_config():
    """Get current configuration."""
    system = _get_system()
    config = system._config_manager.config

    return {
        "audio": {
            "sample_rate": config.audio.sample_rate,
            "num_channels": config.audio.num_channels,
            "buffer_size": config.audio.buffer_size,
            "device_index": config.audio.device_index
        },
        "detection": {
            "freq_min": config.detection.freq_min,
            "freq_max": config.detection.freq_max,
            "min_confidence": config.detection.min_confidence,
            "max_detection_range": config.detection.max_detection_range
        },
        "tracking": {
            "algorithm": config.tracking.algorithm,
            "max_targets": config.tracking.max_targets,
            "track_timeout": config.tracking.track_timeout
        }
    }


# Update configuration
@app.patch("/config")
async def update_config(update: ConfigUpdate):
    """Update a configuration value."""
    system = _get_system()

    try:
        config = system._config_manager.config
        section = getattr(config, update.section, None)

        if section is None:
            raise HTTPException(status_code=400, detail=f"Unknown section: {update.section}")

        if not hasattr(section, update.key):
            raise HTTPException(status_code=400, detail=f"Unknown key: {update.key}")

        setattr(section, update.key, update.value)

        return {"status": "updated", "section": update.section, "key": update.key, "value": update.value}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export data
@app.get("/export/{format}")
async def export_data(format: str):
    """Export detection data in specified format."""
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Supported formats: json, csv")

    if not _detections_history:
        raise HTTPException(status_code=404, detail="No detections to export")

    if format == "json":
        data = [d.dict() for d in _detections_history]
        content = json.dumps(data, indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=detections.json"}
        )

    elif format == "csv":
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "timestamp", "datetime", "azimuth", "elevation", "distance",
            "confidence", "classification", "threat_level", "track_id", "snr"
        ])

        # Data
        for d in _detections_history:
            writer.writerow([
                d.timestamp, d.datetime, d.azimuth, d.elevation, d.distance,
                d.confidence, d.classification, d.threat_level, d.track_id, d.snr
            ])

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=detections.csv"}
        )


# Clear history
@app.delete("/detections")
async def clear_detections():
    """Clear detection history."""
    global _detections_history
    count = len(_detections_history)
    _detections_history = []
    return {"status": "cleared", "count": count}


# Detection callback to add to history
def add_detection(event):
    """Add a detection event to history."""
    global _detections_history

    detection = DetectionResponse(
        timestamp=event.timestamp,
        datetime=datetime.fromtimestamp(event.timestamp).isoformat(),
        azimuth=event.azimuth,
        elevation=event.elevation,
        distance=event.distance,
        confidence=event.confidence,
        classification=event.classification,
        threat_level=event.threat_level,
        track_id=event.track_id,
        snr=event.snr,
        dominant_frequencies=event.dominant_frequencies or []
    )

    _detections_history.append(detection)

    # Trim history if too large
    if len(_detections_history) > _max_history:
        _detections_history = _detections_history[-_max_history:]


# Statistics endpoint
@app.get("/statistics")
async def get_statistics():
    """Get detection statistics."""
    if not _detections_history:
        return {
            "total_detections": 0,
            "classifications": {},
            "threat_levels": {},
            "average_confidence": 0,
            "average_distance": 0
        }

    classifications = {}
    threat_levels = {}
    confidences = []
    distances = []

    for d in _detections_history:
        classifications[d.classification] = classifications.get(d.classification, 0) + 1
        threat_levels[d.threat_level] = threat_levels.get(d.threat_level, 0) + 1
        confidences.append(d.confidence)
        distances.append(d.distance)

    return {
        "total_detections": len(_detections_history),
        "classifications": classifications,
        "threat_levels": threat_levels,
        "average_confidence": sum(confidences) / len(confidences),
        "average_distance": sum(distances) / len(distances),
        "min_distance": min(distances),
        "max_distance": max(distances)
    }


# Audio devices endpoint
@app.get("/audio/devices")
async def get_audio_devices():
    """Get available audio input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    "index": i,
                    "name": dev['name'],
                    "channels": dev['max_input_channels'],
                    "sample_rate": dev['default_samplerate']
                })

        return input_devices
    except Exception as e:
        return []


def create_app(detection_system=None):
    """Create FastAPI app with optional detection system."""
    if detection_system:
        set_detection_system(detection_system)
        detection_system.add_callback(add_detection)
    return app


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
