# Acoustic Drone Detection System

A professional-grade acoustic drone detection, localization, and classification system designed for critical infrastructure defense.

## Current Project Status

**Version:** 2.0 (V2 Upgrades Implemented)
**Last Updated:** February 2026

### Implemented Features

| Module | Status | Description |
|--------|--------|-------------|
| Audio Capture | Done | Multi-channel capture with RingBuffer, selective channel support |
| Signal Processing | Done | Streaming bandpass/notch filters, per-channel AGC |
| Beamforming | Done | MUSIC, MVDR, SRP-PHAT, ESPRIT algorithms |
| TDOA Engine | Done | GCC-PHAT cross-correlation for time delay estimation |
| Localization | Done | TDOA + Multilateration with confidence weighting |
| Tracking | Done | EKF/UKF with GNN data association |
| Detection | Done | Rule-based + ML-ready classification |
| GUI | Done | PyQt6 interface (radar, spectrogram, 3D view) |
| REST API | Done | FastAPI endpoints + WebSocket streaming |
| C++ Core | Done | High-performance signal processing module |

### V2 Upgrades (Completed)

1. **RingBuffer Fix + Channel Selection**
   - Selective microphone support (`enabled_channels`, `channel_map`)
   - Fixed overflow/overwrite invariants
   - Supports 8-32 microphones

2. **Streaming Signal Processing**
   - Replaced `filtfilt` with streaming `lfilter` for notch filters
   - AGC state isolated per channel (no state leakage)
   - Bandpass filter state persistence

3. **TDOA + Multilateration Integration**
   - Real-time TDOA computation with confidence metrics
   - Weighted multilateration with robust loss (Huber)
   - Automatic fallback to DOA-only when TDOA confidence is low

### Test Coverage

```
tests/
├── test_agc_state_isolated_per_channel.py
├── test_audio_channel_selection_mapping.py
├── test_beamforming.py
├── test_detector.py
├── test_main_process_frame_uses_mlat_when_confident.py
├── test_multilateration_with_weights.py
├── test_notch_streaming_chunk_consistency.py
├── test_ringbuffer_overwrite_invariants.py
├── test_signal_processor.py
├── test_tdoa_engine_synthetic_delays.py
└── test_tracking.py
```

## Project Structure

```
acoustic_drone_detection/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── core/
│   │   ├── audio_capture.py    # Multi-channel audio with RingBuffer
│   │   ├── signal_processor.py # Streaming filters, AGC, noise gate
│   │   ├── beamforming.py      # DOA estimation algorithms
│   │   ├── tdoa_engine.py      # TDOA computation (GCC-PHAT)
│   │   ├── localization.py     # Multilateration + hybrid methods
│   │   ├── tracking.py         # EKF/UKF multi-target tracking
│   │   └── cpp_backend.py      # C++ module interface
│   ├── detection/
│   │   ├── detector.py         # Drone presence detection
│   │   ├── classifier.py       # Drone type classification
│   │   └── signature_db.py     # Acoustic signature database
│   ├── hardware/
│   │   ├── audio_interface.py  # Audio device management
│   │   ├── array_geometry.py   # Microphone array configurations
│   │   ├── calibration.py      # Array calibration tools
│   │   └── gpu_detector.py     # CUDA/GPU detection
│   ├── gui/
│   │   ├── main_window.py      # Main PyQt6 window
│   │   ├── radar_display.py    # 2D radar view
│   │   ├── spectrum_view.py    # Spectrogram display
│   │   ├── visualization_3d.py # 3D target visualization
│   │   ├── alerts_panel.py     # Alert management
│   │   ├── settings_panel.py   # Configuration UI
│   │   └── recording_panel.py  # Recording controls
│   ├── api/
│   │   ├── rest_api.py         # FastAPI REST endpoints
│   │   └── websocket_server.py # Real-time streaming
│   ├── analysis/
│   │   ├── audio_loader.py     # Offline audio loading
│   │   ├── batch_processor.py  # Batch analysis
│   │   └── report_generator.py # Report generation
│   └── utils/
│       ├── config_manager.py   # YAML config handling
│       ├── logger.py           # Logging utilities
│       ├── data_export.py      # Export (JSON, CSV, SQLite)
│       └── performance_monitor.py
├── config/
│   ├── default_config.yaml     # Main configuration
│   ├── array_configs/          # Microphone array geometries
│   │   ├── 8_mic_circular.yaml
│   │   ├── 12_mic_spherical.yaml
│   │   └── 16_mic_planar.yaml
│   └── drone_profiles/         # Acoustic signatures
│       ├── dji_mavic.yaml
│       ├── dji_phantom.yaml
│       ├── fpv_racing.yaml
│       └── fixed_wing.yaml
├── cpp_core/                   # C++ performance module
│   ├── CMakeLists.txt
│   ├── src/                    # C++ source files
│   ├── include/                # C++ headers
│   ├── bindings/               # Python bindings (pybind11)
│   ├── cuda/                   # CUDA kernels
│   └── build/                  # Build artifacts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── data/                       # Recordings, models
└── scripts/                    # Utility scripts
```

## Installation

### Prerequisites

- Python 3.9+
- Multi-channel audio interface (8+ channels)
- NVIDIA GPU (optional, for CUDA acceleration)

### Development Installation

```bash
cd acoustic_drone_detection
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### GUI Mode

```bash
python src/main.py --gui
```

### CLI Mode

```bash
python src/main.py
```

### Batch Scripts

```bash
run gui      # Start GUI
run cli      # Start CLI
run tests    # Run tests
```

## Configuration

Main configuration: `config/default_config.yaml`

```yaml
audio:
  sample_rate: 48000
  num_channels: 12
  buffer_size: 512
  # V2: Selective channel support
  requested_channels: null
  enabled_channels: null
  channel_map: null

detection:
  freq_min: 80
  freq_max: 8000
  detection_threshold: 0.5
  tdoa_min_confidence: 0.15  # Minimum confidence for MLAT

beamforming:
  method: music  # das, mvdr, music, esprit, srp_phat

localization:
  method: hybrid  # spherical_intersection, multilateration, hybrid

tracking:
  algorithm: ekf  # ekf, ukf
```

## API Endpoints

```bash
uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | System status |
| `/start` | POST | Start detection |
| `/stop` | POST | Stop detection |
| `/detections` | GET | Recent detections |
| `/ws` | WebSocket | Real-time stream |

## Performance Specifications

| Metric | Value |
|--------|-------|
| Processing latency | <50ms |
| Update rate | 30-100 Hz |
| Detection range | Up to 500m |
| Angular accuracy | ±2° (high precision) |

## Hardware Requirements

### Minimum
- CPU: Intel Core i5 / AMD Ryzen 5
- RAM: 8 GB
- Audio: 8-channel interface

### Recommended
- CPU: Intel Core i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA RTX 3060+
- Audio: 12+ channel professional interface

## Testing

```bash
pytest tests/ -v
```

## Documentation

- [V2 Upgrade Implementation Spec](docs/V2_UPGRADE_IMPLEMENTATION_SPEC.md)

## License

MIT License
