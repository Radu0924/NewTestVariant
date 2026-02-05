# V2 Upgrade Implementation Spec (Codex)

Project root: `D:\TestVersion\acoustic_drone_detection`

This document specifies three concrete upgrades to the existing V1 system. The goal is to keep the system runnable while improving correctness and enabling precise localization (TDOA + multilateration).

Scope:
1. Fix `RingBuffer` invariants and add selective microphone support (`enabled_channels`) in `acoustic_drone_detection/src/core/audio_capture.py`.
2. Replace notch `filtfilt` with streaming notch + implement AGC per channel in `acoustic_drone_detection/src/core/signal_processor.py`.
3. Integrate TDOA + MLAT into runtime pipeline (`acoustic_drone_detection/src/main.py` and `acoustic_drone_detection/src/core/localization.py`).

Non-goals:
- No neutralization features.
- No big architecture rewrite (no engine stages, no GUI/API rewrite) in this patch.
- No packaging restructuring.

---

## 0. V1 Behavior Summary (Baseline)

File: `acoustic_drone_detection/src/main.py`
- Reads a ~100ms frame from `AudioCapture`.
- Preprocess: bandpass + notch + optional AGC/noise gate.
- Detect presence on mono average.
- If detected: DOA via Python beamforming, localization via DOA-only (distance from RMS), classification, tracking.

File: `acoustic_drone_detection/src/core/audio_capture.py`
- Captures `indata` from sounddevice and writes `indata.T` into `RingBuffer`.

File: `acoustic_drone_detection/src/core/signal_processor.py`
- Notch uses `signal.filtfilt` (non-streaming).
- AGC is a single instance applied across multiple channels (state leaks between channels).

File: `acoustic_drone_detection/src/core/localization.py`
- Has multilateration solver but runtime pipeline uses `localize_from_doa()` only.

---

## 1. Upgrade #1: RingBuffer Fix + enabled_channels

### 1.1 Goals
- Support functional microphone selection: any number of enabled mics >= 8.
- Allow capturing more channels than used, and selecting/reordering a subset deterministically.
- Make `RingBuffer` safe under overflow/overwrite; enforce invariants.
- Keep V1 calls working.

### 1.2 Required API Changes (AudioCapture)

File: `acoustic_drone_detection/src/core/audio_capture.py`

Change `AudioCapture.__init__` signature by adding optional parameters:
- `requested_channels: Optional[int] = None`
- `enabled_channels: Optional[list[int]] = None`
- `channel_map: Optional[list[int]] = None`

Behavior:
- If `requested_channels is None`: fall back to existing `num_channels` parameter.
- Do NOT clamp to max 20 internally. Remove `max(8, min(20, num_channels))`. Only enforce min 8 on enabled count.
- Determine enabled selection list `enabled_device_channels`:
  - If `channel_map` is provided: use it as the ordered list of device channels to use.
  - Else use `enabled_channels` as the ordered list.
  - Else default to `list(range(actual_requested_channels))`.
- Enforce: `len(enabled_device_channels) >= 8`.

Store in instance fields:
- `self._requested_channels`
- `self._enabled_device_channels`
- `self._enabled_count`

### 1.3 Channel Validation Helper

Add a private method:
- `_validate_channel_selection(requested: int, enabled_device_channels: list[int]) -> None`

Rules:
- All indices must be integers.
- All indices must satisfy `0 <= idx < requested`.
- No duplicates (duplicates would bias beamforming/TDOA).
- `len(enabled_device_channels) >= 8`.

Raise `ValueError` with clear message if invalid.

### 1.4 Stream Creation (AudioCapture.start)

In `start()`:
- Query device max input channels.
- Compute `requested = min(self._requested_channels, device_info['max_input_channels'])`.
- Validate enabled selection against `requested`.

Create stream with:
- `channels=requested`
- Keep `dtype=np.float32`.

RingBuffer creation:
- Create `RingBuffer(channels=self._enabled_count, buffer_seconds=..., sample_rate=...)`.

Channel status list:
- Create `ChannelStatus` list sized to `enabled_count`.

### 1.5 Callback Changes (_audio_callback)

Current: writes all channels.

New:
- `indata` shape is `(frames, requested)`.
- Select subset and reorder:
  - `selected = indata[:, self._enabled_device_channels]` (frames x enabled_count)
  - `data = selected.T.astype(np.float32)` (enabled_count x frames)

Update:
- `_update_channel_status(data)` uses enabled_count.
- `RingBuffer.write(data)` writes enabled channels only.

### 1.6 RingBuffer Correctness Fix

File: `acoustic_drone_detection/src/core/audio_capture.py` (class `RingBuffer`)

Add invariants (must always hold under lock):
- `0 <= self._available <= self._buffer_size`
- `self._read_pos` and `self._write_pos` are in `[0 .. buffer_size-1]`

Fix `write()` as follows:

Case A: `samples >= buffer_size`
- Keep only last `buffer_size` samples.
- Overwrite entire buffer.
- Set:
  - `self._read_pos = 0`
  - `self._write_pos = 0`
  - `self._available = buffer_size`

Case B: `samples < buffer_size`
- Compute `free = buffer_size - available`.
- If `samples > free`, drop exactly `drop = samples - free` old samples:
  - `read_pos = (read_pos + drop) % buffer_size`
  - `available -= drop`
- Write new samples with wrap-around (1 or 2 slices).
- Update:
  - `write_pos = (write_pos + samples) % buffer_size`
  - `available += samples`

IMPORTANT: remove the current logic that subtracts `samples` blindly; it can push `available` negative.

Fix `read(samples, timeout)`:
- If `samples > buffer_size`: return `None` (or raise `ValueError`). Pick one and document. Recommended: raise `ValueError` because caller bug.

### 1.7 Stats and Reporting

Update `AudioCapture.statistics` to include:
- `requested_channels`
- `enabled_channels_count`
- `enabled_device_channels`

Keep `channels` but redefine:
- `channels` should represent enabled channels (so downstream knows actual matrix shape).

### 1.8 Definition of Done (Upgrade #1)
- `AudioCapture.read(samples)` returns shape `(enabled_count, samples)`.
- Works with enabled_count >= 8.
- Under sustained overload, no negative `available` and no crashes.

### 1.9 Tests (Upgrade #1)
Add tests:
- `tests/test_ringbuffer_overwrite_invariants.py`
- `tests/test_audio_channel_selection_mapping.py`

---

## 2. Upgrade #2: Streaming Notch + AGC Per Channel

### 2.1 Goals
- Remove `filtfilt` from notch filtering.
- Implement causal streaming notch with filter state per channel.
- Implement AGC per channel (state isolated).
- Keep preprocessing API stable.

### 2.2 FilterBank: Replace filtfilt Notch

File: `acoustic_drone_detection/src/core/signal_processor.py`

Current:
- `FilterBank.apply_notch` uses `signal.filtfilt` (non-streaming).

New:
- Build notch filters as (b, a) pairs as now.
- Maintain per-channel filter state for each notch.

Implement state:
- `self._notch_state: list[np.ndarray]`
- One state array per notch frequency.

State init:
- When filters are (re)initialized, do not assume channel count.
- Add method `reset_state(num_channels: int)` that creates:
  - for each notch: `zi` per channel using `scipy.signal.lfilter_zi(b, a)` scaled to 0
  - store shape `(num_channels, len(zi))`

Apply notch:
- For each notch, for each channel:
  - `y, zi_new = lfilter(b, a, x, zi=zi_old)`
  - store `zi_new`

Notes:
- This is causal and introduces phase shift (expected in realtime).

### 2.3 Bandpass: Add Streaming State

Current:
- `sosfilt` without `zi`.

New:
- Maintain `self._bandpass_state` per channel.
- Use `scipy.signal.sosfilt_zi` to initialize per channel.

Implementation:
- In `reset_state(num_channels)` also initialize:
  - `self._bandpass_zi` shape `(num_channels, n_sections, 2)`

Apply bandpass:
- For each channel:
  - `y, zi_new = sosfilt(sos, x, zi=zi_old)`

### 2.4 SignalProcessor: AGC and NoiseGate Per Channel

File: `acoustic_drone_detection/src/core/signal_processor.py`

Current bug:
- Single `AGC` instance is applied to multiple channels.

Fix:
- Keep per-channel lists:
  - `self._agc_list: list[AGC]`
  - `self._noise_gate_list: list[NoiseGate]`

Lazy init:
- On first call to `preprocess()` for multi-channel input, if list sizes do not match channel count, rebuild lists.

Processing:
- For multi-channel:
  - For each channel i:
    - apply FilterBank streaming
    - if noise gate enabled: apply `noise_gate_list[i]`
    - if AGC enabled: apply `agc_list[i]`

### 2.5 FilterBank Integration

Update `SignalProcessor.preprocess()` to call `FilterBank` in streaming mode.
- Add `FilterBank.apply_all_streaming(data)` which:
  - ensures state is initialized to match `num_channels`
  - bandpass then notch then optional highpass

### 2.6 Definition of Done (Upgrade #2)
- Notch filtering uses streaming `lfilter` with state.
- No `filtfilt` usage remains.
- AGC state is isolated per channel.

### 2.7 Tests (Upgrade #2)
Add:
- `tests/test_agc_state_isolated_per_channel.py`
- `tests/test_notch_streaming_chunk_consistency.py`

---

## 3. Upgrade #3: Integrate TDOA + MLAT in Runtime

Files:
- `acoustic_drone_detection/src/main.py`
- `acoustic_drone_detection/src/core/localization.py`

### 3.1 Goals
- Compute TDOA for enabled microphones when detection triggers.
- Use multilateration to estimate 3D position when TDOA confidence is sufficient.
- Fall back to DOA-only when TDOA is insufficient.

### 3.2 main.py: Initialize TDOAEngine

File: `acoustic_drone_detection/src/main.py`

Add imports:
- `from core.tdoa_engine import TDOAEngine, TDOAConfig`

In `DroneDetectionSystem.__init__`:
- Create a `TDOAConfig` and set max delay meters based on enabled mic geometry:
  - `mic_positions` used by beamformer/localizer must match enabled channel order.
  - Compute `max_pair_distance = max(||pi - pj||)` across enabled mic positions.
  - Set `tdoa_config.max_delay_meters = max_pair_distance`.
- Instantiate:
  - `self._tdoa_engine = TDOAEngine(sample_rate=config.audio.sample_rate, config=tdoa_config)`

IMPORTANT: ensure `mic_positions` passed to beamformer/localizer are subsetted to enabled mics.

### 3.3 main.py: Compute TDOAs in _process_frame

In `_process_frame(self, audio_data)` after detection is not NO_DETECTION:

1. Ensure `filtered` is multi-channel enabled order.
2. Compute TDOAs relative to reference channel 0:
- `n = filtered.shape[0]`
- `tdoas = np.zeros(n, dtype=np.float64)`
- `confs = np.ones(n, dtype=np.float64)`
- For each `ch != 0`:
  - `res = self._tdoa_engine.estimate_tdoa(filtered[0], filtered[ch], 0, ch)`
  - `tdoas[ch] = res.delay_seconds`
  - `confs[ch] = res.confidence`

3. Decide reliability:
- `MIN_CONF = 0.15` (configurable later)
- `valid_pairs = (confs[1:] >= MIN_CONF)`
- `num_valid = valid_pairs.sum()`
- Use MLAT if `num_valid >= 3` (at least 4 mics total including reference).

4. Localize:
- If reliable:
  - `loc_result = self._localizer.localize_hybrid(doa.azimuth, doa.elevation, tdoas=tdoas, signal_rms=signal_rms, tdoa_confidences=confs)`
- Else:
  - `loc_result = self._localizer.localize_from_doa(doa.azimuth, doa.elevation, signal_rms)`

This requires `LocalizationEngine.localize_hybrid` signature update (see below).

### 3.4 localization.py: Add Confidence Weights + Robust Loss + Covariance

File: `acoustic_drone_detection/src/core/localization.py`

#### 3.4.1 Dataclass Updates
- In `LocalizationConfig`, add:
  - `robust_loss: str = "huber"`
  - `f_scale: float = 1.0`
- In `LocalizationResult`, add:
  - `covariance: Optional[np.ndarray] = None`

#### 3.4.2 MultilaterationSolver.localize: weights
Change signature:
- `def localize(self, tdoas: np.ndarray, initial_guess: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> LocalizationResult:`

Implementation:
- Convert `tdoas` to range diffs as existing.
- Create weights vector for residuals (length n_mics-1):
  - If `weights is None`, use ones.
  - Else, ignore index 0 (reference) and use `sqrt_w = sqrt(clip(weights[1:], eps, 1.0))`.
- In residual function, multiply residuals by `sqrt_w`.

least_squares:
- Add `loss` and `f_scale`:
  - `loss = self._config.robust_loss` (map unsupported strings to "linear")
  - `f_scale = self._config.f_scale`

Covariance:
- If `result.jac` exists and is well-shaped:
  - `J = result.jac`
  - `JTJ = J.T @ J`
  - `cov = pinv(JTJ)`
  - Estimate sigma2 from residual variance and scale cov.
  - Store 3x3 in `LocalizationResult.covariance`.

Confidence:
- Combine residual and mean weight:
  - `conf_res = exp(-residual / 10)` (existing idea)
  - `conf_w = mean(clip(weights[1:], 0, 1))` if provided else 1
  - `confidence = conf_res * conf_w`

#### 3.4.3 LocalizationEngine.localize_hybrid signature
Change:
- `def localize_hybrid(self, azimuth, elevation, tdoas=None, signal_rms=None, tdoa_confidences=None) -> LocalizationResult:`

Behavior:
- If `tdoas` provided and sufficient:
  - call multilateration with `weights=tdoa_confidences`
  - return that as primary
- Else:
  - fallback to `localize_from_doa`

### 3.5 Definition of Done (Upgrade #3)
- On detection events, system computes TDOA and attempts multilateration.
- Localization result uses MLAT when confidence allows.
- System continues to run when TDOA is weak (fallback).

### 3.6 Tests (Upgrade #3)
Add:
- `tests/test_tdoa_engine_synthetic_delays.py`
- `tests/test_multilateration_with_weights.py`
- `tests/test_main_process_frame_uses_mlat_when_confident.py`

---

## 4. Acceptance Checklist (All Upgrades)

- Audio read returns `(enabled_count, samples)` and enabled_count can be 8, 9, 12, 16.
- Notch filtering is streaming (no filtfilt).
- AGC state isolation per channel verified.
- TDOA integrated and MLAT used when confidence allows.
- No breaking changes to public CLI flow (existing `main.py` runnable).
