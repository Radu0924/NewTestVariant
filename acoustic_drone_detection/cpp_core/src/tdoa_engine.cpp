/**
 * @file tdoa_engine.cpp
 * @brief TDOA estimation and localization implementation
 */

#include "tdoa_engine.hpp"
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drone_detection {

// ============================================================================
// TDOAEngine implementation
// ============================================================================

TDOAEngine::TDOAEngine(const ArrayGeometry& geometry, const TDOAConfig& config)
    : geometry_(geometry)
    , config_(config)
{
    FFTProcessor::Backend backend = FFTProcessor::Backend::CPU;
#ifdef USE_CUDA
    if (config.use_gpu && FFTProcessor::cuda_available()) {
        backend = FFTProcessor::Backend::CUDA;
    }
#endif

    fft_ = std::make_unique<FFTProcessor>(config.fft_size, 2, backend);

    max_delay_samples_ = static_cast<size_t>(
        config.max_delay_seconds * config.sample_rate);

    init_mic_pairs();

    // Pre-allocate buffers
    const size_t num_bins = config.fft_size / 2 + 1;
    fft_a_.resize(num_bins);
    fft_b_.resize(num_bins);
    cross_spectrum_.resize(num_bins);
    correlation_.resize(config.fft_size);
}

TDOAEngine::~TDOAEngine() = default;

void TDOAEngine::init_mic_pairs() {
    const size_t n = geometry_.num_mics();
    mic_pairs_.clear();

    // All unique pairs
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            mic_pairs_.emplace_back(i, j);
        }
    }
}

std::vector<TDOAMeasurement> TDOAEngine::estimate_tdoa(const MatrixXr& input) {
    std::vector<TDOAMeasurement> measurements;
    measurements.reserve(mic_pairs_.size());

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int p = 0; p < static_cast<int>(mic_pairs_.size()); ++p) {
        size_t i = mic_pairs_[p].first;
        size_t j = mic_pairs_[p].second;

        VectorXr mic_i = input.row(i).transpose();
        VectorXr mic_j = input.row(j).transpose();

        TDOAMeasurement meas = estimate_pair(mic_i, mic_j, i, j);

#ifdef _OPENMP
        #pragma omp critical
#endif
        {
            if (meas.confidence >= config_.min_correlation) {
                measurements.push_back(meas);
            }
        }
    }

    return measurements;
}

TDOAMeasurement TDOAEngine::estimate_pair(const VectorXr& mic_i,
                                          const VectorXr& mic_j,
                                          size_t idx_i, size_t idx_j) {
    TDOAMeasurement result;
    result.mic_i = idx_i;
    result.mic_j = idx_j;

    // Compute GCC-PHAT
    VectorXr corr = gcc_phat(mic_i, mic_j);

    // Find peak within valid delay range
    size_t center = corr.size() / 2;
    size_t search_start = center - max_delay_samples_;
    size_t search_end = center + max_delay_samples_;

    search_start = std::max(search_start, static_cast<size_t>(0));
    search_end = std::min(search_end, static_cast<size_t>(corr.size()));

    Real max_val = -1e10f;
    size_t peak_idx = center;

    for (size_t i = search_start; i < search_end; ++i) {
        if (corr(i) > max_val) {
            max_val = corr(i);
            peak_idx = i;
        }
    }

    // Compute delay
    Real delay_samples = static_cast<Real>(peak_idx) - center;

    // Sub-sample interpolation
    if (config_.use_interpolation && peak_idx > 0 && peak_idx < corr.size() - 1) {
        delay_samples = interpolate_peak(corr, peak_idx);
        delay_samples -= center;
    }

    result.delay_samples = delay_samples;
    result.delay_seconds = delay_samples / config_.sample_rate;
    result.confidence = max_val;

    return result;
}

VectorXr TDOAEngine::gcc_phat(const VectorXr& a, const VectorXr& b) {
    const size_t n = std::max(a.size(), b.size());
    const size_t fft_size = config_.fft_size;
    const size_t num_bins = fft_size / 2 + 1;

    // Zero-pad inputs
    MatrixXr inputs(2, fft_size);
    inputs.setZero();
    inputs.row(0).head(std::min(static_cast<size_t>(a.size()), fft_size)) =
        a.head(std::min(static_cast<size_t>(a.size()), fft_size)).transpose();
    inputs.row(1).head(std::min(static_cast<size_t>(b.size()), fft_size)) =
        b.head(std::min(static_cast<size_t>(b.size()), fft_size)).transpose();

    // FFT
    MatrixXc freq;
    fft_->forward(inputs, freq);

    // Cross-spectrum with PHAT weighting
    VectorXc cross(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
        Complex val = freq(0, i) * std::conj(freq(1, i));
        Real mag = std::abs(val);
        if (mag > 1e-10f) {
            cross(i) = val / mag;  // PHAT normalization
        } else {
            cross(i) = 0;
        }
    }

    // IFFT
    MatrixXc cross_mat(1, num_bins);
    cross_mat.row(0) = cross.transpose();

    MatrixXr result;
    fft_->inverse(cross_mat, result);

    // Shift to center zero-lag
    VectorXr output(fft_size);
    size_t half = fft_size / 2;
    output.head(half) = result.row(0).tail(half).transpose();
    output.tail(half) = result.row(0).head(half).transpose();

    return output;
}

Real TDOAEngine::interpolate_peak(const VectorXr& corr, size_t peak_idx) {
    // Parabolic interpolation
    Real y0 = corr(peak_idx - 1);
    Real y1 = corr(peak_idx);
    Real y2 = corr(peak_idx + 1);

    Real offset = 0.5f * (y0 - y2) / (y0 - 2.0f * y1 + y2);

    return static_cast<Real>(peak_idx) + offset;
}

Position3D TDOAEngine::localize(const std::vector<TDOAMeasurement>& measurements) {
    if (measurements.size() < 3) {
        return Position3D();
    }

    // Try least squares first, fall back to grid search
    Position3D result = localize_least_squares(measurements);

    // Validate result
    Real max_range = 1000.0f;  // Maximum expected range
    if (result.distance_to(Position3D()) > max_range) {
        result = localize_grid_search(measurements);
    }

    return result;
}

Position3D TDOAEngine::localize_least_squares(
    const std::vector<TDOAMeasurement>& measurements) {

    const size_t n = measurements.size();

    // Set up linear system: Ax = b
    // For each measurement: ||p - m_i|| - ||p - m_j|| = c * tau_ij

    MatrixXr A(n - 1, 3);
    VectorXr b(n - 1);

    // Use first measurement as reference
    const auto& ref = measurements[0];
    const Position3D& m_i0 = geometry_.get_position(ref.mic_i);
    const Position3D& m_j0 = geometry_.get_position(ref.mic_j);
    Real d0 = SPEED_OF_SOUND * ref.delay_seconds;

    for (size_t k = 1; k < n; ++k) {
        const auto& meas = measurements[k];
        const Position3D& m_i = geometry_.get_position(meas.mic_i);
        const Position3D& m_j = geometry_.get_position(meas.mic_j);
        Real dk = SPEED_OF_SOUND * meas.delay_seconds;

        // Linearized constraint
        A(k-1, 0) = 2 * (m_i.x - m_i0.x - m_j.x + m_j0.x);
        A(k-1, 1) = 2 * (m_i.y - m_i0.y - m_j.y + m_j0.y);
        A(k-1, 2) = 2 * (m_i.z - m_i0.z - m_j.z + m_j0.z);

        b(k-1) = dk * dk - d0 * d0
               - m_i.x*m_i.x - m_i.y*m_i.y - m_i.z*m_i.z
               + m_j.x*m_j.x + m_j.y*m_j.y + m_j.z*m_j.z
               + m_i0.x*m_i0.x + m_i0.y*m_i0.y + m_i0.z*m_i0.z
               - m_j0.x*m_j0.x - m_j0.y*m_j0.y - m_j0.z*m_j0.z;
    }

    // Solve using pseudo-inverse
    VectorXr x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    return Position3D(x(0), x(1), x(2));
}

Position3D TDOAEngine::localize_grid_search(
    const std::vector<TDOAMeasurement>& measurements) {

    const Real range = 500.0f;
    const Real step = 10.0f;

    Position3D best_pos;
    Real best_error = 1e10f;

    for (Real x = -range; x <= range; x += step) {
        for (Real y = -range; y <= range; y += step) {
            for (Real z = 0; z <= range; z += step) {
                Position3D candidate(x, y, z);
                Real error = 0;

                for (const auto& meas : measurements) {
                    const Position3D& m_i = geometry_.get_position(meas.mic_i);
                    const Position3D& m_j = geometry_.get_position(meas.mic_j);

                    Real d_i = candidate.distance_to(m_i);
                    Real d_j = candidate.distance_to(m_j);
                    Real expected_delay = (d_i - d_j) / SPEED_OF_SOUND;

                    Real diff = expected_delay - meas.delay_seconds;
                    error += diff * diff * meas.confidence;
                }

                if (error < best_error) {
                    best_error = error;
                    best_pos = candidate;
                }
            }
        }
    }

    // Refine with finer grid
    const Real fine_range = step * 2;
    const Real fine_step = 1.0f;

    for (Real dx = -fine_range; dx <= fine_range; dx += fine_step) {
        for (Real dy = -fine_range; dy <= fine_range; dy += fine_step) {
            for (Real dz = -fine_range; dz <= fine_range; dz += fine_step) {
                Position3D candidate(best_pos.x + dx, best_pos.y + dy, best_pos.z + dz);
                Real error = 0;

                for (const auto& meas : measurements) {
                    const Position3D& m_i = geometry_.get_position(meas.mic_i);
                    const Position3D& m_j = geometry_.get_position(meas.mic_j);

                    Real d_i = candidate.distance_to(m_i);
                    Real d_j = candidate.distance_to(m_j);
                    Real expected_delay = (d_i - d_j) / SPEED_OF_SOUND;

                    Real diff = expected_delay - meas.delay_seconds;
                    error += diff * diff * meas.confidence;
                }

                if (error < best_error) {
                    best_error = error;
                    best_pos = candidate;
                }
            }
        }
    }

    return best_pos;
}

std::vector<Position3D> TDOAEngine::localize_sources(const MatrixXr& input,
                                                      size_t num_sources) {
    auto measurements = estimate_tdoa(input);

    std::vector<Position3D> sources;
    if (!measurements.empty()) {
        sources.push_back(localize(measurements));
    }

    // For multiple sources, would need more sophisticated separation
    // This is a simplified implementation

    return sources;
}

SphericalCoord TDOAEngine::to_spherical(const Position3D& pos) const {
    SphericalCoord coord;
    coord.distance = std::sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);

    if (coord.distance < 1e-6f) {
        return coord;
    }

    coord.azimuth = rad_to_deg(std::atan2(pos.y, pos.x));
    if (coord.azimuth < 0) coord.azimuth += 360.0f;

    coord.elevation = rad_to_deg(std::asin(pos.z / coord.distance));

    return coord;
}

// ============================================================================
// MultiSourceTDOA implementation
// ============================================================================

MultiSourceTDOA::MultiSourceTDOA(const ArrayGeometry& geometry,
                                 const TDOAConfig& config) {
    tdoa_ = std::make_unique<TDOAEngine>(geometry, config);

    BeamformerConfig bf_config;
    bf_config.sample_rate = config.sample_rate;
    bf_config.fft_size = config.fft_size;
    bf_config.use_gpu = config.use_gpu;

    beamformer_ = std::make_unique<Beamformer>(geometry, bf_config);
}

std::vector<DetectionResult> MultiSourceTDOA::detect(const MatrixXr& input,
                                                     size_t max_sources,
                                                     Real min_power) {
    std::vector<DetectionResult> results;

    // Get DOA estimates from beamformer
    auto doa_results = beamformer_->estimate_doa(input);

    // Localize each detection
    for (size_t i = 0; i < std::min(max_sources, doa_results.size()); ++i) {
        if (doa_results[i].power < min_power) continue;

        DetectionResult det;
        det.direction.azimuth = doa_results[i].azimuth;
        det.direction.elevation = doa_results[i].elevation;
        det.confidence = doa_results[i].confidence;
        det.timestamp_us = 0;  // Would be set by caller

        // Estimate distance using TDOA
        auto measurements = tdoa_->estimate_tdoa(input);
        if (!measurements.empty()) {
            Position3D pos = tdoa_->localize(measurements);
            det.direction.distance = pos.distance_to(Position3D());
        }

        results.push_back(det);
    }

    return results;
}

// ============================================================================
// TDOATracker implementation
// ============================================================================

TDOATracker::TDOATracker(const ArrayGeometry& geometry,
                         const TDOAConfig& tdoa_config,
                         const TrackConfig& track_config)
    : config_(track_config) {
    detector_ = std::make_unique<MultiSourceTDOA>(geometry, tdoa_config);
}

std::vector<TDOATracker::Track> TDOATracker::update(const MatrixXr& input) {
    // Predict
    predict_tracks();

    // Detect
    auto detections = detector_->detect(input);

    // Associate
    std::vector<int> associations(detections.size(), -1);
    associate_detections(detections);

    // Update
    update_tracks(detections, associations);

    // Manage (create/delete tracks)
    manage_tracks();

    // Return confirmed tracks
    std::vector<Track> confirmed;
    for (const auto& track : tracks_) {
        if (track.confirmed) {
            confirmed.push_back(track);
        }
    }

    return confirmed;
}

std::vector<TDOATracker::Track> TDOATracker::get_all_tracks() const {
    return tracks_;
}

void TDOATracker::clear() {
    tracks_.clear();
    next_track_id_ = 0;
}

void TDOATracker::predict_tracks() {
    for (auto& track : tracks_) {
        // Simple constant velocity prediction
        track.position.x += track.velocity.x;
        track.position.y += track.velocity.y;
        track.position.z += track.velocity.z;
        track.age++;
    }
}

void TDOATracker::associate_detections(const std::vector<DetectionResult>& detections) {
    // Hungarian algorithm would be better, but using greedy for simplicity
    std::vector<bool> det_used(detections.size(), false);

    for (auto& track : tracks_) {
        Real min_dist = config_.association_threshold;
        int best_det = -1;

        for (size_t d = 0; d < detections.size(); ++d) {
            if (det_used[d]) continue;

            Position3D det_pos = detections[d].direction.to_cartesian();
            Real dist = track.position.distance_to(det_pos);

            if (dist < min_dist) {
                min_dist = dist;
                best_det = static_cast<int>(d);
            }
        }

        if (best_det >= 0) {
            det_used[best_det] = true;
            track.hits++;
            track.misses = 0;

            // Update position
            Position3D det_pos = detections[best_det].direction.to_cartesian();
            Real alpha = 0.3f;
            track.velocity.x = alpha * (det_pos.x - track.position.x);
            track.velocity.y = alpha * (det_pos.y - track.position.y);
            track.velocity.z = alpha * (det_pos.z - track.position.z);
            track.position = det_pos;
            track.confidence = detections[best_det].confidence;
        } else {
            track.misses++;
        }
    }

    // Create new tracks for unassociated detections
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!det_used[d]) {
            Track new_track;
            new_track.id = next_track_id_++;
            new_track.position = detections[d].direction.to_cartesian();
            new_track.velocity = Position3D();
            new_track.confidence = detections[d].confidence;
            new_track.age = 0;
            new_track.hits = 1;
            new_track.misses = 0;
            new_track.confirmed = false;
            tracks_.push_back(new_track);
        }
    }
}

void TDOATracker::update_tracks(const std::vector<DetectionResult>& detections,
                                const std::vector<int>& associations) {
    // Confirm tracks that have enough hits
    for (auto& track : tracks_) {
        if (!track.confirmed && track.hits >= config_.min_hits_to_confirm) {
            track.confirmed = true;
        }
    }
}

void TDOATracker::manage_tracks() {
    // Remove dead tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) {
                           return t.misses > config_.max_missed_frames;
                       }),
        tracks_.end());
}

} // namespace drone_detection
