/**
 * @file beamforming.cpp
 * @brief Beamforming and DOA estimation implementation
 */

#include "beamforming.hpp"
#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drone_detection {

// ============================================================================
// ArrayGeometry implementation
// ============================================================================

ArrayGeometry ArrayGeometry::circular(size_t num_mics, Real radius) {
    ArrayGeometry geom;
    geom.positions_.resize(num_mics, 3);

    for (size_t i = 0; i < num_mics; ++i) {
        Real angle = TWO_PI * i / num_mics;
        Position3D pos(radius * std::cos(angle), radius * std::sin(angle), 0.0f);
        geom.mic_positions_.push_back(pos);
        geom.positions_.row(i) << pos.x, pos.y, pos.z;
    }

    return geom;
}

ArrayGeometry ArrayGeometry::spherical(size_t num_mics, Real radius) {
    ArrayGeometry geom;
    geom.positions_.resize(num_mics, 3);

    // Fibonacci sphere distribution
    Real golden_ratio = (1.0f + std::sqrt(5.0f)) / 2.0f;

    for (size_t i = 0; i < num_mics; ++i) {
        Real theta = TWO_PI * i / golden_ratio;
        Real phi = std::acos(1.0f - 2.0f * (i + 0.5f) / num_mics);

        Position3D pos(
            radius * std::sin(phi) * std::cos(theta),
            radius * std::sin(phi) * std::sin(theta),
            radius * std::cos(phi)
        );
        geom.mic_positions_.push_back(pos);
        geom.positions_.row(i) << pos.x, pos.y, pos.z;
    }

    return geom;
}

ArrayGeometry ArrayGeometry::planar(size_t rows, size_t cols, Real spacing) {
    ArrayGeometry geom;
    const size_t num_mics = rows * cols;
    geom.positions_.resize(num_mics, 3);

    Real offset_x = (cols - 1) * spacing / 2.0f;
    Real offset_y = (rows - 1) * spacing / 2.0f;

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            Position3D pos(c * spacing - offset_x, r * spacing - offset_y, 0.0f);
            geom.mic_positions_.push_back(pos);
            geom.positions_.row(r * cols + c) << pos.x, pos.y, pos.z;
        }
    }

    return geom;
}

ArrayGeometry ArrayGeometry::linear(size_t num_mics, Real spacing) {
    ArrayGeometry geom;
    geom.positions_.resize(num_mics, 3);

    Real offset = (num_mics - 1) * spacing / 2.0f;

    for (size_t i = 0; i < num_mics; ++i) {
        Position3D pos(i * spacing - offset, 0.0f, 0.0f);
        geom.mic_positions_.push_back(pos);
        geom.positions_.row(i) << pos.x, pos.y, pos.z;
    }

    return geom;
}

void ArrayGeometry::add_microphone(const Position3D& pos) {
    mic_positions_.push_back(pos);
    positions_.conservativeResize(mic_positions_.size(), 3);
    positions_.row(mic_positions_.size() - 1) << pos.x, pos.y, pos.z;
}

const Position3D& ArrayGeometry::get_position(size_t index) const {
    return mic_positions_[index];
}

VectorXc ArrayGeometry::steering_vector(Real azimuth, Real elevation,
                                        Real frequency) const {
    const size_t n = num_mics();
    VectorXc sv(n);

    Real az_rad = deg_to_rad(azimuth);
    Real el_rad = deg_to_rad(elevation);

    // Unit direction vector
    Real dx = std::cos(el_rad) * std::cos(az_rad);
    Real dy = std::cos(el_rad) * std::sin(az_rad);
    Real dz = std::sin(el_rad);

    Real k = TWO_PI * frequency / SPEED_OF_SOUND;

    for (size_t i = 0; i < n; ++i) {
        Real delay = (positions_(i, 0) * dx +
                      positions_(i, 1) * dy +
                      positions_(i, 2) * dz) / SPEED_OF_SOUND;
        sv(i) = std::exp(Complex(0, -k * SPEED_OF_SOUND * delay));
    }

    return sv;
}

MatrixXc ArrayGeometry::steering_matrix(const VectorXr& azimuths,
                                        const VectorXr& elevations,
                                        Real frequency) const {
    const size_t num_dirs = azimuths.size();
    const size_t n = num_mics();

    MatrixXc sm(num_dirs, n);

    for (size_t d = 0; d < num_dirs; ++d) {
        sm.row(d) = steering_vector(azimuths(d), elevations(d), frequency);
    }

    return sm;
}

// ============================================================================
// Beamformer implementation
// ============================================================================

Beamformer::Beamformer(const ArrayGeometry& geometry,
                       const BeamformerConfig& config)
    : geometry_(geometry)
    , config_(config)
{
    FFTProcessor::Backend backend = FFTProcessor::Backend::CPU;
#ifdef USE_CUDA
    if (config.use_gpu && FFTProcessor::cuda_available()) {
        backend = FFTProcessor::Backend::CUDA;
    }
#endif

    fft_ = std::make_unique<FFTProcessor>(config.fft_size, geometry.num_mics(),
                                          backend);

    precompute_steering_vectors();
}

Beamformer::~Beamformer() = default;

void Beamformer::precompute_steering_vectors() {
    const size_t num_bins = config_.fft_size / 2 + 1;
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;

    // Create azimuth and elevation grids
    VectorXr azimuths(az_res);
    VectorXr elevations(el_res);

    for (size_t i = 0; i < az_res; ++i) {
        azimuths(i) = 360.0f * i / az_res;
    }
    for (size_t i = 0; i < el_res; ++i) {
        elevations(i) = config_.elevation_min +
            (config_.elevation_max - config_.elevation_min) * i / (el_res - 1);
    }

    // Compute steering vectors for each frequency bin
    steering_vectors_.resize(num_bins);

    Real freq_per_bin = static_cast<Real>(config_.sample_rate) / config_.fft_size;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int bin = 0; bin < static_cast<int>(num_bins); ++bin) {
        Real freq = bin * freq_per_bin;

        if (freq < config_.min_frequency || freq > config_.max_frequency) {
            steering_vectors_[bin].resize(0, 0);
            continue;
        }

        steering_vectors_[bin].resize(az_res * el_res, geometry_.num_mics());

        for (size_t az_idx = 0; az_idx < az_res; ++az_idx) {
            for (size_t el_idx = 0; el_idx < el_res; ++el_idx) {
                size_t dir_idx = az_idx * el_res + el_idx;
                steering_vectors_[bin].row(dir_idx) =
                    geometry_.steering_vector(azimuths(az_idx),
                                              elevations(el_idx), freq);
            }
        }
    }
}

std::vector<DOAResult> Beamformer::estimate_doa(const MatrixXr& input) {
    MatrixXr spectrum;
    compute_spatial_spectrum(input, spectrum);
    return find_peaks(spectrum, config_.num_sources);
}

void Beamformer::compute_spatial_spectrum(const MatrixXr& input,
                                          MatrixXr& spectrum) {
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;
    const size_t num_bins = config_.fft_size / 2 + 1;

    spectrum.resize(az_res, el_res);
    spectrum.setZero();

    // Compute FFT
    MatrixXc freq_data;
    fft_->forward(input, freq_data);

    // Compute spatial spectrum for each frequency bin
    Real freq_per_bin = static_cast<Real>(config_.sample_rate) / config_.fft_size;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        MatrixXr local_spectrum(az_res, el_res);
        local_spectrum.setZero();

#ifdef _OPENMP
        #pragma omp for
#endif
        for (int bin = 0; bin < static_cast<int>(num_bins); ++bin) {
            if (steering_vectors_[bin].rows() == 0) continue;

            Real freq = bin * freq_per_bin;
            if (freq < config_.min_frequency || freq > config_.max_frequency) {
                continue;
            }

            VectorXc data = freq_data.col(bin);

            // Compute covariance matrix for this bin
            MatrixXc cov = data * data.adjoint();

            MatrixXr bin_spectrum(az_res, el_res);

            switch (config_.algorithm) {
                case DOAAlgorithm::DELAY_SUM:
                    doa_delay_sum(cov, bin_spectrum);
                    break;
                case DOAAlgorithm::MVDR:
                    doa_mvdr(cov, bin_spectrum);
                    break;
                case DOAAlgorithm::MUSIC:
                    doa_music(cov, bin_spectrum);
                    break;
                case DOAAlgorithm::SRP_PHAT:
                    doa_srp_phat(freq_data, bin_spectrum);
                    break;
                default:
                    doa_delay_sum(cov, bin_spectrum);
            }

            local_spectrum += bin_spectrum;
        }

#ifdef _OPENMP
        #pragma omp critical
#endif
        spectrum += local_spectrum;
    }

    // Normalize
    spectrum /= spectrum.maxCoeff();
}

void Beamformer::doa_delay_sum(const MatrixXc& cov, MatrixXr& spectrum) {
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;

    for (size_t az_idx = 0; az_idx < az_res; ++az_idx) {
        for (size_t el_idx = 0; el_idx < el_res; ++el_idx) {
            // Note: steering_vectors_ indexing simplified for this example
            // In practice, use precomputed steering vectors
            Real az = 360.0f * az_idx / az_res;
            Real el = config_.elevation_min +
                (config_.elevation_max - config_.elevation_min) * el_idx / (el_res - 1);

            VectorXc sv = geometry_.steering_vector(az, el, 1000.0f);
            Complex power = sv.adjoint() * cov * sv;
            spectrum(az_idx, el_idx) = std::real(power);
        }
    }
}

void Beamformer::doa_mvdr(const MatrixXc& cov, MatrixXr& spectrum) {
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;

    // Regularize covariance matrix
    MatrixXc cov_reg = cov;
    Real trace = std::real(cov.trace());
    Real reg = 1e-6f * trace / cov.rows();
    cov_reg += reg * MatrixXc::Identity(cov.rows(), cov.cols());

    // Compute inverse
    MatrixXc cov_inv = cov_reg.inverse();

    for (size_t az_idx = 0; az_idx < az_res; ++az_idx) {
        for (size_t el_idx = 0; el_idx < el_res; ++el_idx) {
            Real az = 360.0f * az_idx / az_res;
            Real el = config_.elevation_min +
                (config_.elevation_max - config_.elevation_min) * el_idx / (el_res - 1);

            VectorXc sv = geometry_.steering_vector(az, el, 1000.0f);
            Complex denom = sv.adjoint() * cov_inv * sv;
            spectrum(az_idx, el_idx) = 1.0f / std::max(std::real(denom), 1e-10f);
        }
    }
}

void Beamformer::doa_music(const MatrixXc& cov, MatrixXr& spectrum) {
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;
    const size_t n = geometry_.num_mics();

    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXc> solver(cov);
    MatrixXc eigenvectors = solver.eigenvectors();

    // Noise subspace (smallest eigenvalues)
    size_t noise_dim = n - config_.num_sources;
    MatrixXc noise_subspace = eigenvectors.leftCols(noise_dim);

    // MUSIC spectrum
    for (size_t az_idx = 0; az_idx < az_res; ++az_idx) {
        for (size_t el_idx = 0; el_idx < el_res; ++el_idx) {
            Real az = 360.0f * az_idx / az_res;
            Real el = config_.elevation_min +
                (config_.elevation_max - config_.elevation_min) * el_idx / (el_res - 1);

            VectorXc sv = geometry_.steering_vector(az, el, 1000.0f);
            VectorXc proj = noise_subspace.adjoint() * sv;
            Real power = proj.squaredNorm();
            spectrum(az_idx, el_idx) = 1.0f / std::max(power, 1e-10f);
        }
    }
}

void Beamformer::doa_srp_phat(const MatrixXc& data, MatrixXr& spectrum) {
    // SRP-PHAT uses GCC-PHAT for all microphone pairs
    // Simplified implementation
    doa_delay_sum(data * data.adjoint(), spectrum);
}

std::vector<DOAResult> Beamformer::find_peaks(const MatrixXr& spectrum,
                                              size_t num_peaks) {
    const size_t az_res = config_.azimuth_resolution;
    const size_t el_res = config_.elevation_resolution;

    std::vector<DOAResult> results;

    // Find local maxima
    std::vector<std::tuple<Real, size_t, size_t>> peaks;

    for (size_t az_idx = 1; az_idx < az_res - 1; ++az_idx) {
        for (size_t el_idx = 1; el_idx < el_res - 1; ++el_idx) {
            Real val = spectrum(az_idx, el_idx);

            // Check if local maximum
            bool is_peak = true;
            for (int da = -1; da <= 1; ++da) {
                for (int de = -1; de <= 1; ++de) {
                    if (da == 0 && de == 0) continue;
                    if (spectrum(az_idx + da, el_idx + de) >= val) {
                        is_peak = false;
                        break;
                    }
                }
                if (!is_peak) break;
            }

            if (is_peak) {
                peaks.emplace_back(val, az_idx, el_idx);
            }
        }
    }

    // Sort by power
    std::sort(peaks.begin(), peaks.end(),
              [](const auto& a, const auto& b) {
                  return std::get<0>(a) > std::get<0>(b);
              });

    // Return top peaks
    Real max_power = peaks.empty() ? 1.0f : std::get<0>(peaks[0]);

    for (size_t i = 0; i < std::min(num_peaks, peaks.size()); ++i) {
        DOAResult result;
        result.azimuth = 360.0f * std::get<1>(peaks[i]) / az_res;
        result.elevation = config_.elevation_min +
            (config_.elevation_max - config_.elevation_min) *
            std::get<2>(peaks[i]) / (el_res - 1);
        result.power = std::get<0>(peaks[i]);
        result.confidence = result.power / max_power;
        results.push_back(result);
    }

    return results;
}

void Beamformer::steer(const MatrixXr& input, Real azimuth, Real elevation,
                       VectorXr& output) {
    MatrixXc freq;
    fft_->forward(input, freq);

    VectorXc output_freq(freq.cols());
    output_freq.setZero();

    Real freq_per_bin = static_cast<Real>(config_.sample_rate) / config_.fft_size;

    for (Eigen::Index bin = 0; bin < freq.cols(); ++bin) {
        Real f = bin * freq_per_bin;
        if (f < config_.min_frequency || f > config_.max_frequency) continue;

        VectorXc sv = geometry_.steering_vector(azimuth, elevation, f);
        output_freq(bin) = sv.adjoint().dot(freq.col(bin));
    }

    MatrixXc output_freq_mat(1, freq.cols());
    output_freq_mat.row(0) = output_freq.transpose();

    MatrixXr output_mat;
    fft_->inverse(output_freq_mat, output_mat);
    output = output_mat.row(0).transpose();
}

void Beamformer::null_steer(const MatrixXr& input, Real azimuth, Real elevation,
                            MatrixXr& output) {
    // Apply null constraint using projection
    MatrixXc freq;
    fft_->forward(input, freq);

    Real freq_per_bin = static_cast<Real>(config_.sample_rate) / config_.fft_size;

    for (Eigen::Index bin = 0; bin < freq.cols(); ++bin) {
        Real f = bin * freq_per_bin;
        if (f < config_.min_frequency || f > config_.max_frequency) continue;

        VectorXc sv = geometry_.steering_vector(azimuth, elevation, f);
        sv.normalize();

        // Projection matrix to null space
        MatrixXc proj = MatrixXc::Identity(sv.size(), sv.size()) - sv * sv.adjoint();

        freq.col(bin) = proj * freq.col(bin);
    }

    fft_->inverse(freq, output);
}

// ============================================================================
// AdaptiveBeamformer implementation
// ============================================================================

AdaptiveBeamformer::AdaptiveBeamformer(const ArrayGeometry& geometry,
                                       const BeamformerConfig& config)
    : num_sources_(config.num_sources)
{
    beamformer_ = std::make_unique<Beamformer>(geometry, config);
}

std::vector<DOAResult> AdaptiveBeamformer::update(const MatrixXr& input) {
    auto detections = beamformer_->estimate_doa(input);
    update_trackers(detections);

    // Return tracked sources
    std::vector<DOAResult> results;
    for (const auto& tracker : trackers_) {
        DOAResult result;
        result.azimuth = tracker.azimuth;
        result.elevation = tracker.elevation;
        result.confidence = 1.0f;  // Tracked sources have high confidence
        results.push_back(result);
    }

    return results;
}

void AdaptiveBeamformer::set_num_sources(size_t num) {
    num_sources_ = num;
}

void AdaptiveBeamformer::get_source_audio(size_t source_index,
                                          const MatrixXr& input,
                                          VectorXr& output) {
    if (source_index < trackers_.size()) {
        beamformer_->steer(input, trackers_[source_index].azimuth,
                          trackers_[source_index].elevation, output);
    }
}

void AdaptiveBeamformer::update_trackers(const std::vector<DOAResult>& detections) {
    // Simple tracker update (could be enhanced with Kalman filter)
    if (trackers_.empty()) {
        for (const auto& det : detections) {
            if (trackers_.size() >= num_sources_) break;
            SourceTracker tracker;
            tracker.azimuth = det.azimuth;
            tracker.elevation = det.elevation;
            tracker.azimuth_velocity = 0;
            tracker.elevation_velocity = 0;
            trackers_.push_back(tracker);
        }
        return;
    }

    // Match detections to trackers
    for (auto& tracker : trackers_) {
        Real min_dist = 1e10f;
        const DOAResult* best_match = nullptr;

        for (const auto& det : detections) {
            Real az_diff = det.azimuth - tracker.azimuth;
            if (az_diff > 180) az_diff -= 360;
            if (az_diff < -180) az_diff += 360;

            Real el_diff = det.elevation - tracker.elevation;
            Real dist = std::sqrt(az_diff * az_diff + el_diff * el_diff);

            if (dist < min_dist && dist < 30.0f) {  // 30 degree threshold
                min_dist = dist;
                best_match = &det;
            }
        }

        if (best_match) {
            // Update with exponential smoothing
            Real alpha = 0.3f;
            tracker.azimuth = (1 - alpha) * tracker.azimuth + alpha * best_match->azimuth;
            tracker.elevation = (1 - alpha) * tracker.elevation + alpha * best_match->elevation;
        }
    }
}

} // namespace drone_detection
