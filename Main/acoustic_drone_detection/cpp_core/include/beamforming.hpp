/**
 * @file beamforming.hpp
 * @brief High-performance beamforming and DOA estimation algorithms
 */

#pragma once

#include "common.hpp"
#include "fft_processor.hpp"
#include <memory>
#include <functional>

namespace drone_detection {

/**
 * @brief Microphone array geometry
 */
class ArrayGeometry {
public:
    ArrayGeometry() = default;

    /**
     * @brief Create circular array
     * @param num_mics Number of microphones
     * @param radius Array radius in meters
     */
    static ArrayGeometry circular(size_t num_mics, Real radius);

    /**
     * @brief Create spherical array
     * @param num_mics Number of microphones
     * @param radius Array radius in meters
     */
    static ArrayGeometry spherical(size_t num_mics, Real radius);

    /**
     * @brief Create planar array
     * @param rows Number of rows
     * @param cols Number of columns
     * @param spacing Element spacing in meters
     */
    static ArrayGeometry planar(size_t rows, size_t cols, Real spacing);

    /**
     * @brief Create linear array
     * @param num_mics Number of microphones
     * @param spacing Element spacing in meters
     */
    static ArrayGeometry linear(size_t num_mics, Real spacing);

    /**
     * @brief Add microphone position
     */
    void add_microphone(const Position3D& pos);

    /**
     * @brief Get microphone position
     */
    const Position3D& get_position(size_t index) const;

    /**
     * @brief Get all positions as matrix (num_mics x 3)
     */
    const MatrixXr& positions() const { return positions_; }

    /**
     * @brief Get number of microphones
     */
    size_t num_mics() const { return positions_.rows(); }

    /**
     * @brief Compute steering vector for given direction
     * @param azimuth Azimuth angle in degrees
     * @param elevation Elevation angle in degrees
     * @param frequency Frequency in Hz
     * @return Complex steering vector
     */
    VectorXc steering_vector(Real azimuth, Real elevation, Real frequency) const;

    /**
     * @brief Compute steering vectors for multiple directions
     * @param azimuths Azimuth angles
     * @param elevations Elevation angles
     * @param frequency Frequency in Hz
     * @return Steering matrix (num_directions x num_mics)
     */
    MatrixXc steering_matrix(const VectorXr& azimuths,
                             const VectorXr& elevations,
                             Real frequency) const;

private:
    MatrixXr positions_;  // num_mics x 3
    std::vector<Position3D> mic_positions_;
};

/**
 * @brief DOA estimation algorithm types
 */
enum class DOAAlgorithm {
    DELAY_SUM,    // Delay-and-sum beamforming
    MVDR,         // Minimum Variance Distortionless Response
    MUSIC,        // Multiple Signal Classification
    ESPRIT,       // Estimation of Signal Parameters via Rotational Invariance
    SRP_PHAT      // Steered Response Power - Phase Transform
};

/**
 * @brief Beamformer configuration
 */
struct BeamformerConfig {
    size_t sample_rate = 48000;
    size_t fft_size = 2048;
    size_t num_sources = 1;
    Real min_frequency = 80.0f;
    Real max_frequency = 8000.0f;
    size_t azimuth_resolution = 360;    // Number of azimuth points
    size_t elevation_resolution = 91;   // Number of elevation points
    Real elevation_min = 0.0f;
    Real elevation_max = 90.0f;
    DOAAlgorithm algorithm = DOAAlgorithm::MUSIC;
    bool use_gpu = true;
};

/**
 * @brief High-performance beamformer
 */
class Beamformer {
public:
    Beamformer(const ArrayGeometry& geometry, const BeamformerConfig& config);
    ~Beamformer();

    /**
     * @brief Estimate direction of arrival
     * @param input Multi-channel audio (channels x samples)
     * @return DOA results sorted by power
     */
    std::vector<DOAResult> estimate_doa(const MatrixXr& input);

    /**
     * @brief Compute spatial spectrum
     * @param input Multi-channel audio (channels x samples)
     * @param spectrum Output spectrum (azimuth x elevation)
     */
    void compute_spatial_spectrum(const MatrixXr& input, MatrixXr& spectrum);

    /**
     * @brief Apply beamforming to steer towards direction
     * @param input Multi-channel audio (channels x samples)
     * @param azimuth Target azimuth
     * @param elevation Target elevation
     * @param output Beamformed single-channel output
     */
    void steer(const MatrixXr& input, Real azimuth, Real elevation,
               VectorXr& output);

    /**
     * @brief Get null-steered output (suppress direction)
     */
    void null_steer(const MatrixXr& input, Real azimuth, Real elevation,
                    MatrixXr& output);

    const ArrayGeometry& geometry() const { return geometry_; }
    const BeamformerConfig& config() const { return config_; }

private:
    ArrayGeometry geometry_;
    BeamformerConfig config_;
    std::unique_ptr<FFTProcessor> fft_;

    // Precomputed steering vectors for all directions
    std::vector<MatrixXc> steering_vectors_;  // frequency x (az*el) x num_mics

    // Internal buffers
    MatrixXc covariance_;
    MatrixXc fft_data_;

    void precompute_steering_vectors();

    // Algorithm implementations
    void doa_delay_sum(const MatrixXc& cov, MatrixXr& spectrum);
    void doa_mvdr(const MatrixXc& cov, MatrixXr& spectrum);
    void doa_music(const MatrixXc& cov, MatrixXr& spectrum);
    void doa_srp_phat(const MatrixXc& data, MatrixXr& spectrum);

    std::vector<DOAResult> find_peaks(const MatrixXr& spectrum, size_t num_peaks);
};

/**
 * @brief GPU-accelerated beamformer
 */
#ifdef USE_CUDA
class CUDABeamformer {
public:
    CUDABeamformer(const ArrayGeometry& geometry, const BeamformerConfig& config);
    ~CUDABeamformer();

    std::vector<DOAResult> estimate_doa(const MatrixXr& input);
    void compute_spatial_spectrum(const MatrixXr& input, MatrixXr& spectrum);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

/**
 * @brief Adaptive beamformer with tracking
 */
class AdaptiveBeamformer {
public:
    AdaptiveBeamformer(const ArrayGeometry& geometry,
                       const BeamformerConfig& config);

    /**
     * @brief Update with new audio and return tracked directions
     */
    std::vector<DOAResult> update(const MatrixXr& input);

    /**
     * @brief Set number of sources to track
     */
    void set_num_sources(size_t num);

    /**
     * @brief Get beamformed output for tracked source
     */
    void get_source_audio(size_t source_index, const MatrixXr& input,
                          VectorXr& output);

private:
    std::unique_ptr<Beamformer> beamformer_;
    std::vector<DOAResult> tracked_sources_;
    size_t num_sources_;

    // Kalman filter for tracking
    struct SourceTracker {
        Real azimuth;
        Real elevation;
        Real azimuth_velocity;
        Real elevation_velocity;
        MatrixXr covariance;
    };
    std::vector<SourceTracker> trackers_;

    void update_trackers(const std::vector<DOAResult>& detections);
};

} // namespace drone_detection
