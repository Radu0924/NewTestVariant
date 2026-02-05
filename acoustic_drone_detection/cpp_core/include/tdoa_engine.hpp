/**
 * @file tdoa_engine.hpp
 * @brief Time Difference of Arrival estimation and localization
 */

#pragma once

#include "common.hpp"
#include "beamforming.hpp"
#include "fft_processor.hpp"
#include <memory>

namespace drone_detection {

/**
 * @brief TDOA estimation method
 */
enum class TDOAMethod {
    GCC_PHAT,       // Generalized Cross-Correlation with Phase Transform
    GCC_SCOT,       // Smoothed Coherence Transform
    GCC_ML,         // Maximum Likelihood
    DIRECT_CORR     // Direct cross-correlation
};

/**
 * @brief TDOA measurement between two microphones
 */
struct TDOAMeasurement {
    size_t mic_i;
    size_t mic_j;
    Real delay_samples;
    Real delay_seconds;
    Real confidence;
};

/**
 * @brief TDOA Engine configuration
 */
struct TDOAConfig {
    size_t sample_rate = 48000;
    size_t fft_size = 2048;
    Real max_delay_seconds = 0.01f;  // Maximum expected delay
    TDOAMethod method = TDOAMethod::GCC_PHAT;
    Real min_correlation = 0.3f;     // Minimum correlation for valid measurement
    bool use_interpolation = true;   // Sub-sample interpolation
    bool use_gpu = true;
};

/**
 * @brief TDOA estimation engine
 */
class TDOAEngine {
public:
    TDOAEngine(const ArrayGeometry& geometry, const TDOAConfig& config);
    ~TDOAEngine();

    /**
     * @brief Estimate TDOA for all microphone pairs
     * @param input Multi-channel audio (channels x samples)
     * @return Vector of TDOA measurements
     */
    std::vector<TDOAMeasurement> estimate_tdoa(const MatrixXr& input);

    /**
     * @brief Estimate TDOA for specific pair
     */
    TDOAMeasurement estimate_pair(const VectorXr& mic_i, const VectorXr& mic_j,
                                  size_t idx_i, size_t idx_j);

    /**
     * @brief Compute GCC-PHAT between two signals
     * @param a First signal
     * @param b Second signal
     * @return Cross-correlation
     */
    VectorXr gcc_phat(const VectorXr& a, const VectorXr& b);

    /**
     * @brief Localize source from TDOA measurements
     * @param measurements TDOA measurements
     * @return Estimated source position
     */
    Position3D localize(const std::vector<TDOAMeasurement>& measurements);

    /**
     * @brief Localize multiple sources
     * @param input Multi-channel audio
     * @param num_sources Maximum number of sources
     * @return Vector of source positions
     */
    std::vector<Position3D> localize_sources(const MatrixXr& input,
                                             size_t num_sources = 1);

    /**
     * @brief Convert position to spherical coordinates
     */
    SphericalCoord to_spherical(const Position3D& pos) const;

    const TDOAConfig& config() const { return config_; }

private:
    ArrayGeometry geometry_;
    TDOAConfig config_;
    std::unique_ptr<FFTProcessor> fft_;

    size_t max_delay_samples_;
    std::vector<std::pair<size_t, size_t>> mic_pairs_;

    // Internal buffers
    VectorXc fft_a_, fft_b_;
    VectorXc cross_spectrum_;
    VectorXr correlation_;

    void init_mic_pairs();

    // Sub-sample interpolation
    Real interpolate_peak(const VectorXr& corr, size_t peak_idx);

    // Localization methods
    Position3D localize_spherical_intersection(
        const std::vector<TDOAMeasurement>& measurements);
    Position3D localize_least_squares(
        const std::vector<TDOAMeasurement>& measurements);
    Position3D localize_grid_search(
        const std::vector<TDOAMeasurement>& measurements);
};

/**
 * @brief Multi-source TDOA localizer using CLEAN algorithm
 */
class MultiSourceTDOA {
public:
    MultiSourceTDOA(const ArrayGeometry& geometry, const TDOAConfig& config);

    /**
     * @brief Detect and localize multiple sources
     * @param input Multi-channel audio
     * @param max_sources Maximum number of sources to detect
     * @param min_power Minimum power threshold
     * @return Vector of detected sources
     */
    std::vector<DetectionResult> detect(const MatrixXr& input,
                                        size_t max_sources = 5,
                                        Real min_power = 0.1f);

private:
    std::unique_ptr<TDOAEngine> tdoa_;
    std::unique_ptr<Beamformer> beamformer_;

    // CLEAN algorithm parameters
    Real clean_gain_ = 0.1f;
    size_t clean_iterations_ = 100;
};

/**
 * @brief Real-time TDOA tracker with Kalman filtering
 */
class TDOATracker {
public:
    struct TrackConfig {
        Real process_noise = 0.1f;
        Real measurement_noise = 0.5f;
        Real association_threshold = 10.0f;  // meters
        size_t max_missed_frames = 10;
        size_t min_hits_to_confirm = 3;
    };

    struct Track {
        int id;
        Position3D position;
        Position3D velocity;
        Real confidence;
        size_t age;
        size_t hits;
        size_t misses;
        bool confirmed;
    };

    TDOATracker(const ArrayGeometry& geometry, const TDOAConfig& tdoa_config,
                const TrackConfig& track_config = TrackConfig());

    /**
     * @brief Update tracks with new audio frame
     * @param input Multi-channel audio
     * @return Current confirmed tracks
     */
    std::vector<Track> update(const MatrixXr& input);

    /**
     * @brief Get all tracks (including unconfirmed)
     */
    std::vector<Track> get_all_tracks() const;

    /**
     * @brief Clear all tracks
     */
    void clear();

private:
    std::unique_ptr<MultiSourceTDOA> detector_;
    std::vector<Track> tracks_;
    TrackConfig config_;
    int next_track_id_ = 0;

    void predict_tracks();
    void associate_detections(const std::vector<DetectionResult>& detections);
    void update_tracks(const std::vector<DetectionResult>& detections,
                       const std::vector<int>& associations);
    void manage_tracks();
};

#ifdef USE_CUDA
/**
 * @brief GPU-accelerated TDOA engine
 */
class CUDATDOAEngine {
public:
    CUDATDOAEngine(const ArrayGeometry& geometry, const TDOAConfig& config);
    ~CUDATDOAEngine();

    std::vector<TDOAMeasurement> estimate_tdoa(const MatrixXr& input);
    std::vector<Position3D> localize_sources(const MatrixXr& input,
                                             size_t num_sources = 1);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

} // namespace drone_detection
