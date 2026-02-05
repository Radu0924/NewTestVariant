/**
 * @file signal_processor.hpp
 * @brief High-performance signal processing algorithms
 */

#pragma once

#include "common.hpp"
#include "fft_processor.hpp"
#include <memory>

namespace drone_detection {

/**
 * @brief IIR Biquad filter (second-order section)
 */
class BiquadFilter {
public:
    enum class Type {
        LOWPASS,
        HIGHPASS,
        BANDPASS,
        NOTCH,
        PEAK,
        LOWSHELF,
        HIGHSHELF
    };

    BiquadFilter();

    /**
     * @brief Design filter
     * @param type Filter type
     * @param sample_rate Sample rate in Hz
     * @param freq Center/cutoff frequency in Hz
     * @param q Q factor
     * @param gain_db Gain in dB (for peak/shelf filters)
     */
    void design(Type type, Real sample_rate, Real freq, Real q = 0.707f,
                Real gain_db = 0.0f);

    /**
     * @brief Process single sample
     */
    Real process(Real input);

    /**
     * @brief Process buffer in-place
     */
    void process(Real* buffer, size_t size);

    /**
     * @brief Process Eigen vector in-place
     */
    void process(VectorXr& signal);

    /**
     * @brief Reset filter state
     */
    void reset();

private:
    // Coefficients
    Real b0_, b1_, b2_;
    Real a1_, a2_;

    // State
    Real x1_, x2_;
    Real y1_, y2_;
};

/**
 * @brief Multi-channel filter bank
 */
class FilterBank {
public:
    FilterBank(size_t num_channels, size_t num_filters);

    /**
     * @brief Add bandpass filter
     * @param sample_rate Sample rate
     * @param low_freq Low cutoff frequency
     * @param high_freq High cutoff frequency
     */
    void add_bandpass(Real sample_rate, Real low_freq, Real high_freq);

    /**
     * @brief Process multi-channel audio
     * @param input Input audio (channels x samples)
     * @param output Filtered outputs (num_filters x channels x samples)
     */
    void process(const MatrixXr& input, std::vector<MatrixXr>& output);

    /**
     * @brief Reset all filters
     */
    void reset();

private:
    size_t num_channels_;
    std::vector<std::vector<BiquadFilter>> filters_;
};

/**
 * @brief High-performance signal processor
 */
class SignalProcessor {
public:
    struct Config {
        size_t sample_rate = 48000;
        size_t num_channels = 8;
        size_t fft_size = 2048;
        size_t hop_size = 512;
        Real min_frequency = 80.0f;
        Real max_frequency = 8000.0f;
        bool use_gpu = true;
    };

    SignalProcessor(const Config& config);
    ~SignalProcessor();

    /**
     * @brief Process audio block
     * @param input Raw audio (channels x samples)
     * @param output Processed audio (channels x samples)
     */
    void process(const MatrixXr& input, MatrixXr& output);

    /**
     * @brief Apply bandpass filter
     */
    void apply_bandpass(MatrixXr& signal);

    /**
     * @brief Apply noise reduction
     */
    void apply_noise_reduction(MatrixXr& signal, Real threshold_db = -40.0f);

    /**
     * @brief Compute mel spectrogram
     * @param input Audio signal (channels x samples)
     * @param output Mel spectrogram (channels x num_mel_bins x num_frames)
     */
    void compute_mel_spectrogram(const MatrixXr& input,
                                 std::vector<MatrixXr>& output,
                                 size_t num_mel_bins = 128);

    /**
     * @brief Estimate noise floor
     */
    VectorXr estimate_noise_floor(const MatrixXr& input);

    /**
     * @brief Compute signal energy
     */
    Real compute_energy(const VectorXr& signal);

    /**
     * @brief Compute SNR
     */
    Real compute_snr(const VectorXr& signal, const VectorXr& noise);

    /**
     * @brief Get cross-correlation between two signals
     */
    VectorXr cross_correlation(const VectorXr& a, const VectorXr& b);

    /**
     * @brief Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
     */
    VectorXr gcc_phat(const VectorXr& a, const VectorXr& b);

    const Config& config() const { return config_; }

private:
    Config config_;
    std::unique_ptr<FFTProcessor> fft_;
    std::unique_ptr<STFTProcessor> stft_;
    std::vector<BiquadFilter> bandpass_filters_;

    // Mel filterbank
    MatrixXr mel_filterbank_;
    void init_mel_filterbank(size_t num_mel_bins);

    // Internal buffers
    MatrixXc fft_buffer_;
    MatrixXr power_buffer_;
};

/**
 * @brief Adaptive noise cancellation using LMS algorithm
 */
class AdaptiveNoiseFilter {
public:
    AdaptiveNoiseFilter(size_t filter_length, Real mu = 0.01f);

    /**
     * @brief Process and cancel noise
     * @param primary Primary signal (contains signal + noise)
     * @param reference Reference noise signal
     * @param output Noise-cancelled output
     */
    void process(const VectorXr& primary, const VectorXr& reference,
                 VectorXr& output);

    /**
     * @brief Reset filter coefficients
     */
    void reset();

private:
    size_t filter_length_;
    Real mu_;  // Step size
    VectorXr weights_;
    VectorXr buffer_;
};

} // namespace drone_detection
