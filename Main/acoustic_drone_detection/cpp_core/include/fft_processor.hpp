/**
 * @file fft_processor.hpp
 * @brief High-performance FFT processing with optional CUDA acceleration
 */

#pragma once

#include "common.hpp"
#include <memory>

namespace drone_detection {

/**
 * @brief FFT processor with automatic CPU/GPU selection
 */
class FFTProcessor {
public:
    enum class Backend {
        CPU,
        CUDA
    };

    /**
     * @brief Create FFT processor
     * @param fft_size FFT size (should be power of 2)
     * @param num_channels Number of audio channels
     * @param backend Processing backend (CPU or CUDA)
     */
    FFTProcessor(size_t fft_size, size_t num_channels, Backend backend = Backend::CPU);
    ~FFTProcessor();

    // Disable copy
    FFTProcessor(const FFTProcessor&) = delete;
    FFTProcessor& operator=(const FFTProcessor&) = delete;

    /**
     * @brief Compute forward FFT
     * @param input Time-domain signal (channels x samples)
     * @param output Frequency-domain result (channels x fft_size/2+1)
     */
    void forward(const MatrixXr& input, MatrixXc& output);

    /**
     * @brief Compute inverse FFT
     * @param input Frequency-domain signal (channels x fft_size/2+1)
     * @param output Time-domain result (channels x fft_size)
     */
    void inverse(const MatrixXc& input, MatrixXr& output);

    /**
     * @brief Compute power spectrum
     * @param input Time-domain signal (channels x samples)
     * @param output Power spectrum (channels x fft_size/2+1)
     */
    void power_spectrum(const MatrixXr& input, MatrixXr& output);

    /**
     * @brief Compute cross-spectral density matrix
     * @param input Time-domain signal (channels x samples)
     * @param output Cross-spectral matrix (fft_size/2+1 x channels x channels)
     */
    void cross_spectral_matrix(const MatrixXr& input,
                               std::vector<MatrixXc>& output);

    /**
     * @brief Apply Hanning window to signal
     */
    void apply_window(MatrixXr& signal);

    /**
     * @brief Get FFT size
     */
    size_t fft_size() const { return fft_size_; }

    /**
     * @brief Get number of frequency bins
     */
    size_t num_bins() const { return fft_size_ / 2 + 1; }

    /**
     * @brief Get current backend
     */
    Backend backend() const { return backend_; }

    /**
     * @brief Check if CUDA is available
     */
    static bool cuda_available();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    size_t fft_size_;
    size_t num_channels_;
    Backend backend_;
    VectorXr window_;

    void init_window();
};

/**
 * @brief Short-Time Fourier Transform processor
 */
class STFTProcessor {
public:
    /**
     * @brief Create STFT processor
     * @param fft_size FFT window size
     * @param hop_size Hop size between windows
     * @param num_channels Number of audio channels
     * @param backend Processing backend
     */
    STFTProcessor(size_t fft_size, size_t hop_size, size_t num_channels,
                  FFTProcessor::Backend backend = FFTProcessor::Backend::CPU);

    /**
     * @brief Process audio block and compute STFT
     * @param input Audio signal (channels x samples)
     * @param output STFT frames (num_frames x channels x num_bins)
     */
    void process(const MatrixXr& input, std::vector<MatrixXc>& output);

    /**
     * @brief Reconstruct audio from STFT
     * @param input STFT frames
     * @param output Reconstructed audio (channels x samples)
     */
    void inverse(const std::vector<MatrixXc>& input, MatrixXr& output);

    size_t fft_size() const { return fft_size_; }
    size_t hop_size() const { return hop_size_; }
    size_t num_bins() const { return fft_size_ / 2 + 1; }

private:
    size_t fft_size_;
    size_t hop_size_;
    size_t num_channels_;
    std::unique_ptr<FFTProcessor> fft_;
    MatrixXr overlap_buffer_;
};

} // namespace drone_detection
