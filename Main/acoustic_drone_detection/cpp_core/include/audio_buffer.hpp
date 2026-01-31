/**
 * @file audio_buffer.hpp
 * @brief Lock-free circular audio buffer for real-time processing
 */

#pragma once

#include "common.hpp"
#include <atomic>
#include <cstring>

namespace drone_detection {

/**
 * @brief Lock-free circular buffer for multi-channel audio
 *
 * Thread-safe for single producer, single consumer pattern.
 * Uses cache-line padding to avoid false sharing.
 */
class AudioBuffer {
public:
    AudioBuffer(size_t num_channels, size_t buffer_frames);
    ~AudioBuffer();

    // Disable copy
    AudioBuffer(const AudioBuffer&) = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;

    /**
     * @brief Write audio frames to buffer
     * @param data Interleaved audio data
     * @param num_frames Number of frames to write
     * @return Number of frames actually written
     */
    size_t write(const Real* data, size_t num_frames);

    /**
     * @brief Read audio frames from buffer
     * @param data Output buffer for interleaved audio
     * @param num_frames Number of frames to read
     * @return Number of frames actually read
     */
    size_t read(Real* data, size_t num_frames);

    /**
     * @brief Read audio frames into Eigen matrix (channels x samples)
     * @param output Matrix to store data (will be resized)
     * @param num_frames Number of frames to read
     * @return Number of frames actually read
     */
    size_t read_deinterleaved(MatrixXr& output, size_t num_frames);

    /**
     * @brief Get available frames to read
     */
    size_t available_read() const;

    /**
     * @brief Get available space to write
     */
    size_t available_write() const;

    /**
     * @brief Clear buffer
     */
    void clear();

    /**
     * @brief Get number of channels
     */
    size_t channels() const { return num_channels_; }

    /**
     * @brief Get buffer capacity in frames
     */
    size_t capacity() const { return buffer_frames_; }

private:
    size_t num_channels_;
    size_t buffer_frames_;
    size_t buffer_size_;  // Total samples (channels * frames)

    std::unique_ptr<Real[]> buffer_;

    // Cache-line padded atomic indices
    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};
};

/**
 * @brief Double-buffered audio processor
 *
 * Allows processing while new data is being captured.
 */
class DoubleBuffer {
public:
    DoubleBuffer(size_t num_channels, size_t buffer_frames);

    /**
     * @brief Get buffer for writing new audio
     */
    Real* get_write_buffer();

    /**
     * @brief Get buffer for reading/processing
     */
    const Real* get_read_buffer() const;

    /**
     * @brief Get read buffer as Eigen matrix (channels x samples)
     */
    Eigen::Map<const MatrixXr> get_read_matrix() const;

    /**
     * @brief Swap read and write buffers
     */
    void swap();

    size_t channels() const { return num_channels_; }
    size_t frames() const { return buffer_frames_; }

private:
    size_t num_channels_;
    size_t buffer_frames_;
    std::unique_ptr<Real[]> buffer_a_;
    std::unique_ptr<Real[]> buffer_b_;
    std::atomic<int> active_buffer_{0};
};

} // namespace drone_detection
