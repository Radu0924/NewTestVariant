/**
 * @file audio_buffer.cpp
 * @brief Lock-free circular audio buffer implementation
 */

#include "audio_buffer.hpp"

namespace drone_detection {

// ============================================================================
// AudioBuffer implementation
// ============================================================================

AudioBuffer::AudioBuffer(size_t num_channels, size_t buffer_frames)
    : num_channels_(num_channels)
    , buffer_frames_(buffer_frames)
    , buffer_size_(num_channels * buffer_frames)
{
    buffer_ = std::make_unique<Real[]>(buffer_size_);
    std::fill(buffer_.get(), buffer_.get() + buffer_size_, 0.0f);
}

AudioBuffer::~AudioBuffer() = default;

size_t AudioBuffer::write(const Real* data, size_t num_frames) {
    const size_t available = available_write();
    const size_t frames_to_write = std::min(num_frames, available);

    if (frames_to_write == 0) return 0;

    const size_t write_idx = write_pos_.load(std::memory_order_relaxed);
    const size_t samples_to_write = frames_to_write * num_channels_;

    // Calculate wrap-around
    const size_t write_sample_pos = (write_idx * num_channels_) % buffer_size_;
    const size_t space_to_end = buffer_size_ - write_sample_pos;

    if (samples_to_write <= space_to_end) {
        // No wrap-around needed
        std::memcpy(buffer_.get() + write_sample_pos, data,
                    samples_to_write * sizeof(Real));
    } else {
        // Wrap-around write
        std::memcpy(buffer_.get() + write_sample_pos, data,
                    space_to_end * sizeof(Real));
        std::memcpy(buffer_.get(), data + space_to_end,
                    (samples_to_write - space_to_end) * sizeof(Real));
    }

    write_pos_.store((write_idx + frames_to_write) % buffer_frames_,
                     std::memory_order_release);

    return frames_to_write;
}

size_t AudioBuffer::read(Real* data, size_t num_frames) {
    const size_t available = available_read();
    const size_t frames_to_read = std::min(num_frames, available);

    if (frames_to_read == 0) return 0;

    const size_t read_idx = read_pos_.load(std::memory_order_relaxed);
    const size_t samples_to_read = frames_to_read * num_channels_;

    // Calculate wrap-around
    const size_t read_sample_pos = (read_idx * num_channels_) % buffer_size_;
    const size_t data_to_end = buffer_size_ - read_sample_pos;

    if (samples_to_read <= data_to_end) {
        // No wrap-around needed
        std::memcpy(data, buffer_.get() + read_sample_pos,
                    samples_to_read * sizeof(Real));
    } else {
        // Wrap-around read
        std::memcpy(data, buffer_.get() + read_sample_pos,
                    data_to_end * sizeof(Real));
        std::memcpy(data + data_to_end, buffer_.get(),
                    (samples_to_read - data_to_end) * sizeof(Real));
    }

    read_pos_.store((read_idx + frames_to_read) % buffer_frames_,
                    std::memory_order_release);

    return frames_to_read;
}

size_t AudioBuffer::read_deinterleaved(MatrixXr& output, size_t num_frames) {
    const size_t available = available_read();
    const size_t frames_to_read = std::min(num_frames, available);

    if (frames_to_read == 0) {
        output.resize(num_channels_, 0);
        return 0;
    }

    output.resize(num_channels_, frames_to_read);

    // Read interleaved data
    std::vector<Real> interleaved(frames_to_read * num_channels_);
    read(interleaved.data(), frames_to_read);

    // Deinterleave
    for (size_t frame = 0; frame < frames_to_read; ++frame) {
        for (size_t ch = 0; ch < num_channels_; ++ch) {
            output(ch, frame) = interleaved[frame * num_channels_ + ch];
        }
    }

    return frames_to_read;
}

size_t AudioBuffer::available_read() const {
    const size_t write_idx = write_pos_.load(std::memory_order_acquire);
    const size_t read_idx = read_pos_.load(std::memory_order_relaxed);

    if (write_idx >= read_idx) {
        return write_idx - read_idx;
    } else {
        return buffer_frames_ - read_idx + write_idx;
    }
}

size_t AudioBuffer::available_write() const {
    const size_t write_idx = write_pos_.load(std::memory_order_relaxed);
    const size_t read_idx = read_pos_.load(std::memory_order_acquire);

    if (read_idx > write_idx) {
        return read_idx - write_idx - 1;
    } else {
        return buffer_frames_ - write_idx + read_idx - 1;
    }
}

void AudioBuffer::clear() {
    write_pos_.store(0, std::memory_order_relaxed);
    read_pos_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// DoubleBuffer implementation
// ============================================================================

DoubleBuffer::DoubleBuffer(size_t num_channels, size_t buffer_frames)
    : num_channels_(num_channels)
    , buffer_frames_(buffer_frames)
{
    const size_t size = num_channels * buffer_frames;
    buffer_a_ = std::make_unique<Real[]>(size);
    buffer_b_ = std::make_unique<Real[]>(size);

    std::fill(buffer_a_.get(), buffer_a_.get() + size, 0.0f);
    std::fill(buffer_b_.get(), buffer_b_.get() + size, 0.0f);
}

Real* DoubleBuffer::get_write_buffer() {
    return (active_buffer_.load(std::memory_order_acquire) == 0)
        ? buffer_b_.get() : buffer_a_.get();
}

const Real* DoubleBuffer::get_read_buffer() const {
    return (active_buffer_.load(std::memory_order_acquire) == 0)
        ? buffer_a_.get() : buffer_b_.get();
}

Eigen::Map<const MatrixXr> DoubleBuffer::get_read_matrix() const {
    return Eigen::Map<const MatrixXr>(get_read_buffer(),
                                      num_channels_, buffer_frames_);
}

void DoubleBuffer::swap() {
    int current = active_buffer_.load(std::memory_order_relaxed);
    active_buffer_.store(1 - current, std::memory_order_release);
}

} // namespace drone_detection
