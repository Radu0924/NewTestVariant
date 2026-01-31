/**
 * @file signal_processor.cpp
 * @brief Signal processing algorithms implementation
 */

#include "signal_processor.hpp"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drone_detection {

// ============================================================================
// BiquadFilter implementation
// ============================================================================

BiquadFilter::BiquadFilter()
    : b0_(1), b1_(0), b2_(0)
    , a1_(0), a2_(0)
    , x1_(0), x2_(0)
    , y1_(0), y2_(0)
{}

void BiquadFilter::design(Type type, Real sample_rate, Real freq,
                          Real q, Real gain_db) {
    const Real omega = TWO_PI * freq / sample_rate;
    const Real sin_omega = std::sin(omega);
    const Real cos_omega = std::cos(omega);
    const Real alpha = sin_omega / (2.0f * q);
    const Real A = std::pow(10.0f, gain_db / 40.0f);

    Real b0, b1, b2, a0, a1, a2;

    switch (type) {
        case Type::LOWPASS:
            b0 = (1.0f - cos_omega) / 2.0f;
            b1 = 1.0f - cos_omega;
            b2 = (1.0f - cos_omega) / 2.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case Type::HIGHPASS:
            b0 = (1.0f + cos_omega) / 2.0f;
            b1 = -(1.0f + cos_omega);
            b2 = (1.0f + cos_omega) / 2.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case Type::BANDPASS:
            b0 = alpha;
            b1 = 0.0f;
            b2 = -alpha;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case Type::NOTCH:
            b0 = 1.0f;
            b1 = -2.0f * cos_omega;
            b2 = 1.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case Type::PEAK:
            b0 = 1.0f + alpha * A;
            b1 = -2.0f * cos_omega;
            b2 = 1.0f - alpha * A;
            a0 = 1.0f + alpha / A;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha / A;
            break;

        case Type::LOWSHELF: {
            Real sqrt_A = std::sqrt(A);
            b0 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega + 2.0f * sqrt_A * alpha);
            b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_omega);
            b2 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega - 2.0f * sqrt_A * alpha);
            a0 = (A + 1.0f) + (A - 1.0f) * cos_omega + 2.0f * sqrt_A * alpha;
            a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_omega);
            a2 = (A + 1.0f) + (A - 1.0f) * cos_omega - 2.0f * sqrt_A * alpha;
            break;
        }

        case Type::HIGHSHELF: {
            Real sqrt_A = std::sqrt(A);
            b0 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega + 2.0f * sqrt_A * alpha);
            b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_omega);
            b2 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega - 2.0f * sqrt_A * alpha);
            a0 = (A + 1.0f) - (A - 1.0f) * cos_omega + 2.0f * sqrt_A * alpha;
            a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cos_omega);
            a2 = (A + 1.0f) - (A - 1.0f) * cos_omega - 2.0f * sqrt_A * alpha;
            break;
        }
    }

    // Normalize coefficients
    b0_ = b0 / a0;
    b1_ = b1 / a0;
    b2_ = b2 / a0;
    a1_ = a1 / a0;
    a2_ = a2 / a0;
}

Real BiquadFilter::process(Real input) {
    Real output = b0_ * input + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;

    x2_ = x1_;
    x1_ = input;
    y2_ = y1_;
    y1_ = output;

    return output;
}

void BiquadFilter::process(Real* buffer, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        buffer[i] = process(buffer[i]);
    }
}

void BiquadFilter::process(VectorXr& signal) {
    for (Eigen::Index i = 0; i < signal.size(); ++i) {
        signal(i) = process(signal(i));
    }
}

void BiquadFilter::reset() {
    x1_ = x2_ = y1_ = y2_ = 0.0f;
}

// ============================================================================
// FilterBank implementation
// ============================================================================

FilterBank::FilterBank(size_t num_channels, size_t num_filters)
    : num_channels_(num_channels)
{
    filters_.reserve(num_filters);
}

void FilterBank::add_bandpass(Real sample_rate, Real low_freq, Real high_freq) {
    std::vector<BiquadFilter> channel_filters(num_channels_);

    Real center_freq = std::sqrt(low_freq * high_freq);
    Real bandwidth = high_freq - low_freq;
    Real q = center_freq / bandwidth;

    for (auto& filter : channel_filters) {
        filter.design(BiquadFilter::Type::BANDPASS, sample_rate, center_freq, q);
    }

    filters_.push_back(std::move(channel_filters));
}

void FilterBank::process(const MatrixXr& input, std::vector<MatrixXr>& output) {
    output.resize(filters_.size());

    for (size_t f = 0; f < filters_.size(); ++f) {
        output[f] = input;

        for (size_t ch = 0; ch < num_channels_; ++ch) {
            VectorXr channel = output[f].row(ch).transpose();
            filters_[f][ch].process(channel);
            output[f].row(ch) = channel.transpose();
        }
    }
}

void FilterBank::reset() {
    for (auto& filter_set : filters_) {
        for (auto& filter : filter_set) {
            filter.reset();
        }
    }
}

// ============================================================================
// SignalProcessor implementation
// ============================================================================

SignalProcessor::SignalProcessor(const Config& config)
    : config_(config)
{
    // Determine backend
    FFTProcessor::Backend backend = FFTProcessor::Backend::CPU;
#ifdef USE_CUDA
    if (config.use_gpu && FFTProcessor::cuda_available()) {
        backend = FFTProcessor::Backend::CUDA;
    }
#endif

    fft_ = std::make_unique<FFTProcessor>(config.fft_size, config.num_channels,
                                          backend);
    stft_ = std::make_unique<STFTProcessor>(config.fft_size, config.fft_size / 4,
                                            config.num_channels, backend);

    // Initialize bandpass filters
    bandpass_filters_.resize(config.num_channels);
    Real center_freq = std::sqrt(config.min_frequency * config.max_frequency);
    Real bandwidth = config.max_frequency - config.min_frequency;
    Real q = center_freq / bandwidth;

    for (auto& filter : bandpass_filters_) {
        filter.design(BiquadFilter::Type::BANDPASS, config.sample_rate,
                      center_freq, q);
    }
}

SignalProcessor::~SignalProcessor() = default;

void SignalProcessor::process(const MatrixXr& input, MatrixXr& output) {
    output = input;
    apply_bandpass(output);
}

void SignalProcessor::apply_bandpass(MatrixXr& signal) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (Eigen::Index ch = 0; ch < signal.rows(); ++ch) {
        VectorXr channel = signal.row(ch).transpose();
        bandpass_filters_[ch].process(channel);
        signal.row(ch) = channel.transpose();
    }
}

void SignalProcessor::apply_noise_reduction(MatrixXr& signal, Real threshold_db) {
    MatrixXr power;
    fft_->power_spectrum(signal, power);

    Real threshold = std::pow(10.0f, threshold_db / 10.0f);

    // Spectral gating
    MatrixXc freq;
    fft_->forward(signal, freq);

    for (Eigen::Index ch = 0; ch < freq.rows(); ++ch) {
        for (Eigen::Index bin = 0; bin < freq.cols(); ++bin) {
            if (power(ch, bin) < threshold) {
                freq(ch, bin) *= 0.1f;  // Soft gate
            }
        }
    }

    fft_->inverse(freq, signal);
}

void SignalProcessor::compute_mel_spectrogram(const MatrixXr& input,
                                              std::vector<MatrixXr>& output,
                                              size_t num_mel_bins) {
    if (mel_filterbank_.rows() != static_cast<Eigen::Index>(num_mel_bins)) {
        init_mel_filterbank(num_mel_bins);
    }

    std::vector<MatrixXc> stft_frames;
    stft_->process(input, stft_frames);

    output.resize(config_.num_channels);

    for (size_t ch = 0; ch < config_.num_channels; ++ch) {
        output[ch].resize(num_mel_bins, stft_frames.size());

        for (size_t f = 0; f < stft_frames.size(); ++f) {
            VectorXr power = stft_frames[f].row(ch).cwiseAbs2().transpose();
            output[ch].col(f) = mel_filterbank_ * power;
        }

        // Convert to log scale
        output[ch] = (output[ch].array() + 1e-10f).log10() * 10.0f;
    }
}

void SignalProcessor::init_mel_filterbank(size_t num_mel_bins) {
    const size_t num_bins = config_.fft_size / 2 + 1;

    // Mel scale conversion
    auto hz_to_mel = [](Real hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](Real mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    Real mel_min = hz_to_mel(config_.min_frequency);
    Real mel_max = hz_to_mel(config_.max_frequency);

    VectorXr mel_points(num_mel_bins + 2);
    for (size_t i = 0; i < num_mel_bins + 2; ++i) {
        Real mel = mel_min + i * (mel_max - mel_min) / (num_mel_bins + 1);
        mel_points(i) = mel_to_hz(mel);
    }

    // Convert to FFT bin indices
    VectorXr bin_points = mel_points * config_.fft_size / config_.sample_rate;

    mel_filterbank_.resize(num_mel_bins, num_bins);
    mel_filterbank_.setZero();

    for (size_t m = 0; m < num_mel_bins; ++m) {
        size_t start = static_cast<size_t>(bin_points(m));
        size_t center = static_cast<size_t>(bin_points(m + 1));
        size_t end = static_cast<size_t>(bin_points(m + 2));

        for (size_t k = start; k < center && k < num_bins; ++k) {
            mel_filterbank_(m, k) = static_cast<Real>(k - start) /
                                    (center - start);
        }
        for (size_t k = center; k < end && k < num_bins; ++k) {
            mel_filterbank_(m, k) = static_cast<Real>(end - k) /
                                    (end - center);
        }
    }
}

VectorXr SignalProcessor::estimate_noise_floor(const MatrixXr& input) {
    MatrixXr power;
    fft_->power_spectrum(input, power);

    // Minimum statistics approach
    VectorXr noise_floor(power.cols());

    for (Eigen::Index bin = 0; bin < power.cols(); ++bin) {
        VectorXr bin_power = power.col(bin);
        std::sort(bin_power.data(), bin_power.data() + bin_power.size());

        // Use 10th percentile as noise estimate
        size_t idx = bin_power.size() / 10;
        noise_floor(bin) = bin_power(idx);
    }

    return noise_floor;
}

Real SignalProcessor::compute_energy(const VectorXr& signal) {
    return signal.squaredNorm() / signal.size();
}

Real SignalProcessor::compute_snr(const VectorXr& signal, const VectorXr& noise) {
    Real signal_power = compute_energy(signal);
    Real noise_power = compute_energy(noise);

    if (noise_power < 1e-10f) return 100.0f;  // Very high SNR

    return 10.0f * std::log10(signal_power / noise_power);
}

VectorXr SignalProcessor::cross_correlation(const VectorXr& a, const VectorXr& b) {
    const size_t n = a.size() + b.size() - 1;
    const size_t fft_size = 1 << static_cast<size_t>(std::ceil(std::log2(n)));

    MatrixXr input_a(1, fft_size);
    MatrixXr input_b(1, fft_size);
    input_a.setZero();
    input_b.setZero();
    input_a.row(0).head(a.size()) = a.transpose();
    input_b.row(0).head(b.size()) = b.transpose();

    MatrixXc freq_a, freq_b;
    FFTProcessor fft(fft_size, 1, FFTProcessor::Backend::CPU);
    fft.forward(input_a, freq_a);
    fft.forward(input_b, freq_b);

    MatrixXc cross = freq_a.array() * freq_b.conjugate().array();

    MatrixXr result;
    fft.inverse(cross, result);

    return result.row(0).transpose();
}

VectorXr SignalProcessor::gcc_phat(const VectorXr& a, const VectorXr& b) {
    const size_t n = a.size() + b.size() - 1;
    const size_t fft_size = 1 << static_cast<size_t>(std::ceil(std::log2(n)));

    MatrixXr input_a(1, fft_size);
    MatrixXr input_b(1, fft_size);
    input_a.setZero();
    input_b.setZero();
    input_a.row(0).head(a.size()) = a.transpose();
    input_b.row(0).head(b.size()) = b.transpose();

    MatrixXc freq_a, freq_b;
    FFTProcessor fft(fft_size, 1, FFTProcessor::Backend::CPU);
    fft.forward(input_a, freq_a);
    fft.forward(input_b, freq_b);

    // GCC-PHAT: normalize by magnitude
    MatrixXc cross = freq_a.array() * freq_b.conjugate().array();
    for (Eigen::Index i = 0; i < cross.cols(); ++i) {
        Real mag = std::abs(cross(0, i));
        if (mag > 1e-10f) {
            cross(0, i) /= mag;
        }
    }

    MatrixXr result;
    fft.inverse(cross, result);

    return result.row(0).transpose();
}

// ============================================================================
// AdaptiveNoiseFilter implementation
// ============================================================================

AdaptiveNoiseFilter::AdaptiveNoiseFilter(size_t filter_length, Real mu)
    : filter_length_(filter_length)
    , mu_(mu)
{
    weights_.resize(filter_length);
    weights_.setZero();
    buffer_.resize(filter_length);
    buffer_.setZero();
}

void AdaptiveNoiseFilter::process(const VectorXr& primary,
                                  const VectorXr& reference,
                                  VectorXr& output) {
    output.resize(primary.size());

    for (Eigen::Index i = 0; i < primary.size(); ++i) {
        // Shift buffer
        for (size_t j = filter_length_ - 1; j > 0; --j) {
            buffer_(j) = buffer_(j - 1);
        }
        buffer_(0) = reference(i);

        // Filter output (estimated noise)
        Real estimated_noise = weights_.dot(buffer_);

        // Error signal (desired output)
        Real error = primary(i) - estimated_noise;
        output(i) = error;

        // LMS update
        Real norm = buffer_.squaredNorm() + 1e-10f;
        weights_ += (mu_ / norm) * error * buffer_;
    }
}

void AdaptiveNoiseFilter::reset() {
    weights_.setZero();
    buffer_.setZero();
}

} // namespace drone_detection
