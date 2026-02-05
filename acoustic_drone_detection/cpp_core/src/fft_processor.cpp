/**
 * @file fft_processor.cpp
 * @brief FFT processing implementation with CPU and optional CUDA support
 */

#include "fft_processor.hpp"
#include <unsupported/Eigen/FFT>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace drone_detection {

// ============================================================================
// FFTProcessor CPU implementation
// ============================================================================

class FFTProcessor::Impl {
public:
    Impl(size_t fft_size, size_t num_channels, Backend backend)
        : fft_size_(fft_size)
        , num_channels_(num_channels)
        , backend_(backend)
    {
#ifdef USE_CUDA
        if (backend == Backend::CUDA) {
            init_cuda();
        }
#endif
        // Eigen FFT for CPU
        fft_.SetFlag(Eigen::FFT<Real>::HalfSpectrum);
    }

    ~Impl() {
#ifdef USE_CUDA
        if (backend_ == Backend::CUDA) {
            cleanup_cuda();
        }
#endif
    }

    void forward_cpu(const MatrixXr& input, MatrixXc& output) {
        const size_t num_bins = fft_size_ / 2 + 1;
        output.resize(input.rows(), num_bins);

        VectorXr padded(fft_size_);
        VectorXc freq(num_bins);

        for (Eigen::Index ch = 0; ch < input.rows(); ++ch) {
            // Zero-pad if necessary
            padded.setZero();
            const size_t copy_size = std::min(static_cast<size_t>(input.cols()),
                                              fft_size_);
            padded.head(copy_size) = input.row(ch).head(copy_size).transpose();

            // Compute FFT
            fft_.fwd(freq, padded);
            output.row(ch) = freq.transpose();
        }
    }

    void inverse_cpu(const MatrixXc& input, MatrixXr& output) {
        output.resize(input.rows(), fft_size_);

        VectorXc freq(input.cols());
        VectorXr time(fft_size_);

        for (Eigen::Index ch = 0; ch < input.rows(); ++ch) {
            freq = input.row(ch).transpose();
            fft_.inv(time, freq);
            output.row(ch) = time.transpose();
        }
    }

#ifdef USE_CUDA
    void forward_cuda(const MatrixXr& input, MatrixXc& output);
    void inverse_cuda(const MatrixXc& input, MatrixXr& output);

private:
    void init_cuda() {
        cufftPlan1d(&cuda_plan_forward_, fft_size_, CUFFT_R2C, num_channels_);
        cufftPlan1d(&cuda_plan_inverse_, fft_size_, CUFFT_C2R, num_channels_);

        const size_t num_bins = fft_size_ / 2 + 1;
        cudaMalloc(&d_real_, num_channels_ * fft_size_ * sizeof(float));
        cudaMalloc(&d_complex_, num_channels_ * num_bins * sizeof(cufftComplex));
    }

    void cleanup_cuda() {
        cufftDestroy(cuda_plan_forward_);
        cufftDestroy(cuda_plan_inverse_);
        cudaFree(d_real_);
        cudaFree(d_complex_);
    }

    cufftHandle cuda_plan_forward_;
    cufftHandle cuda_plan_inverse_;
    float* d_real_ = nullptr;
    cufftComplex* d_complex_ = nullptr;
#endif

private:
    size_t fft_size_;
    size_t num_channels_;
    Backend backend_;
    Eigen::FFT<Real> fft_;
};

#ifdef USE_CUDA
void FFTProcessor::Impl::forward_cuda(const MatrixXr& input, MatrixXc& output) {
    const size_t num_bins = fft_size_ / 2 + 1;
    output.resize(input.rows(), num_bins);

    // Copy input to device
    cudaMemcpy(d_real_, input.data(),
               input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Execute FFT
    cufftExecR2C(cuda_plan_forward_, d_real_, d_complex_);

    // Copy result back
    std::vector<std::complex<float>> temp(num_channels_ * num_bins);
    cudaMemcpy(temp.data(), d_complex_,
               temp.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (size_t ch = 0; ch < num_channels_; ++ch) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
            output(ch, bin) = temp[ch * num_bins + bin];
        }
    }
}

void FFTProcessor::Impl::inverse_cuda(const MatrixXc& input, MatrixXr& output) {
    const size_t num_bins = input.cols();
    output.resize(input.rows(), fft_size_);

    // Copy input to device
    std::vector<std::complex<float>> temp(num_channels_ * num_bins);
    for (size_t ch = 0; ch < num_channels_; ++ch) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
            temp[ch * num_bins + bin] = input(ch, bin);
        }
    }
    cudaMemcpy(d_complex_, temp.data(),
               temp.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Execute IFFT
    cufftExecC2R(cuda_plan_inverse_, d_complex_, d_real_);

    // Copy result back and normalize
    cudaMemcpy(output.data(), d_real_,
               output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    output /= static_cast<Real>(fft_size_);
}
#endif

// ============================================================================
// FFTProcessor public interface
// ============================================================================

FFTProcessor::FFTProcessor(size_t fft_size, size_t num_channels, Backend backend)
    : fft_size_(fft_size)
    , num_channels_(num_channels)
    , backend_(backend)
{
#ifndef USE_CUDA
    if (backend == Backend::CUDA) {
        backend_ = Backend::CPU;
    }
#endif
    impl_ = std::make_unique<Impl>(fft_size, num_channels, backend_);
    init_window();
}

FFTProcessor::~FFTProcessor() = default;

void FFTProcessor::init_window() {
    window_.resize(fft_size_);
    for (size_t i = 0; i < fft_size_; ++i) {
        // Hanning window
        window_(i) = 0.5f * (1.0f - std::cos(TWO_PI * i / (fft_size_ - 1)));
    }
}

void FFTProcessor::forward(const MatrixXr& input, MatrixXc& output) {
#ifdef USE_CUDA
    if (backend_ == Backend::CUDA) {
        impl_->forward_cuda(input, output);
        return;
    }
#endif
    impl_->forward_cpu(input, output);
}

void FFTProcessor::inverse(const MatrixXc& input, MatrixXr& output) {
#ifdef USE_CUDA
    if (backend_ == Backend::CUDA) {
        impl_->inverse_cuda(input, output);
        return;
    }
#endif
    impl_->inverse_cpu(input, output);
}

void FFTProcessor::power_spectrum(const MatrixXr& input, MatrixXr& output) {
    MatrixXc freq;
    forward(input, freq);

    output.resize(freq.rows(), freq.cols());
    output = freq.cwiseAbs2();
}

void FFTProcessor::cross_spectral_matrix(const MatrixXr& input,
                                         std::vector<MatrixXc>& output) {
    MatrixXc freq;
    forward(input, freq);

    const size_t num_bins = freq.cols();
    const size_t num_channels = freq.rows();

    output.resize(num_bins);
    for (size_t bin = 0; bin < num_bins; ++bin) {
        output[bin].resize(num_channels, num_channels);

        VectorXc freq_bin = freq.col(bin);
        output[bin] = freq_bin * freq_bin.adjoint();
    }
}

void FFTProcessor::apply_window(MatrixXr& signal) {
    for (Eigen::Index ch = 0; ch < signal.rows(); ++ch) {
        const size_t apply_size = std::min(static_cast<size_t>(signal.cols()),
                                           fft_size_);
        signal.row(ch).head(apply_size).array() *= window_.head(apply_size).array();
    }
}

bool FFTProcessor::cuda_available() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

// ============================================================================
// STFTProcessor implementation
// ============================================================================

STFTProcessor::STFTProcessor(size_t fft_size, size_t hop_size,
                             size_t num_channels, FFTProcessor::Backend backend)
    : fft_size_(fft_size)
    , hop_size_(hop_size)
    , num_channels_(num_channels)
{
    fft_ = std::make_unique<FFTProcessor>(fft_size, num_channels, backend);
    overlap_buffer_.resize(num_channels, fft_size);
    overlap_buffer_.setZero();
}

void STFTProcessor::process(const MatrixXr& input, std::vector<MatrixXc>& output) {
    const size_t num_samples = input.cols();
    const size_t num_frames = (num_samples + hop_size_ - 1) / hop_size_;
    const size_t num_bins = fft_size_ / 2 + 1;

    output.resize(num_frames);

    MatrixXr frame(num_channels_, fft_size_);

    for (size_t f = 0; f < num_frames; ++f) {
        const size_t start = f * hop_size_;
        const size_t end = std::min(start + fft_size_, num_samples);
        const size_t frame_size = end - start;

        // Extract frame
        frame.setZero();
        frame.leftCols(frame_size) = input.middleCols(start, frame_size);

        // Apply window
        fft_->apply_window(frame);

        // Compute FFT
        fft_->forward(frame, output[f]);
    }
}

void STFTProcessor::inverse(const std::vector<MatrixXc>& input, MatrixXr& output) {
    const size_t num_frames = input.size();
    const size_t num_samples = (num_frames - 1) * hop_size_ + fft_size_;

    output.resize(num_channels_, num_samples);
    output.setZero();

    MatrixXr frame;

    for (size_t f = 0; f < num_frames; ++f) {
        fft_->inverse(input[f], frame);

        const size_t start = f * hop_size_;
        const size_t end = std::min(start + fft_size_, num_samples);
        const size_t frame_size = end - start;

        // Overlap-add
        output.middleCols(start, frame_size) += frame.leftCols(frame_size);
    }

    // Normalize by window overlap
    Real window_sum = 0;
    for (size_t i = 0; i < fft_size_; i += hop_size_) {
        window_sum += 1.0f;
    }
    output /= window_sum;
}

} // namespace drone_detection
