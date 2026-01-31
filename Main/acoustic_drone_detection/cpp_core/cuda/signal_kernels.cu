/**
 * @file signal_kernels.cu
 * @brief CUDA kernels for signal processing operations
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

namespace drone_detection {
namespace cuda {

/**
 * @brief Apply Hanning window to signal
 */
__global__ void apply_window_kernel(
    float* signal,
    const float* window,
    int num_channels,
    int signal_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_channels * signal_length) return;

    int sample = idx % signal_length;
    signal[idx] *= window[sample];
}

/**
 * @brief Generate Hanning window
 */
__global__ void generate_hanning_window_kernel(
    float* window,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    float val = 0.5f * (1.0f - cosf(2.0f * CUDART_PI_F * idx / (length - 1)));
    window[idx] = val;
}

/**
 * @brief Compute power spectrum from complex FFT output
 */
__global__ void power_spectrum_kernel(
    const cuFloatComplex* fft_output,
    float* power,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    cuFloatComplex val = fft_output[idx];
    power[idx] = val.x * val.x + val.y * val.y;
}

/**
 * @brief Compute cross-spectral density
 */
__global__ void cross_spectrum_kernel(
    const cuFloatComplex* fft_a,
    const cuFloatComplex* fft_b,
    cuFloatComplex* cross,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    cuFloatComplex a = fft_a[idx];
    cuFloatComplex b = fft_b[idx];
    cuFloatComplex b_conj = make_cuFloatComplex(b.x, -b.y);

    cross[idx] = cuCmulf(a, b_conj);
}

/**
 * @brief Apply GCC-PHAT weighting
 */
__global__ void gcc_phat_weight_kernel(
    cuFloatComplex* cross_spectrum,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    cuFloatComplex val = cross_spectrum[idx];
    float mag = sqrtf(val.x * val.x + val.y * val.y);

    if (mag > 1e-10f) {
        cross_spectrum[idx] = make_cuFloatComplex(val.x / mag, val.y / mag);
    } else {
        cross_spectrum[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

/**
 * @brief Spectral subtraction for noise reduction
 */
__global__ void spectral_subtraction_kernel(
    cuFloatComplex* spectrum,
    const float* noise_estimate,
    float over_subtraction,
    float floor,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    cuFloatComplex val = spectrum[idx];
    float power = val.x * val.x + val.y * val.y;
    float mag = sqrtf(power);
    float phase_x = (mag > 1e-10f) ? val.x / mag : 1.0f;
    float phase_y = (mag > 1e-10f) ? val.y / mag : 0.0f;

    float noise_power = noise_estimate[idx];
    float clean_power = power - over_subtraction * noise_power;

    if (clean_power < floor * power) {
        clean_power = floor * power;
    }

    float clean_mag = sqrtf(fmaxf(clean_power, 0.0f));
    spectrum[idx] = make_cuFloatComplex(clean_mag * phase_x, clean_mag * phase_y);
}

/**
 * @brief Apply biquad filter (batch processing)
 */
__global__ void biquad_filter_kernel(
    float* signal,
    const float* coeffs,  // [b0, b1, b2, a1, a2] per channel
    float* state,  // [x1, x2, y1, y2] per channel
    int num_channels,
    int signal_length
) {
    int ch = blockIdx.x;
    if (ch >= num_channels) return;

    float b0 = coeffs[ch * 5 + 0];
    float b1 = coeffs[ch * 5 + 1];
    float b2 = coeffs[ch * 5 + 2];
    float a1 = coeffs[ch * 5 + 3];
    float a2 = coeffs[ch * 5 + 4];

    float x1 = state[ch * 4 + 0];
    float x2 = state[ch * 4 + 1];
    float y1 = state[ch * 4 + 2];
    float y2 = state[ch * 4 + 3];

    float* ch_signal = signal + ch * signal_length;

    for (int i = 0; i < signal_length; ++i) {
        float x0 = ch_signal[i];
        float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

        ch_signal[i] = y0;

        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }

    state[ch * 4 + 0] = x1;
    state[ch * 4 + 1] = x2;
    state[ch * 4 + 2] = y1;
    state[ch * 4 + 3] = y2;
}

/**
 * @brief Compute RMS energy per channel
 */
__global__ void compute_rms_kernel(
    const float* signal,
    float* rms,
    int num_channels,
    int signal_length
) {
    extern __shared__ float shared_sum[];

    int ch = blockIdx.x;
    int tid = threadIdx.x;

    if (ch >= num_channels) return;

    const float* ch_signal = signal + ch * signal_length;

    // Parallel reduction for sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < signal_length; i += blockDim.x) {
        float val = ch_signal[i];
        local_sum += val * val;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        rms[ch] = sqrtf(shared_sum[0] / signal_length);
    }
}

/**
 * @brief Compute mel filterbank (apply to power spectrum)
 */
__global__ void apply_mel_filterbank_kernel(
    const float* power_spectrum,  // [num_channels, num_bins]
    const float* filterbank,  // [num_mel_bins, num_bins]
    float* mel_output,  // [num_channels, num_mel_bins]
    int num_channels,
    int num_bins,
    int num_mel_bins
) {
    int ch = blockIdx.y;
    int mel_bin = blockIdx.x * blockDim.x + threadIdx.x;

    if (ch >= num_channels || mel_bin >= num_mel_bins) return;

    const float* ch_power = power_spectrum + ch * num_bins;
    const float* mel_filter = filterbank + mel_bin * num_bins;

    float sum = 0.0f;
    for (int i = 0; i < num_bins; ++i) {
        sum += ch_power[i] * mel_filter[i];
    }

    // Convert to log scale
    mel_output[ch * num_mel_bins + mel_bin] = 10.0f * log10f(sum + 1e-10f);
}

// ============================================================================
// Host wrapper functions
// ============================================================================

extern "C" {

void cuda_apply_window(
    float* d_signal,
    const float* d_window,
    int num_channels,
    int signal_length,
    cudaStream_t stream
) {
    int total = num_channels * signal_length;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    apply_window_kernel<<<grid_size, block_size, 0, stream>>>(
        d_signal, d_window, num_channels, signal_length
    );
}

void cuda_generate_hanning_window(
    float* d_window,
    int length,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (length + block_size - 1) / block_size;

    generate_hanning_window_kernel<<<grid_size, block_size, 0, stream>>>(
        d_window, length
    );
}

void cuda_power_spectrum(
    const cuFloatComplex* d_fft_output,
    float* d_power,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    power_spectrum_kernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_output, d_power, size
    );
}

void cuda_gcc_phat(
    const cuFloatComplex* d_fft_a,
    const cuFloatComplex* d_fft_b,
    cuFloatComplex* d_cross,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    cross_spectrum_kernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_a, d_fft_b, d_cross, size
    );

    gcc_phat_weight_kernel<<<grid_size, block_size, 0, stream>>>(
        d_cross, size
    );
}

void cuda_spectral_subtraction(
    cuFloatComplex* d_spectrum,
    const float* d_noise_estimate,
    float over_subtraction,
    float floor,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    spectral_subtraction_kernel<<<grid_size, block_size, 0, stream>>>(
        d_spectrum, d_noise_estimate, over_subtraction, floor, size
    );
}

void cuda_compute_rms(
    const float* d_signal,
    float* d_rms,
    int num_channels,
    int signal_length,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t shared_size = block_size * sizeof(float);

    compute_rms_kernel<<<num_channels, block_size, shared_size, stream>>>(
        d_signal, d_rms, num_channels, signal_length
    );
}

void cuda_apply_mel_filterbank(
    const float* d_power_spectrum,
    const float* d_filterbank,
    float* d_mel_output,
    int num_channels,
    int num_bins,
    int num_mel_bins,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_mel_bins + block.x - 1) / block.x, num_channels);

    apply_mel_filterbank_kernel<<<grid, block, 0, stream>>>(
        d_power_spectrum, d_filterbank, d_mel_output,
        num_channels, num_bins, num_mel_bins
    );
}

} // extern "C"

} // namespace cuda
} // namespace drone_detection
