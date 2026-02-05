/**
 * @file fft_kernels.cu
 * @brief CUDA kernels for FFT-related operations using cuFFT
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

namespace drone_detection {
namespace cuda {

/**
 * @brief Normalize IFFT output (cuFFT doesn't normalize)
 */
__global__ void normalize_ifft_kernel(
    float* data,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    data[idx] *= scale;
}

/**
 * @brief FFT shift (move zero frequency to center)
 */
__global__ void fft_shift_kernel(
    float* data,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length / 2) return;

    int swap_idx = idx + length / 2;

    float temp = data[idx];
    data[idx] = data[swap_idx];
    data[swap_idx] = temp;
}

/**
 * @brief Complex FFT shift
 */
__global__ void fft_shift_complex_kernel(
    cuFloatComplex* data,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length / 2) return;

    int swap_idx = idx + length / 2;

    cuFloatComplex temp = data[idx];
    data[idx] = data[swap_idx];
    data[swap_idx] = temp;
}

/**
 * @brief Batch normalize IFFT output for multi-channel
 */
__global__ void batch_normalize_ifft_kernel(
    float* data,
    int batch_size,
    int fft_size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * fft_size;

    if (idx >= total) return;

    data[idx] *= scale;
}

/**
 * @brief Zero-pad signal for FFT
 */
__global__ void zero_pad_kernel(
    const float* input,
    float* output,
    int input_length,
    int output_length,
    int num_channels
) {
    int ch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ch >= num_channels || idx >= output_length) return;

    float val = 0.0f;
    if (idx < input_length) {
        val = input[ch * input_length + idx];
    }

    output[ch * output_length + idx] = val;
}

/**
 * @brief Overlap-add for STFT reconstruction
 */
__global__ void overlap_add_kernel(
    const float* frame,
    float* output,
    int frame_size,
    int hop_size,
    int frame_idx,
    int output_length,
    int num_channels
) {
    int ch = blockIdx.y;
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (ch >= num_channels || sample >= frame_size) return;

    int output_idx = frame_idx * hop_size + sample;
    if (output_idx >= output_length) return;

    atomicAdd(&output[ch * output_length + output_idx],
              frame[ch * frame_size + sample]);
}

// ============================================================================
// cuFFT wrapper class
// ============================================================================

struct CUFFTHandle {
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    int fft_size;
    int batch_size;

    CUFFTHandle(int fft_size_, int batch_size_)
        : fft_size(fft_size_), batch_size(batch_size_) {

        cufftPlan1d(&plan_forward, fft_size, CUFFT_R2C, batch_size);
        cufftPlan1d(&plan_inverse, fft_size, CUFFT_C2R, batch_size);
    }

    ~CUFFTHandle() {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
    }

    void forward(float* d_input, cuFloatComplex* d_output, cudaStream_t stream) {
        cufftSetStream(plan_forward, stream);
        cufftExecR2C(plan_forward, d_input, d_output);
    }

    void inverse(cuFloatComplex* d_input, float* d_output, cudaStream_t stream) {
        cufftSetStream(plan_inverse, stream);
        cufftExecC2R(plan_inverse, d_input, d_output);

        // Normalize
        int block_size = 256;
        int total = batch_size * fft_size;
        int grid_size = (total + block_size - 1) / block_size;

        float scale = 1.0f / fft_size;
        batch_normalize_ifft_kernel<<<grid_size, block_size, 0, stream>>>(
            d_output, batch_size, fft_size, scale
        );
    }
};

// ============================================================================
// Host wrapper functions
// ============================================================================

extern "C" {

void* cuda_create_fft_handle(int fft_size, int batch_size) {
    return new CUFFTHandle(fft_size, batch_size);
}

void cuda_destroy_fft_handle(void* handle) {
    delete static_cast<CUFFTHandle*>(handle);
}

void cuda_fft_forward(
    void* handle,
    float* d_input,
    cuFloatComplex* d_output,
    cudaStream_t stream
) {
    static_cast<CUFFTHandle*>(handle)->forward(d_input, d_output, stream);
}

void cuda_fft_inverse(
    void* handle,
    cuFloatComplex* d_input,
    float* d_output,
    cudaStream_t stream
) {
    static_cast<CUFFTHandle*>(handle)->inverse(d_input, d_output, stream);
}

void cuda_fft_shift(
    float* d_data,
    int length,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (length / 2 + block_size - 1) / block_size;

    fft_shift_kernel<<<grid_size, block_size, 0, stream>>>(d_data, length);
}

void cuda_fft_shift_complex(
    cuFloatComplex* d_data,
    int length,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (length / 2 + block_size - 1) / block_size;

    fft_shift_complex_kernel<<<grid_size, block_size, 0, stream>>>(d_data, length);
}

void cuda_zero_pad(
    const float* d_input,
    float* d_output,
    int input_length,
    int output_length,
    int num_channels,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((output_length + block.x - 1) / block.x, num_channels);

    zero_pad_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, input_length, output_length, num_channels
    );
}

void cuda_overlap_add(
    const float* d_frame,
    float* d_output,
    int frame_size,
    int hop_size,
    int frame_idx,
    int output_length,
    int num_channels,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((frame_size + block.x - 1) / block.x, num_channels);

    overlap_add_kernel<<<grid, block, 0, stream>>>(
        d_frame, d_output, frame_size, hop_size,
        frame_idx, output_length, num_channels
    );
}

} // extern "C"

} // namespace cuda
} // namespace drone_detection
