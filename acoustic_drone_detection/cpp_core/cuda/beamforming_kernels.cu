/**
 * @file beamforming_kernels.cu
 * @brief CUDA kernels for beamforming operations
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>

namespace drone_detection {
namespace cuda {

// Constants
__constant__ float d_speed_of_sound = 343.0f;
__constant__ float d_two_pi = 6.28318530718f;

/**
 * @brief Compute steering vectors for all directions and frequencies
 */
__global__ void compute_steering_vectors_kernel(
    const float* mic_positions,  // [num_mics, 3]
    cuFloatComplex* steering_vectors,  // [num_dirs, num_bins, num_mics]
    int num_mics,
    int num_dirs,
    int num_bins,
    float freq_per_bin,
    float min_freq,
    float max_freq,
    const float* azimuths,  // [num_azimuth]
    const float* elevations,  // [num_elevation]
    int num_azimuth,
    int num_elevation
) {
    int dir_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (dir_idx >= num_dirs || bin_idx >= num_bins) return;

    float freq = bin_idx * freq_per_bin;
    if (freq < min_freq || freq > max_freq) return;

    int az_idx = dir_idx / num_elevation;
    int el_idx = dir_idx % num_elevation;

    float az_rad = azimuths[az_idx] * CUDART_PI_F / 180.0f;
    float el_rad = elevations[el_idx] * CUDART_PI_F / 180.0f;

    float dx = cosf(el_rad) * cosf(az_rad);
    float dy = cosf(el_rad) * sinf(az_rad);
    float dz = sinf(el_rad);

    float k = d_two_pi * freq / d_speed_of_sound;

    for (int m = 0; m < num_mics; ++m) {
        float mic_x = mic_positions[m * 3 + 0];
        float mic_y = mic_positions[m * 3 + 1];
        float mic_z = mic_positions[m * 3 + 2];

        float delay = (mic_x * dx + mic_y * dy + mic_z * dz) / d_speed_of_sound;
        float phase = -k * d_speed_of_sound * delay;

        int out_idx = (dir_idx * num_bins + bin_idx) * num_mics + m;
        steering_vectors[out_idx] = make_cuFloatComplex(cosf(phase), sinf(phase));
    }
}

/**
 * @brief Compute covariance matrix for each frequency bin
 */
__global__ void compute_covariance_kernel(
    const cuFloatComplex* fft_data,  // [num_channels, num_bins]
    cuFloatComplex* covariance,  // [num_bins, num_channels, num_channels]
    int num_channels,
    int num_bins
) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    for (int i = 0; i < num_channels; ++i) {
        cuFloatComplex xi = fft_data[i * num_bins + bin];

        for (int j = 0; j < num_channels; ++j) {
            cuFloatComplex xj = fft_data[j * num_bins + bin];
            cuFloatComplex xj_conj = make_cuFloatComplex(xj.x, -xj.y);

            cuFloatComplex result = cuCmulf(xi, xj_conj);

            int out_idx = (bin * num_channels + i) * num_channels + j;
            covariance[out_idx] = result;
        }
    }
}

/**
 * @brief Delay-and-sum beamformer spatial spectrum
 */
__global__ void delay_sum_spectrum_kernel(
    const cuFloatComplex* covariance,  // [num_bins, num_channels, num_channels]
    const cuFloatComplex* steering_vectors,  // [num_dirs, num_bins, num_channels]
    float* spectrum,  // [num_dirs]
    int num_dirs,
    int num_bins,
    int num_channels,
    float min_freq_bin,
    float max_freq_bin
) {
    int dir = blockIdx.x * blockDim.x + threadIdx.x;
    if (dir >= num_dirs) return;

    float power = 0.0f;

    for (int bin = (int)min_freq_bin; bin < (int)max_freq_bin && bin < num_bins; ++bin) {
        // w^H * R * w
        cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);

        for (int i = 0; i < num_channels; ++i) {
            cuFloatComplex wi = steering_vectors[(dir * num_bins + bin) * num_channels + i];
            cuFloatComplex wi_conj = make_cuFloatComplex(wi.x, -wi.y);

            for (int j = 0; j < num_channels; ++j) {
                cuFloatComplex wj = steering_vectors[(dir * num_bins + bin) * num_channels + j];
                cuFloatComplex rij = covariance[(bin * num_channels + i) * num_channels + j];

                cuFloatComplex temp = cuCmulf(wi_conj, rij);
                temp = cuCmulf(temp, wj);
                result = cuCaddf(result, temp);
            }
        }

        power += result.x;  // Real part
    }

    spectrum[dir] = power;
}

/**
 * @brief MVDR beamformer spatial spectrum
 */
__global__ void mvdr_spectrum_kernel(
    const cuFloatComplex* cov_inv,  // [num_bins, num_channels, num_channels]
    const cuFloatComplex* steering_vectors,
    float* spectrum,
    int num_dirs,
    int num_bins,
    int num_channels,
    float min_freq_bin,
    float max_freq_bin
) {
    int dir = blockIdx.x * blockDim.x + threadIdx.x;
    if (dir >= num_dirs) return;

    float power = 0.0f;

    for (int bin = (int)min_freq_bin; bin < (int)max_freq_bin && bin < num_bins; ++bin) {
        // 1 / (w^H * R^-1 * w)
        cuFloatComplex denom = make_cuFloatComplex(0.0f, 0.0f);

        for (int i = 0; i < num_channels; ++i) {
            cuFloatComplex wi = steering_vectors[(dir * num_bins + bin) * num_channels + i];
            cuFloatComplex wi_conj = make_cuFloatComplex(wi.x, -wi.y);

            for (int j = 0; j < num_channels; ++j) {
                cuFloatComplex wj = steering_vectors[(dir * num_bins + bin) * num_channels + j];
                cuFloatComplex rij_inv = cov_inv[(bin * num_channels + i) * num_channels + j];

                cuFloatComplex temp = cuCmulf(wi_conj, rij_inv);
                temp = cuCmulf(temp, wj);
                denom = cuCaddf(denom, temp);
            }
        }

        float denom_real = denom.x;
        if (denom_real > 1e-10f) {
            power += 1.0f / denom_real;
        }
    }

    spectrum[dir] = power;
}

/**
 * @brief Apply beamforming weights to steer output
 */
__global__ void apply_steering_kernel(
    const cuFloatComplex* fft_data,  // [num_channels, num_bins]
    const cuFloatComplex* steering_vector,  // [num_bins, num_channels]
    cuFloatComplex* output,  // [num_bins]
    int num_channels,
    int num_bins
) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    for (int ch = 0; ch < num_channels; ++ch) {
        cuFloatComplex data = fft_data[ch * num_bins + bin];
        cuFloatComplex weight = steering_vector[bin * num_channels + ch];
        cuFloatComplex weight_conj = make_cuFloatComplex(weight.x, -weight.y);

        sum = cuCaddf(sum, cuCmulf(weight_conj, data));
    }

    output[bin] = sum;
}

// ============================================================================
// Host wrapper functions
// ============================================================================

extern "C" {

void cuda_compute_steering_vectors(
    const float* d_mic_positions,
    cuFloatComplex* d_steering_vectors,
    int num_mics,
    int num_dirs,
    int num_bins,
    float freq_per_bin,
    float min_freq,
    float max_freq,
    const float* d_azimuths,
    const float* d_elevations,
    int num_azimuth,
    int num_elevation,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((num_dirs + block.x - 1) / block.x,
              (num_bins + block.y - 1) / block.y);

    compute_steering_vectors_kernel<<<grid, block, 0, stream>>>(
        d_mic_positions, d_steering_vectors,
        num_mics, num_dirs, num_bins,
        freq_per_bin, min_freq, max_freq,
        d_azimuths, d_elevations,
        num_azimuth, num_elevation
    );
}

void cuda_compute_covariance(
    const cuFloatComplex* d_fft_data,
    cuFloatComplex* d_covariance,
    int num_channels,
    int num_bins,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_bins + block_size - 1) / block_size;

    compute_covariance_kernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_data, d_covariance, num_channels, num_bins
    );
}

void cuda_delay_sum_spectrum(
    const cuFloatComplex* d_covariance,
    const cuFloatComplex* d_steering_vectors,
    float* d_spectrum,
    int num_dirs,
    int num_bins,
    int num_channels,
    float min_freq_bin,
    float max_freq_bin,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_dirs + block_size - 1) / block_size;

    delay_sum_spectrum_kernel<<<grid_size, block_size, 0, stream>>>(
        d_covariance, d_steering_vectors, d_spectrum,
        num_dirs, num_bins, num_channels,
        min_freq_bin, max_freq_bin
    );
}

void cuda_mvdr_spectrum(
    const cuFloatComplex* d_cov_inv,
    const cuFloatComplex* d_steering_vectors,
    float* d_spectrum,
    int num_dirs,
    int num_bins,
    int num_channels,
    float min_freq_bin,
    float max_freq_bin,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_dirs + block_size - 1) / block_size;

    mvdr_spectrum_kernel<<<grid_size, block_size, 0, stream>>>(
        d_cov_inv, d_steering_vectors, d_spectrum,
        num_dirs, num_bins, num_channels,
        min_freq_bin, max_freq_bin
    );
}

void cuda_apply_steering(
    const cuFloatComplex* d_fft_data,
    const cuFloatComplex* d_steering_vector,
    cuFloatComplex* d_output,
    int num_channels,
    int num_bins,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_bins + block_size - 1) / block_size;

    apply_steering_kernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_data, d_steering_vector, d_output,
        num_channels, num_bins
    );
}

} // extern "C"

} // namespace cuda
} // namespace drone_detection
