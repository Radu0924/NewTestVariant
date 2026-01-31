/**
 * @file common.hpp
 * @brief Common definitions and utilities for drone detection core
 */

#pragma once

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>
#include <Eigen/Core>
#include <complex>
#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace drone_detection {

// Type aliases
using Real = float;
using Complex = std::complex<Real>;

// Eigen matrix types
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Fixed-size types for microphone arrays
template<int N>
using VectorNr = Eigen::Matrix<Real, N, 1>;

template<int N>
using VectorNc = Eigen::Matrix<Complex, N, 1>;

template<int N, int M>
using MatrixNMr = Eigen::Matrix<Real, N, M, Eigen::RowMajor>;

// 3D position
struct Position3D {
    Real x, y, z;

    Position3D() : x(0), y(0), z(0) {}
    Position3D(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}

    Real distance_to(const Position3D& other) const {
        Real dx = x - other.x;
        Real dy = y - other.y;
        Real dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    VectorXr to_vector() const {
        VectorXr v(3);
        v << x, y, z;
        return v;
    }
};

// Spherical coordinates
struct SphericalCoord {
    Real azimuth;      // degrees, 0-360
    Real elevation;    // degrees, -90 to 90
    Real distance;     // meters

    SphericalCoord() : azimuth(0), elevation(0), distance(0) {}
    SphericalCoord(Real az, Real el, Real dist)
        : azimuth(az), elevation(el), distance(dist) {}

    Position3D to_cartesian() const {
        Real az_rad = azimuth * M_PI / 180.0;
        Real el_rad = elevation * M_PI / 180.0;
        Real cos_el = std::cos(el_rad);
        return Position3D(
            distance * cos_el * std::cos(az_rad),
            distance * cos_el * std::sin(az_rad),
            distance * std::sin(el_rad)
        );
    }
};

// Detection result
struct DetectionResult {
    SphericalCoord direction;
    Real confidence;
    Real snr;
    int64_t timestamp_us;

    DetectionResult() : confidence(0), snr(0), timestamp_us(0) {}
};

// DOA estimation result
struct DOAResult {
    Real azimuth;
    Real elevation;
    Real power;
    Real confidence;
};

// Constants
constexpr Real SPEED_OF_SOUND = 343.0f;  // m/s at 20°C
constexpr Real PI = 3.14159265358979323846f;
constexpr Real TWO_PI = 2.0f * PI;

// Utility functions
inline Real deg_to_rad(Real deg) { return deg * PI / 180.0f; }
inline Real rad_to_deg(Real rad) { return rad * 180.0f / PI; }

// CUDA error checking
#ifdef USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(error)); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            throw std::runtime_error("cuFFT error: " + std::to_string(error)); \
        } \
    } while(0)
#endif

} // namespace drone_detection
