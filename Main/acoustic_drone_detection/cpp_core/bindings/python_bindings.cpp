/**
 * @file python_bindings.cpp
 * @brief Python bindings for drone detection C++ core using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "common.hpp"
#include "audio_buffer.hpp"
#include "fft_processor.hpp"
#include "signal_processor.hpp"
#include "beamforming.hpp"
#include "tdoa_engine.hpp"

namespace py = pybind11;
using namespace drone_detection;

// Helper to convert numpy array to Eigen matrix
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
numpy_to_eigen(py::array_t<T> array) {
    auto buf = array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }

    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    return Eigen::Map<MatrixType>(
        static_cast<T*>(buf.ptr),
        buf.shape[0],
        buf.shape[1]
    );
}

// Helper to convert Eigen matrix to numpy array
template<typename Derived>
py::array_t<typename Derived::Scalar>
eigen_to_numpy(const Eigen::MatrixBase<Derived>& matrix) {
    using Scalar = typename Derived::Scalar;

    py::array_t<Scalar> result({matrix.rows(), matrix.cols()});
    auto buf = result.request();
    Scalar* ptr = static_cast<Scalar*>(buf.ptr);

    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
            ptr[i * matrix.cols() + j] = matrix(i, j);
        }
    }

    return result;
}

PYBIND11_MODULE(drone_core_py, m) {
    m.doc() = "Drone Detection C++ Core - High-performance audio processing";

    // ========================================================================
    // Common types
    // ========================================================================

    py::class_<Position3D>(m, "Position3D")
        .def(py::init<>())
        .def(py::init<Real, Real, Real>())
        .def_readwrite("x", &Position3D::x)
        .def_readwrite("y", &Position3D::y)
        .def_readwrite("z", &Position3D::z)
        .def("distance_to", &Position3D::distance_to)
        .def("__repr__", [](const Position3D& p) {
            return "Position3D(" + std::to_string(p.x) + ", " +
                   std::to_string(p.y) + ", " + std::to_string(p.z) + ")";
        });

    py::class_<SphericalCoord>(m, "SphericalCoord")
        .def(py::init<>())
        .def(py::init<Real, Real, Real>())
        .def_readwrite("azimuth", &SphericalCoord::azimuth)
        .def_readwrite("elevation", &SphericalCoord::elevation)
        .def_readwrite("distance", &SphericalCoord::distance)
        .def("to_cartesian", &SphericalCoord::to_cartesian);

    py::class_<DOAResult>(m, "DOAResult")
        .def(py::init<>())
        .def_readwrite("azimuth", &DOAResult::azimuth)
        .def_readwrite("elevation", &DOAResult::elevation)
        .def_readwrite("power", &DOAResult::power)
        .def_readwrite("confidence", &DOAResult::confidence);

    py::class_<DetectionResult>(m, "DetectionResult")
        .def(py::init<>())
        .def_readwrite("direction", &DetectionResult::direction)
        .def_readwrite("confidence", &DetectionResult::confidence)
        .def_readwrite("snr", &DetectionResult::snr)
        .def_readwrite("timestamp_us", &DetectionResult::timestamp_us);

    // ========================================================================
    // FFT Processor
    // ========================================================================

    py::enum_<FFTProcessor::Backend>(m, "FFTBackend")
        .value("CPU", FFTProcessor::Backend::CPU)
        .value("CUDA", FFTProcessor::Backend::CUDA);

    py::class_<FFTProcessor>(m, "FFTProcessor")
        .def(py::init<size_t, size_t, FFTProcessor::Backend>(),
             py::arg("fft_size"),
             py::arg("num_channels"),
             py::arg("backend") = FFTProcessor::Backend::CPU)
        .def("forward", [](FFTProcessor& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            MatrixXc out;
            self.forward(in, out);

            // Convert complex to numpy
            py::array_t<std::complex<Real>> result({out.rows(), out.cols()});
            auto buf = result.request();
            std::complex<Real>* ptr = static_cast<std::complex<Real>*>(buf.ptr);
            for (Eigen::Index i = 0; i < out.rows(); ++i) {
                for (Eigen::Index j = 0; j < out.cols(); ++j) {
                    ptr[i * out.cols() + j] = out(i, j);
                }
            }
            return result;
        })
        .def("power_spectrum", [](FFTProcessor& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            MatrixXr out;
            self.power_spectrum(in, out);
            return eigen_to_numpy(out);
        })
        .def("fft_size", &FFTProcessor::fft_size)
        .def("num_bins", &FFTProcessor::num_bins)
        .def_static("cuda_available", &FFTProcessor::cuda_available);

    // ========================================================================
    // Signal Processor
    // ========================================================================

    py::class_<SignalProcessor::Config>(m, "SignalProcessorConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &SignalProcessor::Config::sample_rate)
        .def_readwrite("num_channels", &SignalProcessor::Config::num_channels)
        .def_readwrite("fft_size", &SignalProcessor::Config::fft_size)
        .def_readwrite("hop_size", &SignalProcessor::Config::hop_size)
        .def_readwrite("min_frequency", &SignalProcessor::Config::min_frequency)
        .def_readwrite("max_frequency", &SignalProcessor::Config::max_frequency)
        .def_readwrite("use_gpu", &SignalProcessor::Config::use_gpu);

    py::class_<SignalProcessor>(m, "SignalProcessor")
        .def(py::init<const SignalProcessor::Config&>())
        .def("process", [](SignalProcessor& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            MatrixXr out;
            self.process(in, out);
            return eigen_to_numpy(out);
        })
        .def("apply_bandpass", [](SignalProcessor& self, py::array_t<Real> signal) {
            MatrixXr sig = numpy_to_eigen(signal);
            self.apply_bandpass(sig);
            return eigen_to_numpy(sig);
        })
        .def("gcc_phat", [](SignalProcessor& self,
                           py::array_t<Real> a, py::array_t<Real> b) {
            auto buf_a = a.request();
            auto buf_b = b.request();

            VectorXr vec_a = Eigen::Map<VectorXr>(
                static_cast<Real*>(buf_a.ptr), buf_a.shape[0]);
            VectorXr vec_b = Eigen::Map<VectorXr>(
                static_cast<Real*>(buf_b.ptr), buf_b.shape[0]);

            VectorXr result = self.gcc_phat(vec_a, vec_b);

            py::array_t<Real> output(result.size());
            std::memcpy(output.mutable_data(), result.data(),
                       result.size() * sizeof(Real));
            return output;
        })
        .def("compute_energy", [](SignalProcessor& self, py::array_t<Real> signal) {
            auto buf = signal.request();
            VectorXr vec = Eigen::Map<VectorXr>(
                static_cast<Real*>(buf.ptr), buf.shape[0]);
            return self.compute_energy(vec);
        });

    // ========================================================================
    // Array Geometry
    // ========================================================================

    py::class_<ArrayGeometry>(m, "ArrayGeometry")
        .def(py::init<>())
        .def_static("circular", &ArrayGeometry::circular)
        .def_static("spherical", &ArrayGeometry::spherical)
        .def_static("planar", &ArrayGeometry::planar)
        .def_static("linear", &ArrayGeometry::linear)
        .def("add_microphone", &ArrayGeometry::add_microphone)
        .def("get_position", &ArrayGeometry::get_position)
        .def("num_mics", &ArrayGeometry::num_mics)
        .def("steering_vector", [](const ArrayGeometry& self,
                                   Real azimuth, Real elevation, Real frequency) {
            VectorXc sv = self.steering_vector(azimuth, elevation, frequency);
            py::array_t<std::complex<Real>> result(sv.size());
            std::memcpy(result.mutable_data(), sv.data(),
                       sv.size() * sizeof(std::complex<Real>));
            return result;
        });

    // ========================================================================
    // Beamformer
    // ========================================================================

    py::enum_<DOAAlgorithm>(m, "DOAAlgorithm")
        .value("DELAY_SUM", DOAAlgorithm::DELAY_SUM)
        .value("MVDR", DOAAlgorithm::MVDR)
        .value("MUSIC", DOAAlgorithm::MUSIC)
        .value("ESPRIT", DOAAlgorithm::ESPRIT)
        .value("SRP_PHAT", DOAAlgorithm::SRP_PHAT);

    py::class_<BeamformerConfig>(m, "BeamformerConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &BeamformerConfig::sample_rate)
        .def_readwrite("fft_size", &BeamformerConfig::fft_size)
        .def_readwrite("num_sources", &BeamformerConfig::num_sources)
        .def_readwrite("min_frequency", &BeamformerConfig::min_frequency)
        .def_readwrite("max_frequency", &BeamformerConfig::max_frequency)
        .def_readwrite("azimuth_resolution", &BeamformerConfig::azimuth_resolution)
        .def_readwrite("elevation_resolution", &BeamformerConfig::elevation_resolution)
        .def_readwrite("elevation_min", &BeamformerConfig::elevation_min)
        .def_readwrite("elevation_max", &BeamformerConfig::elevation_max)
        .def_readwrite("algorithm", &BeamformerConfig::algorithm)
        .def_readwrite("use_gpu", &BeamformerConfig::use_gpu);

    py::class_<Beamformer>(m, "Beamformer")
        .def(py::init<const ArrayGeometry&, const BeamformerConfig&>())
        .def("estimate_doa", [](Beamformer& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            return self.estimate_doa(in);
        })
        .def("compute_spatial_spectrum", [](Beamformer& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            MatrixXr spectrum;
            self.compute_spatial_spectrum(in, spectrum);
            return eigen_to_numpy(spectrum);
        })
        .def("steer", [](Beamformer& self, py::array_t<Real> input,
                        Real azimuth, Real elevation) {
            MatrixXr in = numpy_to_eigen(input);
            VectorXr out;
            self.steer(in, azimuth, elevation, out);

            py::array_t<Real> result(out.size());
            std::memcpy(result.mutable_data(), out.data(),
                       out.size() * sizeof(Real));
            return result;
        });

    // ========================================================================
    // TDOA Engine
    // ========================================================================

    py::enum_<TDOAMethod>(m, "TDOAMethod")
        .value("GCC_PHAT", TDOAMethod::GCC_PHAT)
        .value("GCC_SCOT", TDOAMethod::GCC_SCOT)
        .value("GCC_ML", TDOAMethod::GCC_ML)
        .value("DIRECT_CORR", TDOAMethod::DIRECT_CORR);

    py::class_<TDOAConfig>(m, "TDOAConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &TDOAConfig::sample_rate)
        .def_readwrite("fft_size", &TDOAConfig::fft_size)
        .def_readwrite("max_delay_seconds", &TDOAConfig::max_delay_seconds)
        .def_readwrite("method", &TDOAConfig::method)
        .def_readwrite("min_correlation", &TDOAConfig::min_correlation)
        .def_readwrite("use_interpolation", &TDOAConfig::use_interpolation)
        .def_readwrite("use_gpu", &TDOAConfig::use_gpu);

    py::class_<TDOAMeasurement>(m, "TDOAMeasurement")
        .def(py::init<>())
        .def_readwrite("mic_i", &TDOAMeasurement::mic_i)
        .def_readwrite("mic_j", &TDOAMeasurement::mic_j)
        .def_readwrite("delay_samples", &TDOAMeasurement::delay_samples)
        .def_readwrite("delay_seconds", &TDOAMeasurement::delay_seconds)
        .def_readwrite("confidence", &TDOAMeasurement::confidence);

    py::class_<TDOAEngine>(m, "TDOAEngine")
        .def(py::init<const ArrayGeometry&, const TDOAConfig&>())
        .def("estimate_tdoa", [](TDOAEngine& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            return self.estimate_tdoa(in);
        })
        .def("localize", &TDOAEngine::localize)
        .def("localize_sources", [](TDOAEngine& self, py::array_t<Real> input,
                                    size_t num_sources) {
            MatrixXr in = numpy_to_eigen(input);
            return self.localize_sources(in, num_sources);
        }, py::arg("input"), py::arg("num_sources") = 1)
        .def("to_spherical", &TDOAEngine::to_spherical)
        .def("gcc_phat", [](TDOAEngine& self,
                           py::array_t<Real> a, py::array_t<Real> b) {
            auto buf_a = a.request();
            auto buf_b = b.request();

            VectorXr vec_a = Eigen::Map<VectorXr>(
                static_cast<Real*>(buf_a.ptr), buf_a.shape[0]);
            VectorXr vec_b = Eigen::Map<VectorXr>(
                static_cast<Real*>(buf_b.ptr), buf_b.shape[0]);

            VectorXr result = self.gcc_phat(vec_a, vec_b);

            py::array_t<Real> output(result.size());
            std::memcpy(output.mutable_data(), result.data(),
                       result.size() * sizeof(Real));
            return output;
        });

    // ========================================================================
    // TDOA Tracker
    // ========================================================================

    py::class_<TDOATracker::TrackConfig>(m, "TrackConfig")
        .def(py::init<>())
        .def_readwrite("process_noise", &TDOATracker::TrackConfig::process_noise)
        .def_readwrite("measurement_noise", &TDOATracker::TrackConfig::measurement_noise)
        .def_readwrite("association_threshold", &TDOATracker::TrackConfig::association_threshold)
        .def_readwrite("max_missed_frames", &TDOATracker::TrackConfig::max_missed_frames)
        .def_readwrite("min_hits_to_confirm", &TDOATracker::TrackConfig::min_hits_to_confirm);

    py::class_<TDOATracker::Track>(m, "Track")
        .def_readwrite("id", &TDOATracker::Track::id)
        .def_readwrite("position", &TDOATracker::Track::position)
        .def_readwrite("velocity", &TDOATracker::Track::velocity)
        .def_readwrite("confidence", &TDOATracker::Track::confidence)
        .def_readwrite("age", &TDOATracker::Track::age)
        .def_readwrite("confirmed", &TDOATracker::Track::confirmed);

    py::class_<TDOATracker>(m, "TDOATracker")
        .def(py::init<const ArrayGeometry&, const TDOAConfig&,
                      const TDOATracker::TrackConfig&>(),
             py::arg("geometry"),
             py::arg("tdoa_config"),
             py::arg("track_config") = TDOATracker::TrackConfig())
        .def("update", [](TDOATracker& self, py::array_t<Real> input) {
            MatrixXr in = numpy_to_eigen(input);
            return self.update(in);
        })
        .def("get_all_tracks", &TDOATracker::get_all_tracks)
        .def("clear", &TDOATracker::clear);

    // ========================================================================
    // Version info
    // ========================================================================

    m.attr("__version__") = "1.0.0";

#ifdef USE_CUDA
    m.attr("CUDA_AVAILABLE") = true;
#else
    m.attr("CUDA_AVAILABLE") = false;
#endif
}
