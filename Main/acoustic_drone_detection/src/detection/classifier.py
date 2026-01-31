"""
Drone Classification Module

Provides ML-based drone type classification:
- CNN for spectrogram classification
- Feature extraction (MFCC, spectral features)
- Ensemble classification
- Confidence estimation
"""

import numpy as np
from scipy.signal import spectrogram
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import os


class DroneType(Enum):
    """Supported drone types."""
    QUADCOPTER_SMALL = "quadcopter_small"
    QUADCOPTER_MEDIUM = "quadcopter_medium"
    QUADCOPTER_LARGE = "quadcopter_large"
    HEXACOPTER = "hexacopter"
    OCTOCOPTER = "octocopter"
    FPV_RACING = "fpv_racing"
    FIXED_WING = "fixed_wing"
    HELICOPTER = "helicopter"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of drone classification."""
    primary_class: DroneType
    primary_confidence: float
    all_probabilities: Dict[DroneType, float]
    features_used: List[str]
    timestamp: float = 0.0


@dataclass
class ClassifierConfig:
    """Classifier configuration."""
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    min_confidence: float = 0.3
    use_ensemble: bool = True


class FeatureExtractor:
    """
    Extract features from audio for classification.

    Supports MFCC, spectral features, and spectrogram features.
    """

    def __init__(self, sample_rate: int = 48000, config: Optional[ClassifierConfig] = None):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Sample rate in Hz.
            config: Classifier configuration.
        """
        self._sample_rate = sample_rate
        self._config = config or ClassifierConfig()

    def extract_mfcc(self, data: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features.

        Args:
            data: Audio signal.

        Returns:
            MFCC feature array.
        """
        # Simple MFCC computation
        n_fft = self._config.n_fft
        hop_length = self._config.hop_length
        n_mfcc = self._config.n_mfcc

        # Compute power spectrum
        f, t, Sxx = spectrogram(
            data, fs=self._sample_rate,
            nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Create mel filterbank
        n_mels = self._config.n_mels
        mel_filters = self._create_mel_filterbank(len(f), n_mels, f[-1])

        # Apply filterbank
        mel_spec = np.dot(mel_filters, Sxx)

        # Log compression
        mel_spec = np.log(mel_spec + 1e-12)

        # DCT for MFCC
        mfcc = np.zeros((n_mfcc, mel_spec.shape[1]))
        for i in range(mel_spec.shape[1]):
            mfcc[:, i] = self._dct(mel_spec[:, i], n_mfcc)

        return mfcc

    def _create_mel_filterbank(
        self,
        n_freq: int,
        n_mels: int,
        max_freq: float
    ) -> np.ndarray:
        """Create mel filterbank."""
        mel_low = 0
        mel_high = 2595 * np.log10(1 + max_freq / 700)

        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        bin_points = np.floor((n_freq * 2) * hz_points / (max_freq * 2)).astype(int)

        filters = np.zeros((n_mels, n_freq))

        for i in range(n_mels):
            for j in range(int(bin_points[i]), int(bin_points[i + 1])):
                if j < n_freq and bin_points[i + 1] != bin_points[i]:
                    filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(int(bin_points[i + 1]), int(bin_points[i + 2])):
                if j < n_freq and bin_points[i + 2] != bin_points[i + 1]:
                    filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

        return filters

    @staticmethod
    def _dct(x: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Compute DCT-II."""
        n = len(x)
        result = np.zeros(n_coeffs)
        for k in range(n_coeffs):
            result[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(n) + 1) / (2 * n)))
        return result * np.sqrt(2 / n)

    def extract_spectral_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features.

        Args:
            data: Audio signal.

        Returns:
            Dictionary of spectral features.
        """
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(data))
        frequencies = np.fft.rfftfreq(len(data), 1 / self._sample_rate)

        # Normalize
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-12)

        # Spectral centroid
        centroid = np.sum(frequencies * spectrum_norm)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum_norm))

        # Spectral rolloff (95%)
        cumsum = np.cumsum(spectrum_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.95)
        rolloff = frequencies[min(rolloff_idx, len(frequencies) - 1)]

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-12)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-12)

        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(data)))) / (2 * len(data))

        return {
            'spectral_centroid': centroid,
            'spectral_bandwidth': bandwidth,
            'spectral_rolloff': rolloff,
            'spectral_flatness': flatness,
            'zero_crossing_rate': zcr,
            'rms_energy': np.sqrt(np.mean(data ** 2))
        }

    def extract_all_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract all features as a single vector.

        Args:
            data: Audio signal.

        Returns:
            Feature vector.
        """
        # MFCC features (mean and std)
        mfcc = self.extract_mfcc(data)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Spectral features
        spectral = self.extract_spectral_features(data)
        spectral_vec = np.array(list(spectral.values()))

        # Combine
        features = np.concatenate([mfcc_mean, mfcc_std, spectral_vec])

        return features


class RuleBasedClassifier:
    """
    Rule-based drone classifier.

    Uses acoustic signatures and heuristics for classification.
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize rule-based classifier.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self._sample_rate = sample_rate

        # Define frequency signatures for different drone types
        self._signatures = {
            DroneType.QUADCOPTER_SMALL: {
                'motor_range': (300, 600),
                'prop_range': (1000, 4000),
                'expected_harmonics': 4
            },
            DroneType.QUADCOPTER_MEDIUM: {
                'motor_range': (200, 400),
                'prop_range': (800, 3000),
                'expected_harmonics': 4
            },
            DroneType.QUADCOPTER_LARGE: {
                'motor_range': (100, 250),
                'prop_range': (500, 2000),
                'expected_harmonics': 4
            },
            DroneType.FPV_RACING: {
                'motor_range': (500, 1000),
                'prop_range': (2000, 6000),
                'expected_harmonics': 3
            },
            DroneType.HEXACOPTER: {
                'motor_range': (150, 350),
                'prop_range': (600, 2500),
                'expected_harmonics': 6
            },
            DroneType.FIXED_WING: {
                'motor_range': (50, 150),
                'prop_range': (200, 1000),
                'expected_harmonics': 2
            }
        }

    def classify(
        self,
        dominant_frequencies: np.ndarray,
        spectral_features: Dict[str, float]
    ) -> ClassificationResult:
        """
        Classify based on rules.

        Args:
            dominant_frequencies: Array of dominant frequencies.
            spectral_features: Dictionary of spectral features.

        Returns:
            ClassificationResult object.
        """
        if len(dominant_frequencies) == 0:
            return ClassificationResult(
                primary_class=DroneType.UNKNOWN,
                primary_confidence=0.0,
                all_probabilities={DroneType.UNKNOWN: 1.0},
                features_used=['frequency_analysis']
            )

        scores = {}

        for drone_type, signature in self._signatures.items():
            score = 0.0

            # Check motor frequency range
            motor_freqs = dominant_frequencies[
                (dominant_frequencies >= signature['motor_range'][0]) &
                (dominant_frequencies <= signature['motor_range'][1])
            ]
            if len(motor_freqs) > 0:
                score += 0.4

            # Check propeller frequency range
            prop_freqs = dominant_frequencies[
                (dominant_frequencies >= signature['prop_range'][0]) &
                (dominant_frequencies <= signature['prop_range'][1])
            ]
            if len(prop_freqs) > 0:
                score += 0.3

            # Check harmonic structure
            if len(dominant_frequencies) >= 2:
                # Check for harmonic relationships
                f0 = dominant_frequencies[0]
                harmonic_matches = 0
                for f in dominant_frequencies[1:]:
                    ratio = f / f0
                    if abs(ratio - round(ratio)) < 0.1:
                        harmonic_matches += 1

                harmonic_score = harmonic_matches / signature['expected_harmonics']
                score += 0.3 * min(1.0, harmonic_score)

            scores[drone_type] = score

        # Add unknown
        scores[DroneType.UNKNOWN] = 0.1

        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            probabilities = {k: v / total for k, v in scores.items()}
        else:
            probabilities = {DroneType.UNKNOWN: 1.0}

        # Find primary class
        primary_class = max(probabilities, key=probabilities.get)
        primary_confidence = probabilities[primary_class]

        return ClassificationResult(
            primary_class=primary_class,
            primary_confidence=primary_confidence,
            all_probabilities=probabilities,
            features_used=['frequency_analysis', 'harmonic_analysis']
        )


class MLClassifier:
    """
    Machine learning based classifier.

    Uses pre-trained models for classification.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML classifier.

        Args:
            model_path: Path to pre-trained model.
        """
        self._model = None
        self._model_loaded = False
        self._classes = list(DroneType)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load pre-trained model."""
        try:
            import torch
            self._model = torch.load(model_path, map_location='cpu')
            self._model.eval()
            self._model_loaded = True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._model_loaded = False

    def classify(self, features: np.ndarray) -> ClassificationResult:
        """
        Classify using ML model.

        Args:
            features: Feature vector.

        Returns:
            ClassificationResult object.
        """
        if not self._model_loaded:
            # Return uniform distribution if no model
            prob = 1.0 / len(self._classes)
            return ClassificationResult(
                primary_class=DroneType.UNKNOWN,
                primary_confidence=prob,
                all_probabilities={c: prob for c in self._classes},
                features_used=['ml_features']
            )

        try:
            import torch
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                output = self._model(input_tensor)
                probabilities = torch.softmax(output, dim=1).numpy()[0]

            prob_dict = {
                self._classes[i]: float(probabilities[i])
                for i in range(len(self._classes))
            }

            primary_idx = np.argmax(probabilities)
            primary_class = self._classes[primary_idx]
            primary_confidence = float(probabilities[primary_idx])

            return ClassificationResult(
                primary_class=primary_class,
                primary_confidence=primary_confidence,
                all_probabilities=prob_dict,
                features_used=['ml_features']
            )

        except Exception as e:
            print(f"Classification error: {e}")
            return ClassificationResult(
                primary_class=DroneType.UNKNOWN,
                primary_confidence=0.0,
                all_probabilities={DroneType.UNKNOWN: 1.0},
                features_used=['ml_features']
            )


class DroneClassifier:
    """
    Main drone classification engine.

    Combines rule-based and ML classification methods.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        config: Optional[ClassifierConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize drone classifier.

        Args:
            sample_rate: Sample rate in Hz.
            config: Classifier configuration.
            model_path: Path to ML model.
        """
        self._sample_rate = sample_rate
        self._config = config or ClassifierConfig()

        self._feature_extractor = FeatureExtractor(sample_rate, config)
        self._rule_classifier = RuleBasedClassifier(sample_rate)
        self._ml_classifier = MLClassifier(model_path)

        self._lock = threading.Lock()

    def classify(
        self,
        data: np.ndarray,
        dominant_frequencies: Optional[np.ndarray] = None,
        timestamp: float = 0.0
    ) -> ClassificationResult:
        """
        Classify drone type.

        Args:
            data: Audio signal.
            dominant_frequencies: Pre-computed dominant frequencies.
            timestamp: Classification timestamp.

        Returns:
            ClassificationResult object.
        """
        if data.ndim > 1:
            data = data.mean(axis=0)

        # Extract features
        features = self._feature_extractor.extract_all_features(data)
        spectral_features = self._feature_extractor.extract_spectral_features(data)

        # Get dominant frequencies if not provided
        if dominant_frequencies is None:
            from scipy.signal import find_peaks
            spectrum = np.abs(np.fft.rfft(data))
            frequencies = np.fft.rfftfreq(len(data), 1 / self._sample_rate)
            peaks, _ = find_peaks(spectrum, height=np.mean(spectrum) + 2 * np.std(spectrum))
            dominant_frequencies = frequencies[peaks][:10]

        # Rule-based classification
        rule_result = self._rule_classifier.classify(dominant_frequencies, spectral_features)

        # ML classification
        ml_result = self._ml_classifier.classify(features)

        # Combine results (ensemble)
        if self._config.use_ensemble and ml_result.primary_confidence > 0.1:
            # Weighted average of probabilities
            combined_probs = {}
            rule_weight = 0.4
            ml_weight = 0.6

            all_classes = set(rule_result.all_probabilities.keys()) | set(ml_result.all_probabilities.keys())

            for cls in all_classes:
                rule_prob = rule_result.all_probabilities.get(cls, 0.0)
                ml_prob = ml_result.all_probabilities.get(cls, 0.0)
                combined_probs[cls] = rule_weight * rule_prob + ml_weight * ml_prob

            primary_class = max(combined_probs, key=combined_probs.get)
            primary_confidence = combined_probs[primary_class]

            result = ClassificationResult(
                primary_class=primary_class,
                primary_confidence=primary_confidence,
                all_probabilities=combined_probs,
                features_used=['frequency_analysis', 'harmonic_analysis', 'ml_features'],
                timestamp=timestamp
            )
        else:
            result = rule_result
            result.timestamp = timestamp

        return result

    def get_top_k(
        self,
        result: ClassificationResult,
        k: int = 3
    ) -> List[Tuple[DroneType, float]]:
        """
        Get top-k classification results.

        Args:
            result: Classification result.
            k: Number of top results.

        Returns:
            List of (DroneType, probability) tuples.
        """
        sorted_probs = sorted(
            result.all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_probs[:k]
