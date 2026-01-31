"""
Acoustic Signatures Database Module

Manages drone acoustic signatures for classification:
- Signature storage and retrieval
- Signature matching
- Database persistence
"""

import numpy as np
import json
import yaml
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
from datetime import datetime


@dataclass
class AcousticSignature:
    """Acoustic signature of a drone type."""
    signature_id: str
    name: str
    drone_type: str
    manufacturer: str = ""
    model: str = ""

    # Frequency characteristics
    fundamental_frequency: float = 0.0
    harmonic_frequencies: List[float] = field(default_factory=list)
    frequency_range: Tuple[float, float] = (100.0, 8000.0)

    # Spectral features
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0

    # MFCC template
    mfcc_template: Optional[np.ndarray] = None

    # Spectral template (normalized)
    spectral_template: Optional[np.ndarray] = None

    # Metadata
    sample_rate: int = 48000
    created_at: str = ""
    updated_at: str = ""
    confidence_threshold: float = 0.5
    notes: str = ""


@dataclass
class SignatureMatch:
    """Result of signature matching."""
    signature: AcousticSignature
    similarity_score: float
    confidence: float
    matched_features: List[str]


class SignatureDatabase:
    """
    Database for storing and matching acoustic signatures.

    Provides CRUD operations and similarity matching.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize signature database.

        Args:
            db_path: Path to database file (JSON or YAML).
        """
        self._db_path = db_path
        self._signatures: Dict[str, AcousticSignature] = {}
        self._index: Dict[str, List[str]] = {}  # Type -> signature IDs

        if db_path and os.path.exists(db_path):
            self.load(db_path)

    def add_signature(self, signature: AcousticSignature) -> str:
        """
        Add a signature to the database.

        Args:
            signature: Acoustic signature to add.

        Returns:
            Signature ID.
        """
        if not signature.signature_id:
            signature.signature_id = self._generate_id(signature)

        if not signature.created_at:
            signature.created_at = datetime.now().isoformat()

        signature.updated_at = datetime.now().isoformat()

        self._signatures[signature.signature_id] = signature

        # Update index
        if signature.drone_type not in self._index:
            self._index[signature.drone_type] = []
        if signature.signature_id not in self._index[signature.drone_type]:
            self._index[signature.drone_type].append(signature.signature_id)

        return signature.signature_id

    def get_signature(self, signature_id: str) -> Optional[AcousticSignature]:
        """Get signature by ID."""
        return self._signatures.get(signature_id)

    def get_by_type(self, drone_type: str) -> List[AcousticSignature]:
        """Get all signatures for a drone type."""
        sig_ids = self._index.get(drone_type, [])
        return [self._signatures[sid] for sid in sig_ids if sid in self._signatures]

    def remove_signature(self, signature_id: str) -> bool:
        """Remove a signature from the database."""
        if signature_id not in self._signatures:
            return False

        signature = self._signatures[signature_id]

        # Remove from index
        if signature.drone_type in self._index:
            if signature_id in self._index[signature.drone_type]:
                self._index[signature.drone_type].remove(signature_id)

        del self._signatures[signature_id]
        return True

    def update_signature(
        self,
        signature_id: str,
        **kwargs
    ) -> Optional[AcousticSignature]:
        """
        Update a signature.

        Args:
            signature_id: Signature to update.
            **kwargs: Fields to update.

        Returns:
            Updated signature or None.
        """
        if signature_id not in self._signatures:
            return None

        signature = self._signatures[signature_id]
        old_type = signature.drone_type

        for key, value in kwargs.items():
            if hasattr(signature, key):
                setattr(signature, key, value)

        signature.updated_at = datetime.now().isoformat()

        # Update index if type changed
        if signature.drone_type != old_type:
            if old_type in self._index and signature_id in self._index[old_type]:
                self._index[old_type].remove(signature_id)

            if signature.drone_type not in self._index:
                self._index[signature.drone_type] = []
            self._index[signature.drone_type].append(signature_id)

        return signature

    def match(
        self,
        spectrum: np.ndarray,
        dominant_frequencies: np.ndarray,
        mfcc: Optional[np.ndarray] = None,
        top_k: int = 3
    ) -> List[SignatureMatch]:
        """
        Match observed features against signatures.

        Args:
            spectrum: Observed spectrum (normalized).
            dominant_frequencies: Dominant frequencies.
            mfcc: MFCC features (optional).
            top_k: Number of top matches to return.

        Returns:
            List of SignatureMatch objects.
        """
        matches = []

        for sig_id, signature in self._signatures.items():
            score, matched_features = self._compute_similarity(
                signature, spectrum, dominant_frequencies, mfcc
            )

            if score > 0:
                matches.append(SignatureMatch(
                    signature=signature,
                    similarity_score=score,
                    confidence=min(1.0, score / signature.confidence_threshold),
                    matched_features=matched_features
                ))

        # Sort by score
        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        return matches[:top_k]

    def _compute_similarity(
        self,
        signature: AcousticSignature,
        spectrum: np.ndarray,
        dominant_frequencies: np.ndarray,
        mfcc: Optional[np.ndarray]
    ) -> Tuple[float, List[str]]:
        """Compute similarity score between observation and signature."""
        score = 0.0
        matched_features = []

        # Frequency matching
        if len(dominant_frequencies) > 0 and signature.fundamental_frequency > 0:
            # Check fundamental
            f0 = signature.fundamental_frequency
            closest = dominant_frequencies[
                np.argmin(np.abs(dominant_frequencies - f0))
            ]
            if abs(closest - f0) < f0 * 0.1:  # 10% tolerance
                score += 0.3
                matched_features.append('fundamental_frequency')

            # Check harmonics
            harmonic_matches = 0
            for h in signature.harmonic_frequencies:
                if len(dominant_frequencies) > 0:
                    closest = dominant_frequencies[
                        np.argmin(np.abs(dominant_frequencies - h))
                    ]
                    if abs(closest - h) < h * 0.1:
                        harmonic_matches += 1

            if len(signature.harmonic_frequencies) > 0:
                harmonic_score = harmonic_matches / len(signature.harmonic_frequencies)
                score += 0.3 * harmonic_score
                if harmonic_score > 0.5:
                    matched_features.append('harmonic_frequencies')

        # Spectral template matching
        if signature.spectral_template is not None and len(spectrum) > 0:
            template = signature.spectral_template

            # Resize if needed
            if len(template) != len(spectrum):
                template = np.interp(
                    np.linspace(0, 1, len(spectrum)),
                    np.linspace(0, 1, len(template)),
                    template
                )

            # Normalize
            spectrum_norm = spectrum / (np.linalg.norm(spectrum) + 1e-12)
            template_norm = template / (np.linalg.norm(template) + 1e-12)

            # Correlation
            correlation = np.abs(np.sum(spectrum_norm * template_norm))
            score += 0.2 * correlation
            if correlation > 0.7:
                matched_features.append('spectral_template')

        # MFCC matching
        if mfcc is not None and signature.mfcc_template is not None:
            mfcc_mean = np.mean(mfcc, axis=1) if mfcc.ndim > 1 else mfcc
            template_mean = np.mean(signature.mfcc_template, axis=1) \
                if signature.mfcc_template.ndim > 1 else signature.mfcc_template

            # Resize if needed
            min_len = min(len(mfcc_mean), len(template_mean))
            mfcc_mean = mfcc_mean[:min_len]
            template_mean = template_mean[:min_len]

            # Cosine similarity
            norm_mfcc = np.linalg.norm(mfcc_mean) + 1e-12
            norm_template = np.linalg.norm(template_mean) + 1e-12
            similarity = np.dot(mfcc_mean, template_mean) / (norm_mfcc * norm_template)

            score += 0.2 * max(0, similarity)
            if similarity > 0.8:
                matched_features.append('mfcc_template')

        return score, matched_features

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save database to file.

        Args:
            filepath: Output file path.
        """
        path = filepath or self._db_path
        if not path:
            return

        data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'signatures': []
        }

        for sig_id, signature in self._signatures.items():
            sig_data = {
                'signature_id': signature.signature_id,
                'name': signature.name,
                'drone_type': signature.drone_type,
                'manufacturer': signature.manufacturer,
                'model': signature.model,
                'fundamental_frequency': signature.fundamental_frequency,
                'harmonic_frequencies': signature.harmonic_frequencies,
                'frequency_range': list(signature.frequency_range),
                'spectral_centroid': signature.spectral_centroid,
                'spectral_bandwidth': signature.spectral_bandwidth,
                'spectral_rolloff': signature.spectral_rolloff,
                'sample_rate': signature.sample_rate,
                'created_at': signature.created_at,
                'updated_at': signature.updated_at,
                'confidence_threshold': signature.confidence_threshold,
                'notes': signature.notes
            }

            # Handle numpy arrays
            if signature.mfcc_template is not None:
                sig_data['mfcc_template'] = signature.mfcc_template.tolist()
            if signature.spectral_template is not None:
                sig_data['spectral_template'] = signature.spectral_template.tolist()

            data['signatures'].append(sig_data)

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load database from file.

        Args:
            filepath: Input file path.
        """
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

        self._signatures.clear()
        self._index.clear()

        for sig_data in data.get('signatures', []):
            signature = AcousticSignature(
                signature_id=sig_data['signature_id'],
                name=sig_data['name'],
                drone_type=sig_data['drone_type'],
                manufacturer=sig_data.get('manufacturer', ''),
                model=sig_data.get('model', ''),
                fundamental_frequency=sig_data.get('fundamental_frequency', 0.0),
                harmonic_frequencies=sig_data.get('harmonic_frequencies', []),
                frequency_range=tuple(sig_data.get('frequency_range', [100.0, 8000.0])),
                spectral_centroid=sig_data.get('spectral_centroid', 0.0),
                spectral_bandwidth=sig_data.get('spectral_bandwidth', 0.0),
                spectral_rolloff=sig_data.get('spectral_rolloff', 0.0),
                sample_rate=sig_data.get('sample_rate', 48000),
                created_at=sig_data.get('created_at', ''),
                updated_at=sig_data.get('updated_at', ''),
                confidence_threshold=sig_data.get('confidence_threshold', 0.5),
                notes=sig_data.get('notes', '')
            )

            # Handle numpy arrays
            if 'mfcc_template' in sig_data and sig_data['mfcc_template']:
                signature.mfcc_template = np.array(sig_data['mfcc_template'])
            if 'spectral_template' in sig_data and sig_data['spectral_template']:
                signature.spectral_template = np.array(sig_data['spectral_template'])

            self.add_signature(signature)

        self._db_path = filepath

    def _generate_id(self, signature: AcousticSignature) -> str:
        """Generate unique ID for signature."""
        content = f"{signature.name}_{signature.drone_type}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def count(self) -> int:
        """Get number of signatures in database."""
        return len(self._signatures)

    @property
    def all_signatures(self) -> List[AcousticSignature]:
        """Get all signatures."""
        return list(self._signatures.values())

    @property
    def drone_types(self) -> List[str]:
        """Get list of drone types in database."""
        return list(self._index.keys())


def create_signature_from_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    name: str,
    drone_type: str,
    **metadata
) -> AcousticSignature:
    """
    Create a signature from audio recording.

    Args:
        audio_data: Audio signal.
        sample_rate: Sample rate in Hz.
        name: Signature name.
        drone_type: Type of drone.
        **metadata: Additional metadata.

    Returns:
        AcousticSignature object.
    """
    from scipy.signal import find_peaks, spectrogram

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio_data))
    frequencies = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

    # Find dominant frequencies
    threshold = np.mean(spectrum) + 2 * np.std(spectrum)
    peaks, _ = find_peaks(spectrum, height=threshold, distance=10)
    dominant_freqs = frequencies[peaks][:10]

    # Fundamental and harmonics
    fundamental = dominant_freqs[0] if len(dominant_freqs) > 0 else 0.0
    harmonics = dominant_freqs[1:].tolist() if len(dominant_freqs) > 1 else []

    # Spectral features
    spectrum_norm = spectrum / (np.sum(spectrum) + 1e-12)
    centroid = np.sum(frequencies * spectrum_norm)
    bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum_norm))

    cumsum = np.cumsum(spectrum_norm)
    rolloff_idx = np.searchsorted(cumsum, 0.95)
    rolloff = frequencies[min(rolloff_idx, len(frequencies) - 1)]

    return AcousticSignature(
        signature_id='',  # Will be generated
        name=name,
        drone_type=drone_type,
        manufacturer=metadata.get('manufacturer', ''),
        model=metadata.get('model', ''),
        fundamental_frequency=float(fundamental),
        harmonic_frequencies=harmonics,
        spectral_centroid=float(centroid),
        spectral_bandwidth=float(bandwidth),
        spectral_rolloff=float(rolloff),
        spectral_template=spectrum / (np.linalg.norm(spectrum) + 1e-12),
        sample_rate=sample_rate,
        notes=metadata.get('notes', '')
    )
