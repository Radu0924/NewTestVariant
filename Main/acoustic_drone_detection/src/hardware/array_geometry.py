"""
Microphone Array Geometry Module

Defines and manages microphone array configurations:
- Predefined geometries (circular, spherical, planar, linear)
- Custom geometry support
- Geometry validation
- Import/export from files
"""

import numpy as np
import yaml
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import math


@dataclass
class MicrophonePosition:
    """Position of a single microphone."""
    id: int
    x: float  # meters
    y: float  # meters
    z: float  # meters
    orientation: Optional[Tuple[float, float, float]] = None  # (roll, pitch, yaw)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])


@dataclass
class ArrayGeometry:
    """Complete microphone array geometry."""
    name: str
    geometry_type: str  # circular, spherical, planar, linear, custom
    num_microphones: int
    positions: List[MicrophonePosition]
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_mic: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_positions_array(self) -> np.ndarray:
        """Get positions as numpy array (N x 3)."""
        return np.array([mic.to_array() for mic in self.positions])

    def get_position(self, mic_id: int) -> Optional[MicrophonePosition]:
        """Get position for specific microphone."""
        for mic in self.positions:
            if mic.id == mic_id:
                return mic
        return None


class GeometryGenerator:
    """
    Generates standard microphone array geometries.

    Supports circular, spherical, planar, and linear arrays.
    """

    @staticmethod
    def circular(
        num_mics: int,
        radius: float = 0.1,
        center: Tuple[float, float, float] = (0, 0, 0),
        plane: str = "xy"
    ) -> ArrayGeometry:
        """
        Generate circular array.

        Args:
            num_mics: Number of microphones.
            radius: Array radius in meters.
            center: Center position.
            plane: Plane for the array (xy, xz, yz).

        Returns:
            ArrayGeometry object.
        """
        positions = []

        for i in range(num_mics):
            angle = 2 * np.pi * i / num_mics

            if plane == "xy":
                x = radius * np.cos(angle) + center[0]
                y = radius * np.sin(angle) + center[1]
                z = center[2]
            elif plane == "xz":
                x = radius * np.cos(angle) + center[0]
                y = center[1]
                z = radius * np.sin(angle) + center[2]
            else:  # yz
                x = center[0]
                y = radius * np.cos(angle) + center[1]
                z = radius * np.sin(angle) + center[2]

            positions.append(MicrophonePosition(id=i, x=x, y=y, z=z))

        return ArrayGeometry(
            name=f"circular_{num_mics}_mic",
            geometry_type="circular",
            num_microphones=num_mics,
            positions=positions,
            center=np.array(center),
            metadata={'radius': radius, 'plane': plane}
        )

    @staticmethod
    def spherical(
        num_mics: int,
        radius: float = 0.1,
        center: Tuple[float, float, float] = (0, 0, 0)
    ) -> ArrayGeometry:
        """
        Generate spherical array using Fibonacci lattice.

        Args:
            num_mics: Number of microphones.
            radius: Sphere radius in meters.
            center: Center position.

        Returns:
            ArrayGeometry object.
        """
        positions = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(num_mics):
            theta = 2 * np.pi * i / golden_ratio
            phi = np.arccos(1 - 2 * (i + 0.5) / num_mics)

            x = radius * np.sin(phi) * np.cos(theta) + center[0]
            y = radius * np.sin(phi) * np.sin(theta) + center[1]
            z = radius * np.cos(phi) + center[2]

            positions.append(MicrophonePosition(id=i, x=x, y=y, z=z))

        return ArrayGeometry(
            name=f"spherical_{num_mics}_mic",
            geometry_type="spherical",
            num_microphones=num_mics,
            positions=positions,
            center=np.array(center),
            metadata={'radius': radius}
        )

    @staticmethod
    def planar_rectangular(
        rows: int,
        cols: int,
        spacing: float = 0.1,
        center: Tuple[float, float, float] = (0, 0, 0)
    ) -> ArrayGeometry:
        """
        Generate rectangular planar array.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            spacing: Spacing between microphones in meters.
            center: Center position.

        Returns:
            ArrayGeometry object.
        """
        positions = []

        width = (cols - 1) * spacing
        height = (rows - 1) * spacing

        mic_id = 0
        for row in range(rows):
            for col in range(cols):
                x = col * spacing - width / 2 + center[0]
                y = row * spacing - height / 2 + center[1]
                z = center[2]

                positions.append(MicrophonePosition(id=mic_id, x=x, y=y, z=z))
                mic_id += 1

        return ArrayGeometry(
            name=f"planar_{rows}x{cols}_mic",
            geometry_type="planar",
            num_microphones=rows * cols,
            positions=positions,
            center=np.array(center),
            metadata={'rows': rows, 'cols': cols, 'spacing': spacing}
        )

    @staticmethod
    def linear(
        num_mics: int,
        spacing: float = 0.1,
        axis: str = "x",
        center: Tuple[float, float, float] = (0, 0, 0)
    ) -> ArrayGeometry:
        """
        Generate linear array.

        Args:
            num_mics: Number of microphones.
            spacing: Spacing between microphones in meters.
            axis: Array axis (x, y, z).
            center: Center position.

        Returns:
            ArrayGeometry object.
        """
        positions = []
        length = (num_mics - 1) * spacing

        for i in range(num_mics):
            offset = i * spacing - length / 2

            if axis == "x":
                x = offset + center[0]
                y = center[1]
                z = center[2]
            elif axis == "y":
                x = center[0]
                y = offset + center[1]
                z = center[2]
            else:  # z
                x = center[0]
                y = center[1]
                z = offset + center[2]

            positions.append(MicrophonePosition(id=i, x=x, y=y, z=z))

        return ArrayGeometry(
            name=f"linear_{num_mics}_mic",
            geometry_type="linear",
            num_microphones=num_mics,
            positions=positions,
            center=np.array(center),
            metadata={'spacing': spacing, 'axis': axis}
        )

    @staticmethod
    def custom(positions: List[Tuple[float, float, float]], name: str = "custom") -> ArrayGeometry:
        """
        Create array from custom positions.

        Args:
            positions: List of (x, y, z) tuples.
            name: Array name.

        Returns:
            ArrayGeometry object.
        """
        mic_positions = [
            MicrophonePosition(id=i, x=pos[0], y=pos[1], z=pos[2])
            for i, pos in enumerate(positions)
        ]

        center = np.mean([np.array(pos) for pos in positions], axis=0)

        return ArrayGeometry(
            name=name,
            geometry_type="custom",
            num_microphones=len(positions),
            positions=mic_positions,
            center=center
        )


class GeometryValidator:
    """
    Validates microphone array geometries.

    Checks for spatial aliasing, minimum separation, and other constraints.
    """

    def __init__(
        self,
        min_spacing: float = 0.01,  # 1 cm minimum
        max_spacing: float = 1.0,    # 1 m maximum
        sound_speed: float = 343.0
    ):
        """
        Initialize validator.

        Args:
            min_spacing: Minimum microphone spacing in meters.
            max_spacing: Maximum microphone spacing in meters.
            sound_speed: Speed of sound in m/s.
        """
        self._min_spacing = min_spacing
        self._max_spacing = max_spacing
        self._sound_speed = sound_speed

    def validate(self, geometry: ArrayGeometry) -> Tuple[bool, List[str]]:
        """
        Validate array geometry.

        Args:
            geometry: ArrayGeometry to validate.

        Returns:
            Tuple of (is_valid, list of warnings/errors).
        """
        messages = []
        is_valid = True

        # Check minimum number of microphones
        if geometry.num_microphones < 4:
            messages.append("ERROR: At least 4 microphones required for 3D localization")
            is_valid = False

        # Check microphone spacing
        positions = geometry.get_positions_array()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])

                if distance < self._min_spacing:
                    messages.append(
                        f"WARNING: Microphones {i} and {j} are too close "
                        f"({distance:.3f}m < {self._min_spacing}m)"
                    )

                if distance > self._max_spacing:
                    messages.append(
                        f"WARNING: Microphones {i} and {j} are far apart "
                        f"({distance:.3f}m > {self._max_spacing}m)"
                    )

        # Check for spatial aliasing
        max_distance = self._get_max_mic_distance(positions)
        aliasing_freq = self._sound_speed / (2 * max_distance)

        if aliasing_freq < 8000:
            messages.append(
                f"WARNING: Spatial aliasing above {aliasing_freq:.0f} Hz "
                f"(max spacing: {max_distance:.3f}m)"
            )

        # Check for coplanar configuration
        if self._is_coplanar(positions):
            messages.append(
                "INFO: Array is coplanar - elevation estimation may be limited"
            )

        # Check for collinear configuration
        if self._is_collinear(positions):
            messages.append(
                "WARNING: Array is nearly collinear - 3D localization will be poor"
            )
            is_valid = False

        return is_valid, messages

    def _get_max_mic_distance(self, positions: np.ndarray) -> float:
        """Get maximum distance between any two microphones."""
        max_dist = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                max_dist = max(max_dist, dist)
        return max_dist

    def _is_coplanar(self, positions: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if all positions are in a plane."""
        if len(positions) < 4:
            return True

        # Fit a plane and check residuals
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid

        # SVD to find normal
        _, s, vh = np.linalg.svd(centered)

        # Check if smallest singular value is near zero
        return s[-1] < tolerance

    def _is_collinear(self, positions: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if all positions are on a line."""
        if len(positions) < 3:
            return True

        # Check variance in perpendicular directions
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid

        _, s, _ = np.linalg.svd(centered)

        # Check if two smallest singular values are near zero
        return s[-1] < tolerance and s[-2] < tolerance

    def get_frequency_range(
        self,
        geometry: ArrayGeometry
    ) -> Tuple[float, float]:
        """
        Get effective frequency range for the array.

        Args:
            geometry: Array geometry.

        Returns:
            Tuple of (min_freq, max_freq) in Hz.
        """
        positions = geometry.get_positions_array()

        # Minimum frequency (based on array aperture)
        aperture = self._get_max_mic_distance(positions)
        min_freq = self._sound_speed / (4 * aperture)

        # Maximum frequency (spatial aliasing limit)
        min_distance = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distance = min(min_distance, dist)

        max_freq = self._sound_speed / (2 * min_distance)

        return min_freq, max_freq


class GeometryIO:
    """
    Import/export array geometries from/to files.

    Supports YAML and JSON formats.
    """

    @staticmethod
    def load_yaml(filepath: str) -> ArrayGeometry:
        """
        Load geometry from YAML file.

        Args:
            filepath: Path to YAML file.

        Returns:
            ArrayGeometry object.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return GeometryIO._from_dict(data)

    @staticmethod
    def save_yaml(geometry: ArrayGeometry, filepath: str) -> None:
        """
        Save geometry to YAML file.

        Args:
            geometry: ArrayGeometry to save.
            filepath: Output file path.
        """
        data = GeometryIO._to_dict(geometry)

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def load_json(filepath: str) -> ArrayGeometry:
        """
        Load geometry from JSON file.

        Args:
            filepath: Path to JSON file.

        Returns:
            ArrayGeometry object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return GeometryIO._from_dict(data)

    @staticmethod
    def save_json(geometry: ArrayGeometry, filepath: str) -> None:
        """
        Save geometry to JSON file.

        Args:
            geometry: ArrayGeometry to save.
            filepath: Output file path.
        """
        data = GeometryIO._to_dict(geometry)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _to_dict(geometry: ArrayGeometry) -> Dict:
        """Convert geometry to dictionary."""
        return {
            'name': geometry.name,
            'geometry_type': geometry.geometry_type,
            'num_microphones': geometry.num_microphones,
            'center': geometry.center.tolist(),
            'reference_mic': geometry.reference_mic,
            'positions': [
                {
                    'id': mic.id,
                    'x': mic.x,
                    'y': mic.y,
                    'z': mic.z,
                    'orientation': mic.orientation
                }
                for mic in geometry.positions
            ],
            'metadata': geometry.metadata
        }

    @staticmethod
    def _from_dict(data: Dict) -> ArrayGeometry:
        """Create geometry from dictionary."""
        positions = [
            MicrophonePosition(
                id=pos['id'],
                x=pos['x'],
                y=pos['y'],
                z=pos['z'],
                orientation=pos.get('orientation')
            )
            for pos in data['positions']
        ]

        return ArrayGeometry(
            name=data['name'],
            geometry_type=data['geometry_type'],
            num_microphones=data['num_microphones'],
            positions=positions,
            center=np.array(data.get('center', [0, 0, 0])),
            reference_mic=data.get('reference_mic', 0),
            metadata=data.get('metadata', {})
        )


class ArrayGeometryManager:
    """
    Manages microphone array geometries.

    Provides generation, validation, and persistence of array configurations.
    """

    def __init__(self):
        """Initialize geometry manager."""
        self._geometries: Dict[str, ArrayGeometry] = {}
        self._generator = GeometryGenerator()
        self._validator = GeometryValidator()
        self._current: Optional[ArrayGeometry] = None

    def create_circular(
        self,
        num_mics: int,
        radius: float = 0.1,
        name: Optional[str] = None
    ) -> ArrayGeometry:
        """Create and register a circular array."""
        geometry = self._generator.circular(num_mics, radius)
        if name:
            geometry.name = name
        self._geometries[geometry.name] = geometry
        return geometry

    def create_spherical(
        self,
        num_mics: int,
        radius: float = 0.1,
        name: Optional[str] = None
    ) -> ArrayGeometry:
        """Create and register a spherical array."""
        geometry = self._generator.spherical(num_mics, radius)
        if name:
            geometry.name = name
        self._geometries[geometry.name] = geometry
        return geometry

    def create_planar(
        self,
        rows: int,
        cols: int,
        spacing: float = 0.1,
        name: Optional[str] = None
    ) -> ArrayGeometry:
        """Create and register a planar rectangular array."""
        geometry = self._generator.planar_rectangular(rows, cols, spacing)
        if name:
            geometry.name = name
        self._geometries[geometry.name] = geometry
        return geometry

    def create_linear(
        self,
        num_mics: int,
        spacing: float = 0.1,
        name: Optional[str] = None
    ) -> ArrayGeometry:
        """Create and register a linear array."""
        geometry = self._generator.linear(num_mics, spacing)
        if name:
            geometry.name = name
        self._geometries[geometry.name] = geometry
        return geometry

    def create_custom(
        self,
        positions: List[Tuple[float, float, float]],
        name: str
    ) -> ArrayGeometry:
        """Create and register a custom array."""
        geometry = self._generator.custom(positions, name)
        self._geometries[name] = geometry
        return geometry

    def load(self, filepath: str) -> ArrayGeometry:
        """Load geometry from file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            geometry = GeometryIO.load_yaml(filepath)
        else:
            geometry = GeometryIO.load_json(filepath)

        self._geometries[geometry.name] = geometry
        return geometry

    def save(self, name: str, filepath: str) -> None:
        """Save geometry to file."""
        geometry = self._geometries.get(name)
        if geometry is None:
            raise ValueError(f"Geometry '{name}' not found")

        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            GeometryIO.save_yaml(geometry, filepath)
        else:
            GeometryIO.save_json(geometry, filepath)

    def validate(self, name: str) -> Tuple[bool, List[str]]:
        """Validate a geometry."""
        geometry = self._geometries.get(name)
        if geometry is None:
            return False, [f"Geometry '{name}' not found"]

        return self._validator.validate(geometry)

    def set_current(self, name: str) -> bool:
        """Set the current active geometry."""
        geometry = self._geometries.get(name)
        if geometry:
            self._current = geometry
            return True
        return False

    @property
    def current(self) -> Optional[ArrayGeometry]:
        """Get current active geometry."""
        return self._current

    @property
    def available_geometries(self) -> List[str]:
        """Get list of available geometry names."""
        return list(self._geometries.keys())

    def get(self, name: str) -> Optional[ArrayGeometry]:
        """Get geometry by name."""
        return self._geometries.get(name)
