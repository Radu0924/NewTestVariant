"""
Acoustic Drone Detection System - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="acoustic_drone_detection",
    version="1.0.0",
    author="Drone Detection Team",
    author_email="contact@dronedetection.example.com",
    description="Professional acoustic drone detection, localization, and classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/acoustic-drone-detection",
    project_urls={
        "Bug Tracker": "https://github.com/example/acoustic-drone-detection/issues",
        "Documentation": "https://acoustic-drone-detection.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "Topic :: Security",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "gui": [
            "PyQt6>=6.5.0",
            "pyqtgraph>=0.13.0",
            "vispy>=0.12.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "scikit-learn>=1.2.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.22.0",
            "websockets>=11.0",
        ],
        "export": [
            "pandas>=2.0.0",
            "openpyxl>=3.1.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-qt>=4.2.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "full": [
            "PyQt6>=6.5.0",
            "pyqtgraph>=0.13.0",
            "vispy>=0.12.0",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "scikit-learn>=1.2.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.22.0",
            "websockets>=11.0",
            "pandas>=2.0.0",
            "openpyxl>=3.1.0",
            "librosa>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drone-detect=src.main:main",
            "drone-detect-gui=src.gui.main_window:main",
            "drone-calibrate=scripts.calibrate:main",
            "drone-benchmark=scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    data_files=[
        ("config", [
            "config/default_config.yaml",
        ]),
        ("config/array_configs", [
            "config/array_configs/8_mic_circular.yaml",
            "config/array_configs/12_mic_spherical.yaml",
            "config/array_configs/16_mic_planar.yaml",
        ]),
        ("config/drone_profiles", [
            "config/drone_profiles/dji_mavic.yaml",
            "config/drone_profiles/dji_phantom.yaml",
            "config/drone_profiles/fpv_racing.yaml",
            "config/drone_profiles/fixed_wing.yaml",
        ]),
    ],
)
