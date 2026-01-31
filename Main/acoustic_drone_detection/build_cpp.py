#!/usr/bin/env python3
"""
Build script for C++ core library

Usage:
    python build_cpp.py           # Build with default options
    python build_cpp.py --cuda    # Build with CUDA support
    python build_cpp.py --debug   # Build debug version
    python build_cpp.py --clean   # Clean build directory
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def find_cmake():
    """Find CMake executable."""
    cmake_paths = [
        'cmake',
        r'C:\Program Files\CMake\bin\cmake.exe',
        r'C:\Program Files (x86)\CMake\bin\cmake.exe',
        '/usr/bin/cmake',
        '/usr/local/bin/cmake',
    ]

    for path in cmake_paths:
        try:
            result = subprocess.run([path, '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except FileNotFoundError:
            continue

    return None


def find_visual_studio():
    """Find Visual Studio installation."""
    vswhere = r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

    if os.path.exists(vswhere):
        result = subprocess.run([
            vswhere, '-latest', '-property', 'installationPath'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()

    return None


def build_cpp(args):
    """Build the C++ core library."""
    script_dir = Path(__file__).parent.absolute()
    cpp_dir = script_dir / 'cpp_core'
    build_dir = cpp_dir / 'build'

    # Clean if requested
    if args.clean:
        if build_dir.exists():
            print(f"Cleaning {build_dir}...")
            shutil.rmtree(build_dir)
        return 0

    # Find CMake
    cmake = find_cmake()
    if not cmake:
        print("ERROR: CMake not found. Please install CMake.")
        return 1

    print(f"Using CMake: {cmake}")

    # Create build directory
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure CMake options
    cmake_args = [
        cmake,
        str(cpp_dir),
        f'-DCMAKE_BUILD_TYPE={"Debug" if args.debug else "Release"}',
        f'-DUSE_CUDA={"ON" if args.cuda else "OFF"}',
        '-DBUILD_PYTHON_BINDINGS=ON',
    ]

    # Platform-specific configuration
    if sys.platform == 'win32':
        vs_path = find_visual_studio()
        if vs_path:
            print(f"Using Visual Studio: {vs_path}")
            cmake_args.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64'])
        else:
            cmake_args.extend(['-G', 'Ninja'])
    else:
        cmake_args.extend(['-G', 'Ninja'])

    # Configure
    print("\n=== Configuring ===")
    print(f"Command: {' '.join(cmake_args)}")

    result = subprocess.run(cmake_args, cwd=build_dir)
    if result.returncode != 0:
        print("ERROR: CMake configuration failed")
        return 1

    # Build
    print("\n=== Building ===")
    build_args = [cmake, '--build', '.', '--config',
                  'Debug' if args.debug else 'Release']

    if args.parallel:
        build_args.extend(['--parallel', str(args.parallel)])

    result = subprocess.run(build_args, cwd=build_dir)
    if result.returncode != 0:
        print("ERROR: Build failed")
        return 1

    # Install Python module
    print("\n=== Installing Python module ===")

    # Find the built module
    if sys.platform == 'win32':
        module_name = 'drone_core_py.pyd'
        search_dirs = [
            build_dir / 'Release',
            build_dir / 'Debug',
            build_dir,
        ]
    else:
        module_name = 'drone_core_py.so'
        search_dirs = [build_dir]

    module_path = None
    for search_dir in search_dirs:
        candidate = search_dir / module_name
        if candidate.exists():
            module_path = candidate
            break

    if module_path:
        # Copy to package directory
        dest = script_dir / 'src' / module_name
        shutil.copy2(module_path, dest)
        print(f"Installed: {dest}")

        # Also try to install to site-packages
        try:
            import site
            site_packages = site.getsitepackages()[0]
            dest_site = Path(site_packages) / module_name
            shutil.copy2(module_path, dest_site)
            print(f"Installed to site-packages: {dest_site}")
        except Exception as e:
            print(f"Note: Could not install to site-packages: {e}")
    else:
        print(f"WARNING: Could not find {module_name}")

    print("\n=== Build complete ===")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Build C++ core library')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable CUDA support')
    parser.add_argument('--debug', action='store_true',
                        help='Build debug version')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build directory')
    parser.add_argument('--parallel', '-j', type=int, default=None,
                        help='Number of parallel jobs')

    args = parser.parse_args()
    return build_cpp(args)


if __name__ == '__main__':
    sys.exit(main())
