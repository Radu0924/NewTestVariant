#!/usr/bin/env python
"""
Management script for the Acoustic Drone Detection System.
usage: manage.py [gui|cli|tests]
"""
import sys
import subprocess
from pathlib import Path

def run_gui():
    """Run the GUI application."""
    print("Starting GUI...")
    try:
        from src.gui.main_window import main
        main()
    except ImportError as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def run_cli():
    """Run the CLI application."""
    print("Starting CLI mode...")
    try:
        from src.main import main
        sys.argv = [sys.argv[0]] # Clear manage.py args for the CLI
        main() 
    except ImportError as e:
        print(f"Error starting CLI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite."""
    print("Running tests...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/"], check=True)
    except subprocess.CalledProcessError:
        print("Tests failed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

def print_help():
    print("Usage: python manage.py <command>")
    print("Commands:")
    print("  gui      Start the Graphical User Interface")
    print("  cli      Start the Command Line Interface")
    print("  tests    Run the test suite")

def main():
    # Add src to python path to look for modules
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "gui":
        run_gui()
    elif command == "cli":
        run_cli()
    elif command == "tests" or command == "test":
        run_tests()
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
