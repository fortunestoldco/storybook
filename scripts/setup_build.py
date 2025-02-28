import os
import sys
import subprocess
from pathlib import Path

def setup_build_environment():
    """Set up the build environment for llama-cpp-python."""
    # Set environment variables for optimized build
    build_flags = [
        "-DLLAMA_CUBLAS=OFF",
        "-DLLAMA_METAL=OFF",
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++"
    ]
    os.environ["CMAKE_ARGS"] = " ".join(build_flags)
    
    if sys.platform == "win32":
        # Windows-specific settings
        os.environ["CMAKE_GENERATOR"] = "Visual Studio 16 2019"
        print("Checking for Visual Studio Build Tools...")
        if not check_vs_buildtools():
            print("Please install Visual Studio Build Tools with C++ workload")
            sys.exit(1)
    else:
        # Linux/macOS settings
        os.environ["CMAKE_GENERATOR"] = "Unix Makefiles"
        print("Checking for GCC/G++...")
        if not check_gcc():
            print("Please install build-essential or equivalent")
            sys.exit(1)
    
    try:
        # Install build dependencies first
        subprocess.run([sys.executable, "-m", "pip", "install", "cmake", "ninja"], check=True)
        # Then install project requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"], check=True)
        print("Build environment setup complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up build environment: {e}")
        sys.exit(1)

def check_vs_buildtools():
    """Check if Visual Studio Build Tools are installed."""
    if sys.platform != "win32":
        return True
    
    vswhere = (
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\"
        "Installer\\vswhere.exe"
    )
    if not os.path.exists(vswhere):
        return False
    
    try:
        result = subprocess.run(
            [vswhere, "-latest", "-property", "installationPath"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def check_gcc():
    """Check if GCC/G++ are installed."""
    try:
        subprocess.run(["gcc", "--version"], capture_output=True)
        subprocess.run(["g++", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    setup_build_environment()