import os
import sys
import subprocess
from pathlib import Path

def setup_build_environment():
    """Set up the build environment for llama-cpp-python."""
    # Set environment variables for optimized build
    os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF"
    
    if sys.platform == "win32":
        # Windows-specific settings
        os.environ["CMAKE_GENERATOR"] = "Visual Studio 16 2019"
    else:
        # Linux/macOS settings
        os.environ["CMAKE_GENERATOR"] = "Unix Makefiles"
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"], check=True)
        print("Build environment setup complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up build environment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_build_environment()