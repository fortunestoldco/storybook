FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Set build environment variables
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++"
ENV CMAKE_GENERATOR="Unix Makefiles"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Command to run the application
CMD ["uvicorn", "storybook.main:app", "--host", "0.0.0.0", "--port", "8080"]
