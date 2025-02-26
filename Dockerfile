# Dockerfile for Novel Generation System
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.5.1

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy pyproject.toml and poetry.lock (if exists)
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment in the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the application
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Set volume for output
VOLUME ["/app/output"]

# Entry point
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["--output", "/app/output"]
