# Multi-stage Dockerfile for LLM4CodeSec Benchmark with NVIDIA CUDA support
# Built for optimal performance with CUDA-enabled GPUs

# Build stage - Poetry setup and dependency installation
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 as builder

# Set environment variables for non-interactive builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip3 install --upgrade pip && \
    pip3 install poetry==1.8.3

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies with CUDA support
RUN poetry install --no-dev --no-root && \
    pip install flash-attn==2.7.4.post1 --no-build-isolation \
    rm -rf $POETRY_CACHE_DIR

# Runtime stage - Optimized for GPU performance
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    libssl3 \
    libffi8 \
    libjpeg8 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ .

RUN mkdir -p /app/results /app/data /app/logs && \
    chmod -R 755 /app

# Create a Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

ENTRYPOINT ["./run_benchmark.sh"]

CMD []

# Labels for documentation
LABEL maintainer="llm4codesec-benchmark"
LABEL description="LLM4CodeSec Benchmark with NVIDIA CUDA support for vulnerability detection"
LABEL version="0.1.0"
LABEL cuda.version="12.8.1"
LABEL python.version="3.12"
