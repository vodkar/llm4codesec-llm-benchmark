# Multi-stage Dockerfile for LLM4CodeSec Benchmark with NVIDIA CUDA support
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set environment variables for non-interactive builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-poetry \
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


# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root && \
    poetry env activate && \
    pip install flash-attn==2.7.4.post1 --no-build-isolation \
    poetry cache clear PyPI --all

WORKDIR /app

COPY src/ .

# Create a Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

ENTRYPOINT ["poetry", "run"]

CMD []

# Labels for documentation
LABEL maintainer="llm4codesec-benchmark"
LABEL description="LLM4CodeSec Benchmark with NVIDIA CUDA support for vulnerability detection"
LABEL version="0.1.0"
LABEL cuda.version="12.8.1"
LABEL python.version="3.12"
