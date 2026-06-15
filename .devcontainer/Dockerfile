# .devcontainer/Dockerfile

# Build stage - install packages in an isolated layer
FROM python:3.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.20 /uv /uvx /bin/

# Set UV environment for build
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Explicit working directory for the build
WORKDIR /build

# Copy project files and install packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.13-slim-trixie

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv into runtime to add packages at whim
COPY --from=ghcr.io/astral-sh/uv:0.8.20 /uv /uvx /bin/

# Create user and setup workspace
RUN useradd --create-home --shell /bin/bash vscode
USER vscode
WORKDIR /home/vscode

# Copy virtual environment from builder
COPY --from=builder --chown=vscode:vscode /build/.venv .venv

# Put the venv on PATH and tell uv where it lives
ENV PATH="/home/vscode/.venv/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT="/home/vscode/.venv"

# Setup Jupyter kernel
RUN python -m ipykernel install --user --name scryptiq-VE --display-name "scryptiq-VE"

# Activate venv in interactive shells
RUN echo 'source /home/vscode/.venv/bin/activate' >> ~/.bashrc