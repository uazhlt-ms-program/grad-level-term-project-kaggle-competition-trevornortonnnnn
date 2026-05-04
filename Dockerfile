FROM python:3.13-slim

LABEL author="Trevor Norton"
LABEL description="Reproducible Python 3.13 container for the classification project."

# Keep Python behavior predictable inside Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

# Project root inside the container.
WORKDIR /app

# System packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

# Copy the project files after dependencies are installed.
COPY . .

# Default command: verify the environment.
CMD ["python", "scripts/check_environment.py"]