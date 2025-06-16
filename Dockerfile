# AI Chatbot Web GUI - Production Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user
RUN groupadd -r chatbot && useradd -r -g chatbot chatbot

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn eventlet

# Copy application code
COPY . .

# Remove unnecessary files
RUN rm -rf \
    .git \
    .github \
    chatbot_env \
    venv \
    env \
    tests \
    *.md \
    .gitignore \
    .env.example \
    demo.py \
    test_*.py \
    setup_github.sh \
    start.sh \
    start.bat

# Create necessary directories
RUN mkdir -p logs static/uploads && \
    chown -R chatbot:chatbot /app

# Switch to non-root user
USER chatbot

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "--timeout", "60", "--keepalive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "app:app"]

# Multi-stage build for development
FROM base as development

# Switch back to root for development dependencies
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy pylint bandit safety

# Install Node.js for frontend tooling
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Switch back to chatbot user
USER chatbot

# Development command
CMD ["python", "app.py"]