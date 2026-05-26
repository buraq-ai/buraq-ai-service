# ============================================================
# Stage 1: Build dependencies
# ============================================================
FROM python:3.11 AS builder

WORKDIR /app

# Create a virtual environment in a neutral location
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install PyTorch CPU-only first (much smaller, no CUDA)
RUN pip install --no-cache-dir --default-timeout=120 torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN addgroup --system buraqgroup && \
    adduser --system --group buraquser

# Copy the entire virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code and set ownership of the entire app directory
COPY --chown=buraquser:buraqgroup . .
RUN mkdir -p /app/.cache && chown -R buraquser:buraqgroup /app

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Hugging Face cache directory (writable by non-root user)
ENV HF_HOME=/app/.cache/huggingface

# Switch to non-root user
USER buraquser

# Expose the port matching main.py (8001)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Start the application
CMD ["uvicorn", "main:app", \
    "--host", "0.0.0.0", \
    "--port", "8001"]