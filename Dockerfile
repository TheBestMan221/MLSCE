# ─────────────────────────────────────────────────────
# Dockerfile  –  F1 ML Prediction System (Streamlit)
# Build : docker build -t f1-ml-app .
# Run   : docker run -p 8501:8501 f1-ml-app
# ─────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="f1-ml-project"
LABEL description="Formula 1 Race Prediction System - Streamlit App"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY src/ ./src/
COPY ml_pipeline/ ./ml_pipeline/
COPY mlops/ ./mlops/
COPY params.yaml .

# Copy pre-trained models and processed data (must exist on host before building)
# If not present, the app starts but shows "train models first" warning
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Expose Streamlit port
EXPOSE 8501

# Streamlit config: disable browser auto-open, bind to 0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
