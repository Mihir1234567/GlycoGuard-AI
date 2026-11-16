# Production Dockerfile for GlycoGuard AI
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run training to ensure serialized models and metadata exist
RUN python -m src.train

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: run FastAPI backend and static web dashboard
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
