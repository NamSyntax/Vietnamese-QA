# Base img
FROM python:3.10-slim

# Workdir
WORKDIR /app

# Env: no pyc, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Sys deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy reqs
COPY requirements.txt .

# Install py deps
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn pyyaml pydantic

# Copy app data
COPY configs/ configs/
COPY src/ src/
COPY main.py .

# Copy models (optional backup using wildcards to prevent failing if folder missing)
COPY models* ./models/

# API port
EXPOSE 8000

# Config var
ENV CONFIG_PATH=configs/config.yaml

# Start API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
