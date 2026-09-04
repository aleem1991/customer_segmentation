# Use official slim Python runtime as base image
FROM python:3.12-slim

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the container workspace directory
WORKDIR /app

# Install compilation tools needed for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache layers
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy unified YAML configuration and source directory
COPY config.yaml .
COPY src/ src/

# Copy processed datasets and serialized models needed for server inference
COPY data/processed/ data/processed/
RUN mkdir -p models
COPY models/ models/

# Expose port (7860 is default for Hugging Face Spaces, 8000 for local)
EXPOSE 7860
EXPOSE 8000

# Run FastAPI prediction service via Uvicorn with port override fallback
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
