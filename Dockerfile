# Use an official Python runtime as a parent image
FROM python:3.13.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# (Optional) Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Set environment variables for production
ENV GUNICORN_CMD_ARGS="--workers 3 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --log-level error"

# Start the application with Gunicorn and Uvicorn worker
CMD sh -c "gunicorn main:app $GUNICORN_CMD_ARGS"