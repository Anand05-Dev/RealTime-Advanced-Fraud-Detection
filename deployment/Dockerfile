FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY deployment/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY deployment/ /app/
COPY models/ /app/models/

# Environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE $PORT

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
