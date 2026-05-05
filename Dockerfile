FROM python:3.11-slim

# System deps for XGBoost / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# PORT is injected by Railway / Render at runtime
ENV PORT=8000

EXPOSE $PORT

# Start server — reads $PORT so the platform can route traffic correctly
CMD uvicorn server.app:app --host 0.0.0.0 --port $PORT
