FROM python:3.11-slim

# System deps for XGBoost / numpy / SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# ── Non-root user (security best practice) ──────────────────────────────────────
# Note: if using Railway Volume at /data, also set env var RAILWAY_RUN_UID=0
#       in Railway dashboard → Variables to avoid volume permission issues.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# PORT is injected by Railway / Render at runtime
ENV PORT=8000

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
