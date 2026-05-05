FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Install deps in a separate layer so rebuilds are fast
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source — data (144 MB CSV) is mounted at runtime, not baked in
COPY config/ ./config/
COPY dashboard/ ./dashboard/
COPY src/     ./src/
COPY tests/   ./tests/

RUN mkdir -p /var/lib/ml-platform && \
    chown -R appuser:appgroup /app /var/lib/ml-platform
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz')"

CMD ["python", "-m", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--no-access-log"]
