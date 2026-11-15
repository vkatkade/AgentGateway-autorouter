FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY extproc ./extproc
COPY autorouter ./autorouter

# Plaintext gRPC (h2c) on $PORT. Deploy Cloud Run with --use-http2.
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["python", "-m", "autorouter.server"]
