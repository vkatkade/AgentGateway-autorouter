FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN useradd -r -u 10001 app
USER app
ENV PORT=8080
# One worker keeps a single pooled gRPC channel to the router; threads serve concurrent requests.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 180 app:app
