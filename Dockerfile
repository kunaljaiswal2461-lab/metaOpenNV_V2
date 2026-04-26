# Root image used by the JUDGE-facing API Space (Kj2461/metaOpenNV_V2).
# For the GPU TRAINING Space, see Dockerfile.train (different base + CMD).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV WINDOW_SIZE=20

EXPOSE 7860

CMD ["python", "app.py"]
