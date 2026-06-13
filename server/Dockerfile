# Gateway + BiCodec vocoder image (CPU). For GPU, use a CUDA base image and
# install the matching torch wheel.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Install the maliba_ai package (vocoder + config) without heavy extras.
COPY maliba_ai /app/maliba_ai
COPY setup.py README.md /app/
RUN pip install --no-cache-dir --no-deps -e .

COPY server /app/server

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
