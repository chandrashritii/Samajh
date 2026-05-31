FROM python:3.11-slim

# ffmpeg for STT/TTS audio (yt-dlp audio extraction + Saaras chunking)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST from PyTorch's CPU index. The default torch wheel
# on Linux is the CUDA build (~2 GB) — useless on a CPU host, and big enough to
# blow the free-tier build/memory. Pin the CPU build, then the rest resolves
# against it instead of re-pulling CUDA.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Warm the embedding model at build so the first request isn't slow (matches the
# full model id the app loads, so build-warm and runtime hit the same cache key).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Keep the ML stack lean on a 512 MB free instance.
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=8000

EXPOSE 8000

# shell form so $PORT (set by the host) expands
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
