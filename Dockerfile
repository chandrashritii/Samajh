FROM python:3.11-slim

# ffmpeg for STT/TTS audio (yt-dlp audio extraction + Saaras chunking)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Warm the embedding model at build so the first request isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PORT=8000
EXPOSE 8000

# shell form so $PORT (set by the host) expands
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
