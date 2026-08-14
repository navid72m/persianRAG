# Persian RAG — serve (Streamlit UI) or ingest inside Docker.
#
# Build:
#   docker build -t persian-rag .
#
# Serve (query API / Streamlit UI):
#   docker run -d -p 8501:8501 \
#     -e OPENCODE_API_KEY=... -e JINA_API_KEY=... \
#     -e QDRANT_URL=... -e QDRANT_API_KEY=... \
#     -v "$PWD":/app/data \
#     -e PARENT_DB_PATH=/app/data/parents.sqlite \
#     -e SPARSE_VOCAB_PATH=/app/data/sparse_vocab.json \
#     persian-rag
#
# Ingest (must have the PDF mounted; OCR requires the text layer or Tesseract):
#   docker run --rm \
#     -e JINA_API_KEY=... -e QDRANT_URL=... -e QDRANT_API_KEY=... \
#     -v "$PWD":/app/data \
#     -e PARENT_DB_PATH=/app/data/parents.sqlite \
#     -e SPARSE_VOCAB_PATH=/app/data/sparse_vocab.json \
#     persian-rag python -m persian_rag.ingest /app/data/document.pdf
#
# GPU (CUDA) option: replace the base image + torch install for CUDA wheels
# and run with `--gpus all`. Default image is CPU-only.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_BACKEND=local \
    EMBED_DEVICE=cpu \
    HF_HUB_DISABLE_TELEMETRY=1

# Tesseract for the OCR fallback (scanned PDFs).
# Some networks block deb.debian.org — pass --build-arg APT_MIRROR=... to swap
# (e.g. http://ftp.debian.org/debian). The security suite is skipped: most
# mirrors don't carry it and tesseract + deps all live in main.
ARG APT_MIRROR=http://deb.debian.org/debian
RUN CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") && \
    KEYRING=$(ls /usr/share/keyrings/debian-archive-keyring* | head -1) && \
    printf 'Types: deb\nURIs: %s\nSuites: %s %s-updates\nComponents: main\nSigned-By: %s\n' \
        "$APT_MIRROR" "$CODENAME" "$CODENAME" "$KEYRING" \
        > /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fas \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default PyPI aarch64 wheel drags in hundreds of MB
# of CUDA libs we don't need. (For GPU: swap TORCH_INDEX for cuXXX and run
# with --gpus all.)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --timeout 300 --retries 10 --no-cache-dir --index-url ${TORCH_INDEX} torch

COPY requirements.txt .
RUN pip install --timeout 300 --retries 10 --no-cache-dir -r requirements.txt

# Pre-download the embedding model so first startup doesn't stall.
# Network to huggingface.co may need a mirror: --build-arg HF_ENDPOINT=https://hf-mirror.com
ARG HF_ENDPOINT=
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)" \
    || echo "model pre-download failed — will download on first run"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
