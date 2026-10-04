# Stage 1: install everything the project needs
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY src/ ./src/

# Force the CPU-only build of torch: this project has no GPU, and the
# default build pulls ~1.5GB of unused NVIDIA CUDA libraries.
RUN pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: the actual runtime image
FROM python:3.12-slim AS runtime
WORKDIR /app

# Keep the embedding model cache inside the image (not a user home dir),
# so it survives switching to a non-root user below.
ENV HF_HOME=/app/.cache/huggingface

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# The vector store and the extracted law live inside the image: no PDF,
# no re-running extract.py or build_index.py when the container starts.
COPY data/processed/civil_code.json ./data/processed/civil_code.json
COPY data/chroma_index ./data/chroma_index

# Download the embedding model once, at BUILD time, so every container
# start is reproducible and needs no internet access.
RUN python -c "from legalrag.embeddings import embed_query; embed_query('warmup')"

RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "legalrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
