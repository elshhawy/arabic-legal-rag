"""Build the vector index from the extracted articles.

Settings come from params.yaml (index.chunk_size, index.overlap), so a
DVC run and an MLflow run can reproduce exactly the same index.

    python scripts/build_index.py

chunk_size = 0 means one chunk per whole article (the baseline). Every
chunk keeps its article number, so citations stay article-based.
"""

import json
from pathlib import Path

import chromadb
import yaml

from legalrag.chunking import chunk_text
from legalrag.embeddings import MODEL_NAME, embed_passages, token_offsets

DATA_PATH = Path("data/processed/civil_code.json")
PARAMS_PATH = Path("params.yaml")
INDEX_PATH = Path("data/chroma_index")
COLLECTION_NAME = "civil_code_articles"
ADD_BATCH_SIZE = 500


def load_index_params():
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))["index"]
    chunk_size = int(params["chunk_size"])
    overlap = int(params["overlap"])
    if chunk_size < 0 or overlap < 0:
        raise ValueError("chunk_size and overlap must be >= 0")
    return chunk_size, overlap


def build_chunks(active, chunk_size, overlap):
    """Return one dict per chunk: unique id, text, and article metadata."""
    chunks = []
    for a in active:
        text = a["text_ar"]
        offsets = token_offsets(text) if chunk_size > 0 else []
        pieces = chunk_text(text, offsets, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{a['article_number']}-{i}",
                    "text": piece,
                    "metadata": {
                        "article_number": a["article_number"],
                        "citation": a["citation"],
                        "book": a["book"],
                        "chapter": a["chapter"],
                        "chunk_index": i,
                        "chunk_count": len(pieces),
                    },
                }
            )
    return chunks


def main():
    chunk_size, overlap = load_index_params()
    articles = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    active = [a for a in articles if not a["is_repealed"]]

    chunks = build_chunks(active, chunk_size, overlap)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"

    client = chromadb.PersistentClient(path=str(INDEX_PATH))

    # Drop any old collection so a rerun always reflects the current data.
    # On the very first run there is nothing to delete, so errors are ignored.
    # TODO: catch Chroma's specific "not found" exception instead of Exception.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001, S110
        pass

    # The settings used to build this index are stored with it: anything
    # reading the index later can check the model matches (Session 4 incident)
    # and can see which chunking produced it.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": MODEL_NAME,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    )

    texts = [c["text"] for c in chunks]
    vectors = embed_passages(texts)

    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end].tolist(),
            documents=texts[start:end],
            metadatas=[c["metadata"] for c in chunks[start:end]],
        )

    print(
        f"Indexed {len(active)} articles as {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap}) into {INDEX_PATH}"
    )


if __name__ == "__main__":
    main()