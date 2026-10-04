"""Build the vector index once from the extracted articles.

Run it whenever civil_code.json changes:
    python build_index.py
"""

import json
from pathlib import Path

import chromadb

from legalrag.embeddings import MODEL_NAME, embed_passages

DATA_PATH = Path("data/processed/civil_code.json")
INDEX_PATH = Path("data/chroma_index")
COLLECTION_NAME = "civil_code_articles"


def main():
    articles = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    active = [a for a in articles if not a["is_repealed"]]

    client = chromadb.PersistentClient(path=str(INDEX_PATH))

    # Drop any old collection so a rerun always reflects the current data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # The model name is stored as collection metadata -- this is the fix
    # from the Session 4 incident: anything reading this index later can
    # check it matches before trusting the vectors.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": MODEL_NAME},
    )

    texts = [a["text_ar"] for a in active]
    vectors = embed_passages(texts)

    collection.add(
        ids=[str(a["article_number"]) for a in active],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[
            {
                "article_number": a["article_number"],
                "citation": a["citation"],
                "book": a["book"],
                "chapter": a["chapter"],
            }
            for a in active
        ],
    )

    print(f"Indexed {len(active)} articles into {INDEX_PATH}")


if __name__ == "__main__":
    main()