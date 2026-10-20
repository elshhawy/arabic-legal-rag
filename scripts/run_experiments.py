"""Run the chunking experiments and log each one to MLflow.

Each run builds a throwaway in-memory index (the DVC-tracked data/chroma_index
is never touched), scores retrieval on the eval set, and logs params + metrics.
No LLM is called here: faithfulness is added in a separate step.

    python scripts/run_experiments.py
"""

import hashlib
import json
import os
import time
from pathlib import Path

import chromadb
import mlflow
from build_index import ADD_BATCH_SIZE, build_chunks

from legalrag.embeddings import MODEL_NAME, embed_passages, search

CORPUS = Path("data/processed/civil_code.json")
EVAL_SET = Path("data/eval/test_set.jsonl")
EXPERIMENT_NAME = "arabic-legal-rag-chunking"
K = 3

# (run name, chunk_size in e5 tokens, overlap). chunk_size 0 = whole article.
CONFIGS = [
    ("baseline-whole-article", 0, 0),
    ("tokens-256-overlap-32", 256, 32),
    ("tokens-128-overlap-32", 128, 32),
    ("tokens-64-overlap-16", 64, 16),
    ("tokens-32-overlap-8", 32, 8),
]


def md5_of(path):
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def build_memory_index(chunks, name):
    collection = chromadb.EphemeralClient().create_collection(name=name)
    texts = [c["text"] for c in chunks]
    vectors = embed_passages(texts)
    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=[c["id"] for c in chunks[start:end]],
            embeddings=vectors[start:end].tolist(),
            documents=texts[start:end],
            metadatas=[c["metadata"] for c in chunks[start:end]],
        )
    return collection


def evaluate(collection, rows):
    stats = {}
    details = []
    for row in rows:
        results = search(collection, row["question"], k=K)
        numbers = [r["metadata"]["article_number"] for r in results]
        expected = row["article_number"]
        hit1 = numbers[:1] == [expected]
        hitk = expected in numbers
        details.append(
            {"id": row["id"], "lang": row["lang"], "expected": expected, "got": numbers}
        )
        for key in ("all", row["lang"]):
            s = stats.setdefault(key, {"n": 0, "hit1": 0, "hitk": 0})
            s["n"] += 1
            s["hit1"] += hit1
            s["hitk"] += hitk
    metrics = {}
    for key, s in stats.items():
        metrics[f"hit_at_1_{key}"] = s["hit1"] / s["n"]
        metrics[f"hit_at_{K}_{key}"] = s["hitk"] / s["n"]
    return metrics, details


def main():
    articles = json.loads(CORPUS.read_text(encoding="utf-8"))
    active = [a for a in articles if not a["is_repealed"]]
    lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    common = {
        "embedding_model": MODEL_NAME,
        "k": K,
        "n_questions": len(rows),
        "corpus_md5": md5_of(CORPUS),
        "eval_set_md5": md5_of(EVAL_SET),
    }

    for run_name, chunk_size, overlap in CONFIGS:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"chunk_size": chunk_size, "overlap": overlap, **common})
            t0 = time.perf_counter()
            chunks = build_chunks(active, chunk_size, overlap)
            collection = build_memory_index(chunks, run_name)
            build_seconds = round(time.perf_counter() - t0, 1)
            metrics, details = evaluate(collection, rows)
            metrics["n_chunks"] = len(chunks)
            metrics["index_build_seconds"] = build_seconds
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"questions": details}, "retrieval_details.json")
            print(
                f"{run_name}: chunks={len(chunks)} build={build_seconds}s "
                f"hit@1 all={metrics['hit_at_1_all']:.2f} "
                f"hit@{K} all={metrics[f'hit_at_{K}_all']:.2f} "
                f"ar={metrics[f'hit_at_{K}_ar']:.2f} en={metrics[f'hit_at_{K}_en']:.2f}"
            )


if __name__ == "__main__":
    main()
