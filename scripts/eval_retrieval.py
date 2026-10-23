"""Retrieval-only evaluation: does the source article come back in the top k?

No LLM is called. Run: python scripts/eval_retrieval.py
"""

import json
from pathlib import Path

import chromadb

from legalrag.embeddings import search

TEST_SET = Path("data/eval/test_set.jsonl")
INDEX_PATH = "data/chroma_index"
COLLECTION_NAME = "civil_code_articles"
K = 3


def main():
    lines = TEST_SET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    collection = chromadb.PersistentClient(path=INDEX_PATH).get_collection(
        COLLECTION_NAME
    )
    stats = {}
    for row in rows:
        results = search(collection, row["question"], k=K)
        numbers = [r["metadata"]["article_number"] for r in results]
        hit_at_1 = numbers[:1] == [row["article_number"]]
        hit_at_k = row["article_number"] in numbers
        for key in ("all", row["lang"]):
            s = stats.setdefault(key, {"n": 0, "hit1": 0, "hitk": 0})
            s["n"] += 1
            s["hit1"] += hit_at_1
            s["hitk"] += hit_at_k
        if not hit_at_k:
            print("MISS", row["id"], row["lang"], "expected", row["article_number"], "got", numbers)
    for key, s in stats.items():
        n = s["n"]
        print(f"{key}: n={n} hit@1={s['hit1'] / n:.2f} hit@{K}={s['hitk'] / n:.2f}")


if __name__ == "__main__":
    main()
