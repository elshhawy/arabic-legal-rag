"""CI quality gate: fail if RAGAS faithfulness on the eval set is below 0.75.

    python scripts/faithfulness_gate.py

Runs the real pipeline (data/chroma_index + Gemini) on data/eval/test_set.jsonl,
scores answered questions with the faithfulness judge and exits 1 when:
  - the mean faithfulness of answered questions is below THRESHOLD, or
  - no question could be scored, or any question failed to score.
Refusals are reported (refusal_rate) but never counted as faithful answers.
Costs about 3 Gemini calls per question and respects the 15 requests/minute
free-tier limit (about 5 minutes for 20 questions).
"""

import json
import sys
from pathlib import Path

import chromadb
from faithfulness_scoring import score_faithfulness

INDEX_PATH = "data/chroma_index"
COLLECTION_NAME = "civil_code_articles"
EVAL_SET = Path("data/eval/test_set.jsonl")
K = 3
THRESHOLD = 0.75


def verdict(metrics):
    """Return (passed, reason)."""
    if metrics["faithfulness_n_failed"]:
        return False, f"{metrics['faithfulness_n_failed']} question(s) failed to score"
    if "faithfulness" not in metrics:
        return False, "no answered question could be scored"
    if metrics["faithfulness"] < THRESHOLD:
        return False, f"faithfulness {metrics['faithfulness']:.3f} < {THRESHOLD}"
    return True, f"faithfulness {metrics['faithfulness']:.3f} >= {THRESHOLD}"


def main():
    lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    client = chromadb.PersistentClient(path=INDEX_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    metrics, _ = score_faithfulness(collection, rows, K)
    print(json.dumps(metrics, indent=2))
    passed, reason = verdict(metrics)
    print(("PASS: " if passed else "FAIL: ") + reason)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
