import numpy as np

from legalrag import embeddings
from legalrag.llm import build_prompt


class FakeCollection:
    """Mimics chromadb's collection.query() without any real index."""

    def __init__(self):
        self.last_call = None

    def query(self, query_embeddings, n_results):
        # Remember how we were called so the test can inspect it.
        self.last_call = (query_embeddings, n_results)
        return {
            "metadatas": [[{"article_number": 147, "citation": "Egyptian Civil Code, Article 147"}]],
            "documents": [["TEXT-147"]],
            "distances": [[0.25]],
        }


def test_search_reshapes_chroma_output(monkeypatch):
    # Replace the real embedding model with a tiny fake vector.
    monkeypatch.setattr(embeddings, "embed_query", lambda q: np.array([0.1, 0.2]))
    collection = FakeCollection()
    results = embeddings.search(collection, "any question", k=1)
    assert results == [
        {
            "metadata": {"article_number": 147, "citation": "Egyptian Civil Code, Article 147"},
            "text": "TEXT-147",
            "distance": 0.25,
        }
    ]
    assert collection.last_call[1] == 1  # k is forwarded as n_results


def test_prompt_keeps_article_order_and_labels():
    results = [
        {"metadata": {"article_number": 5}, "text": "FIVE"},
        {"metadata": {"article_number": 9}, "text": "NINE"},
    ]
    prompt = build_prompt("Q?", results)
    assert prompt.index("[مادة 5]") < prompt.index("[مادة 9]")
    assert prompt.endswith("السؤال: Q?")