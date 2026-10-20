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
    assert collection.last_call[1] == 1 * embeddings.OVERFETCH  # k articles, more chunks


def test_prompt_keeps_article_order_and_labels():
    results = [
        {"metadata": {"article_number": 5}, "text": "FIVE"},
        {"metadata": {"article_number": 9}, "text": "NINE"},
    ]
    prompt = build_prompt("Q?", results)
    assert prompt.index("[مادة 5]") < prompt.index("[مادة 9]")
    assert prompt.endswith("السؤال: Q?")


class ChunkedFakeCollection:
    """Three chunks, best first: two belong to article 10, one to article 20."""

    def query(self, query_embeddings, n_results):
        metas = [
            {"article_number": 10, "citation": "C10", "chunk_index": 1},
            {"article_number": 20, "citation": "C20", "chunk_index": 0},
            {"article_number": 10, "citation": "C10", "chunk_index": 0},
        ]
        return {
            "metadatas": [metas],
            "documents": [["TEN-B", "TWENTY", "TEN-A"]],
            "distances": [[0.1, 0.2, 0.3]],
        }


def test_search_merges_chunks_of_the_same_article(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", lambda q: np.array([0.1, 0.2]))
    results = embeddings.search(ChunkedFakeCollection(), "q", k=2)
    assert [r["metadata"]["article_number"] for r in results] == [10, 20]
    assert results[0]["text"] == "TEN-A\nTEN-B"  # reading order, not rank order
    assert results[0]["distance"] == 0.1  # best chunk distance


def test_search_returns_at_most_k_articles(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", lambda q: np.array([0.1, 0.2]))
    results = embeddings.search(ChunkedFakeCollection(), "q", k=1)
    assert [r["metadata"]["article_number"] for r in results] == [10]


class CountingFakeCollection:
    def get(self, include):
        metas = [{"article_number": 10}, {"article_number": 10}, {"article_number": 20}]
        return {"metadatas": metas}


def test_count_articles_counts_articles_not_chunks():
    assert embeddings.count_articles(CountingFakeCollection()) == 2
