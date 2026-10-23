"""Turn text into embedding vectors, using one fixed model everywhere.

Why this file exists
---------------------
Retrieval only works if the SAME embedding model, with the SAME settings,
encodes both the articles (once, when building the index) and every user
question (every time someone asks). If these ever drift apart -- a
different model, a missing prefix, a different library version -- search
returns nonsense or nothing, silently.

Session 4's monitoring material describes exactly this failure: an index
rebuilt with a new embedding model while the query path still used the old
one caused 18% of Arabic answers to fail with zero retrieved chunks.

This module is the ONLY place that touches sentence_transformers directly.
Everything else in the project calls embed_passages() or embed_query().
"""

from sentence_transformers import SentenceTransformer

# Pinned once. Changing this invalidates every vector already stored: a
# change here means rebuilding the whole index from scratch, not patching it.
MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384  # this model's fixed output size, used later by the vector store

# e5 models were trained with these exact prefixes attached to every input.
# Skipping them, or mixing them up (e.g. embedding a question with
# "passage: "), quietly lowers retrieval quality without raising any error.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# How many chunks to fetch per article we want back. Fetching more than k
# chunks lets us merge chunks of the same article and still return k articles.
OVERFETCH = 4

_model = None  # loaded once per process, on first use


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts):
    """Embed a list of article texts, for building the index."""
    model = _get_model()
    prefixed = [PASSAGE_PREFIX + text for text in texts]
    return model.encode(prefixed, normalize_embeddings=True)


def embed_query(text):
    """Embed one user question, for searching the index."""
    model = _get_model()
    prefixed = QUERY_PREFIX + text
    return model.encode(prefixed, normalize_embeddings=True)


def merge_chunks_by_article(chunks, k):
    """Merge chunks of the same article; return at most k articles, best first.

    `chunks` must be sorted best-first (as Chroma returns them). Chunks of one
    article are joined in reading order (chunk_index). With overlapping chunks
    the overlapped words appear twice in the joined text.
    """
    by_article = {}
    for chunk in chunks:
        by_article.setdefault(chunk["metadata"]["article_number"], []).append(chunk)

    merged = []
    for group in by_article.values():  # dicts keep insertion (= best-first) order
        group.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
        merged.append(
            {
                "metadata": group[0]["metadata"],
                "text": "\n".join(c["text"] for c in group),
                "distance": min(c["distance"] for c in group),
            }
        )
    return merged[:k]


def search(collection, question, k=3):
    """Find the k articles closest in meaning to the question."""
    vector = embed_query(question)
    results = collection.query(
        query_embeddings=[vector.tolist()], n_results=k * OVERFETCH
    )
    chunks = [
        {"metadata": meta, "text": doc, "distance": dist}
        for meta, doc, dist in zip(
            results["metadatas"][0], results["documents"][0], results["distances"][0]
        )
    ]
    return merge_chunks_by_article(chunks, k)


def count_articles(collection):
    """Number of distinct articles in the index (not the number of chunks)."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return len({m["article_number"] for m in metadatas})


def token_offsets(text):
    """One (start_char, end_char) pair per model token of `text`.

    Special tokens are excluded. Used by chunking to cut the ORIGINAL text on
    token boundaries, so every chunk is a verbatim slice of the article.
    """
    tokenizer = _get_model().tokenizer
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(pair) for pair in encoded["offset_mapping"]]