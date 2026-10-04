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


def search(collection, question, k=3):
    """Find the k articles closest in meaning to the question."""
    vector = embed_query(question)
    results = collection.query(query_embeddings=[vector.tolist()], n_results=k)
    return [
        {"metadata": meta, "text": doc, "distance": dist}
        for meta, doc, dist in zip(
            results["metadatas"][0], results["documents"][0], results["distances"][0]
        )
    ]