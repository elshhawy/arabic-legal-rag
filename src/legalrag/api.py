import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

from legalrag.embeddings import MODEL_NAME, search

INDEX_PATH = "data/chroma_index"
COLLECTION_NAME = "civil_code_articles"

app = FastAPI(title="Arabic Legal RAG")

_client = chromadb.PersistentClient(path=INDEX_PATH)
_collection = _client.get_collection(COLLECTION_NAME)

# Fail loudly, not silently: the index must have been built with the SAME
# embedding model this process will use for queries. See the embedding-drift
# incident from Session 4.
_stored_model = _collection.metadata.get("embedding_model")
if _stored_model != MODEL_NAME:
    raise RuntimeError(
        f"Index was built with '{_stored_model}' but this process uses "
        f"'{MODEL_NAME}'. Rebuild the index: python build_index.py"
    )


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    article_number: int
    citation: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = search(_collection, request.question, k=3)
    sources = [
        Source(
            article_number=r["metadata"]["article_number"],
            citation=r["metadata"]["citation"],
            text=r["text"],
        )
        for r in results
    ]
    # Placeholder answer for now -- the LLM step comes next.
    answer = f"Found {len(sources)} relevant articles."
    return AskResponse(answer=answer, sources=sources)