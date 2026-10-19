import logging
import time
import uuid

import chromadb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from legalrag.embeddings import MODEL_NAME, count_articles, search
from legalrag.llm import LLMError, generate_answer
from legalrag.logging_config import configure_logging, request_id_var

configure_logging()
logger = logging.getLogger(__name__)

INDEX_PATH = "data/chroma_index"
COLLECTION_NAME = "civil_code_articles"

app = FastAPI(title="Arabic Legal RAG")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        },
    )
    return response

_client = chromadb.PersistentClient(path=INDEX_PATH)
_collection = _client.get_collection(COLLECTION_NAME)

# Fail loudly, not silently: the index must have been built with the SAME
# embedding model this process will use for queries. See the embedding-drift
# incident from Session 4.
_stored_model = _collection.metadata.get("embedding_model")
if _stored_model != MODEL_NAME:
    raise RuntimeError(
        f"Index was built with '{_stored_model}' but this process uses "
        f"'{MODEL_NAME}'. Rebuild the index: python scripts/build_index.py"
    )


_article_count = count_articles(_collection)


class AskRequest(BaseModel):
    # Strip whitespace first, then require at least 1 character,
    # so "" and "   " both return 422.
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]  # article citations, e.g. "Egyptian Civil Code, Article 147"


@app.get("/health")
def health():
    return {"status": "healthy", "documents_indexed": _article_count}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    t0 = time.perf_counter()
    results = search(_collection, request.question, k=3)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    sources = [r["metadata"]["citation"] for r in results]
    t1 = time.perf_counter()
    try:
        answer = generate_answer(request.question, results)
    except LLMError as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Answer generation is temporarily unavailable.",
        ) from exc
    logger.info(
        "ask_completed",
        extra={
            "question_chars": len(request.question),
            "retrieved_articles": [r["metadata"]["article_number"] for r in results],
            "retrieval_ms": retrieval_ms,
            "llm_ms": round((time.perf_counter() - t1) * 1000, 1),
        },
    )
    return AskResponse(answer=answer, sources=sources)
