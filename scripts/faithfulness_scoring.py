"""Score RAGAS faithfulness for one chunking configuration.

Two fixes that come from debug_faithfulness.json (the judge returned 0.0 twice):

1. Citation labels. The answer model sees "[<article word> N]" before every
   article and cites it as "(<article word> N)". The judge used to get the bare
   article text, so it marked a correct, cited answer unfaithful ("the context
   does not mention Article 728"). The judge now gets the same labelled contexts.
2. Refusals. "Not enough information" is not a claim about the law, yet it was
   judged against the retrieved text and scored 0. A refusal is counted in
   refusal_rate and left out of the faithfulness mean.

Answers and scores are cached on disk (key = models + question + contexts), so a
429 or a crash can be resumed without paying for the same Gemini calls twice.

Env: GEMINI_API_KEY and GEMINI_MODEL (required), JUDGE_MODEL (optional, defaults
to GEMINI_MODEL), FAITHFULNESS_QUESTIONS (optional, score only the first N rows).
"""

import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from legalrag.embeddings import search
from legalrag.llm import generate_answer

OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CACHE_PATH = Path(".cache/faithfulness_cache.json")

# Arabic text is written as escapes on purpose (it must never pass through a shell).
# REFUSAL_MARKER is the start of the refusal sentence that llm.SYSTEM_INSTRUCTION
# tells the model to use; ARTICLE_WORD is the word used in "[<word> N]".
REFUSAL_MARKER = (
    "\u0644\u0627 \u062a\u0648\u062c\u062f \u0641\u064a "
    "\u0627\u0644\u0645\u0648\u0627\u062f "
    "\u0627\u0644\u0645\u0633\u062a\u0631\u062c\u0639\u0629 "
    "\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0643\u0627\u0641\u064a\u0629"
)
ARTICLE_WORD = "\u0645\u0627\u062f\u0629"

# Free tier: 15 requests/minute (measured: a 429 at the 16th call).
# One question = 1 answer call + 2 judge calls = 3 calls, so >= 12 s per question.
SECONDS_PER_QUESTION = 13.0
RETRY_WAITS = (30, 60, 120)


def labelled_contexts(results):
    """Same '[<article word> N]' labels the answer model saw in llm.build_prompt."""
    return [
        f"[{ARTICLE_WORD} {r['metadata']['article_number']}]\n{r['text']}"
        for r in results
    ]


def is_refusal(answer):
    return REFUSAL_MARKER in answer


def cache_key(generation_model, judge_model, question, contexts):
    raw = json.dumps(
        [generation_model, judge_model, question, contexts], ensure_ascii=False
    )
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def summarize(details):
    """Mean faithfulness over scored (answered) questions + counts."""
    scored = [d["faithfulness"] for d in details if d.get("faithfulness") is not None]
    metrics = {
        "faithfulness_n_questions": len(details),
        "faithfulness_n_scored": len(scored),
        "faithfulness_n_failed": sum(1 for d in details if d.get("error")),
        "refusal_rate": (
            sum(1 for d in details if d.get("refused")) / len(details)
            if details
            else 0.0
        ),
    }
    if scored:
        metrics["faithfulness"] = sum(scored) / len(scored)
    return metrics


def model_names():
    generation = os.environ["GEMINI_MODEL"]
    return generation, os.getenv("JUDGE_MODEL") or generation


async def _with_retries(label, make_call):
    for wait in (*RETRY_WAITS, None):
        try:
            return await make_call()
        except Exception as exc:  # noqa: BLE001 - every API error is retried alike
            if wait is None:
                raise
            print(f"  {label}: {type(exc).__name__}, retrying in {wait}s")
            await asyncio.sleep(wait)


async def _score_question(metric, models, row, results, cache, clock):
    generation_model, judge_model = models
    contexts = labelled_contexts(results)
    key = cache_key(generation_model, judge_model, row["question"], contexts)
    entry = cache.get(key, {})
    detail = {
        "id": row["id"],
        "lang": row["lang"],
        "expected": row["article_number"],
        "retrieved": [r["metadata"]["article_number"] for r in results],
    }

    try:
        if "answer" not in entry:
            await _throttle(clock)
            entry["answer"] = await _with_retries(
                "answer",
                lambda: asyncio.to_thread(generate_answer, row["question"], results),
            )
            entry["refused"] = is_refusal(entry["answer"])
            cache[key] = entry
        if not entry["refused"] and "faithfulness" not in entry:
            score = await _with_retries(
                "judge",
                lambda: metric.ascore(
                    user_input=row["question"],
                    response=entry["answer"],
                    retrieved_contexts=contexts,
                ),
            )
            value = float(getattr(score, "value", score))
            if math.isnan(value):
                raise ValueError("judge returned NaN")
            entry["faithfulness"] = value
            cache[key] = entry
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        detail["error"] = f"{type(exc).__name__}: {exc}"

    detail["answer"] = entry.get("answer")
    detail["refused"] = entry.get("refused", False)
    detail["faithfulness"] = entry.get("faithfulness")
    return detail


async def _throttle(clock):
    wait = SECONDS_PER_QUESTION - (time.monotonic() - clock["last"])
    if wait > 0:
        await asyncio.sleep(wait)
    clock["last"] = time.monotonic()


async def _score_all(collection, rows, k, limit):
    load_dotenv()
    models = model_names()
    client = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=OPENAI_COMPAT_URL)
    metric = Faithfulness(llm=llm_factory(models[1], provider="openai", client=client))
    cache = load_cache()
    clock = {"last": 0.0}
    details = []
    for row in rows[:limit]:
        results = search(collection, row["question"], k=k)
        details.append(await _score_question(metric, models, row, results, cache, clock))
        save_cache(cache)
        print(
            f"  q{row['id']}: refused={details[-1]['refused']} "
            f"faithfulness={details[-1]['faithfulness']} "
            f"error={details[-1].get('error')}"
        )
    return summarize(details), details


def score_faithfulness(collection, rows, k):
    """Return (metrics, per-question details). One event loop per configuration."""
    limit = int(os.getenv("FAITHFULNESS_QUESTIONS", "0")) or None
    return asyncio.run(_score_all(collection, rows, k, limit))
