"""Generate a small evaluation set: one question per sampled Civil Code article.

Each row keeps the source article number, so retrieval can be scored (does the
right article come back?). Questions are written by Gemini from the article
text, so they are probably easier than real user questions.

    python scripts/make_eval_set.py
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

DATA_PATH = Path("data/processed/civil_code.json")
OUT_PATH = Path("data/eval/test_set.jsonl")
N_QUESTIONS = 20
SEED = 7
MIN_CHARS, MAX_CHARS = 200, 700
SLEEP_SECONDS = 5  # free tier allows 15 requests/minute; 5 s keeps us at 12
RETRY_WAIT_SECONDS = 45

PROMPTS = {
    "ar": """اقرأ المادة التالية من القانون المدني المصري، واكتب سؤالًا واحدًا بالعربية
تجيب عنه هذه المادة وحدها، كما يسأله شخص عادي لا يعرف رقم المادة.
لا تذكر رقم المادة في السؤال. اكتب السؤال فقط دون أي شرح.

المادة:
{text}""",
    "en": """Read this article of the Egyptian Civil Code (Arabic text). Write ONE
question in English that this article alone answers, as an ordinary person who
does not know the article number would ask it. Do not mention the article
number. Output only the question.

Article:
{text}""",
}


def generate_with_retry(client, model, prompt):
    """Call Gemini; on a 429 (rate limit) wait and retry a few times."""
    for attempt in range(5):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == 4:
                raise
            print(f"rate limited, waiting {RETRY_WAIT_SECONDS}s ...")
            time.sleep(RETRY_WAIT_SECONDS)


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env")

    articles = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    pool = [
        a
        for a in articles
        if not a["is_repealed"] and MIN_CHARS <= len(a["text_ar"]) <= MAX_CHARS
    ]
    sample = random.Random(SEED).sample(pool, N_QUESTIONS)

    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=60_000)
    )
    rows = []
    for i, article in enumerate(sample):
        lang = "ar" if i % 2 == 0 else "en"
        response = generate_with_retry(
            client, model, PROMPTS[lang].format(text=article["text_ar"])
        )
        question = (response.text or "").strip().strip('"')
        if not question:
            raise SystemExit(f"Empty question for article {article['article_number']}")
        rows.append(
            {
                "id": i,
                "lang": lang,
                "question": question,
                "article_number": article["article_number"],
                "citation": article["citation"],
                "source_text": article["text_ar"],
            }
        )
        print(f"{i + 1}/{N_QUESTIONS} done")
        time.sleep(SLEEP_SECONDS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to {OUT_PATH} (pool size {len(pool)})")


if __name__ == "__main__":
    main()