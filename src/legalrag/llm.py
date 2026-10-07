"""Generate a grounded Arabic answer from retrieved Civil Code articles.

The LLM is only allowed to use the articles we hand it. If they do not
contain the answer, it must say so instead of inventing law.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# The model name comes from .env (GEMINI_MODEL) so we can switch models
# without touching code. No silent default: Google retires model names.
TIMEOUT_MS = 60_000  # fail after 60s instead of hanging forever

SYSTEM_INSTRUCTION = """أنت مساعد قانوني متخصص في القانون المدني المصري.
أجب عن السؤال بالاعتماد فقط على المواد المقدمة في السياق، ولا تستخدم أي معلومة من خارجها.
اذكر رقم كل مادة تستند إليها بصيغة (مادة N).
إذا كانت المواد المقدمة لا تكفي للإجابة، قل صراحة: "لا توجد في المواد المسترجعة معلومات كافية للإجابة عن هذا السؤال"، ولا تحاول التخمين.
اكتب الإجابة بالعربية الفصحى المبسطة وبإيجاز."""


class LLMError(RuntimeError):
    """Raised when answer generation fails (missing key, API error, empty reply)."""


def build_prompt(question, results):
    """Put the retrieved articles and the question into one prompt."""
    context = "\n\n".join(
        f"[مادة {r['metadata']['article_number']}]\n{r['text']}" for r in results
    )
    return f"السياق (مواد القانون المدني):\n{context}\n\nالسؤال: {question}"


def generate_answer(question, results):
    """Ask Gemini to answer using only the retrieved articles."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set (check your .env file).")

    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise LLMError("GEMINI_MODEL is not set (check your .env file).")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=TIMEOUT_MS),
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=build_prompt(question, results),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
            ),
        )
    except Exception as exc:
        raise LLMError(f"Gemini call failed: {exc}") from exc

    if not response.text:
        raise LLMError("Gemini returned an empty response.")
    return response.text.strip()
