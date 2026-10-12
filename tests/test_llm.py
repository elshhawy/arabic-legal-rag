from types import SimpleNamespace

import pytest

from legalrag import llm

RESULTS = [
    {"metadata": {"article_number": 147}, "text": "TEXT-OF-147", "distance": 0.1},
    {"metadata": {"article_number": 157}, "text": "TEXT-OF-157", "distance": 0.2},
]


def make_fake_client(reply_text=None, error=None, captured=None):
    """Build a stand-in for genai.Client that never touches the network."""

    class FakeModels:
        def generate_content(self, **kwargs):
            if captured is not None:
                captured.update(kwargs)
            if error is not None:
                raise error
            return SimpleNamespace(text=reply_text)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    return FakeClient


def test_build_prompt_contains_articles_and_question():
    prompt = llm.build_prompt("QUESTION-HERE", RESULTS)
    assert "QUESTION-HERE" in prompt
    assert "TEXT-OF-147" in prompt and "TEXT-OF-157" in prompt
    assert "147" in prompt and "157" in prompt


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.generate_answer("q", RESULTS)


def test_missing_model_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_MODEL"):
        llm.generate_answer("q", RESULTS)


def test_success_returns_stripped_text_and_sends_context(monkeypatch):
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    monkeypatch.setattr(
        llm.genai, "Client", make_fake_client(reply_text="  answer  ", captured=captured)
    )
    assert llm.generate_answer("q", RESULTS) == "answer"
    assert captured["model"] == "some-model"
    assert "TEXT-OF-147" in captured["contents"]


def test_empty_reply_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    monkeypatch.setattr(llm.genai, "Client", make_fake_client(reply_text=""))
    with pytest.raises(llm.LLMError, match="empty"):
        llm.generate_answer("q", RESULTS)


def test_api_failure_is_wrapped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    monkeypatch.setattr(
        llm.genai, "Client", make_fake_client(error=RuntimeError("boom"))
    )
    with pytest.raises(llm.LLMError, match="boom"):
        llm.generate_answer("q", RESULTS)
