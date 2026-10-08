from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FAKE_RESULTS = [
    {
        "metadata": {
            "article_number": 147,
            "citation": "Egyptian Civil Code, Article 147",
        },
        "text": "fake article text",
        "distance": 0.1,
    }
]


@pytest.fixture(scope="module")
def api_module():
    # api.py opens the vector index when it is imported, so skip if it is absent.
    if not Path("data/chroma_index").exists():
        pytest.skip("data/chroma_index not found (build it with scripts/build_index.py)")
    from legalrag import api

    return api


@pytest.fixture
def client(api_module, monkeypatch):
    # No embedding model and no Gemini call in these tests.
    monkeypatch.setattr(api_module, "search", lambda *a, **k: FAKE_RESULTS)
    monkeypatch.setattr(api_module, "generate_answer", lambda q, r: "fake answer")
    return TestClient(api_module.app)


def test_health_ok_and_has_request_id(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(response.headers["X-Request-ID"]) == 12


def test_ask_returns_expected_schema(client):
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["sources"] == [
        {
            "article_number": 147,
            "citation": "Egyptian Civil Code, Article 147",
            "text": "fake article text",
        }
    ]


def test_ask_missing_question_is_422(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_wrong_type_is_422(client):
    assert client.post("/ask", json={"question": 123}).status_code == 422


def test_ask_llm_failure_is_503_without_leaking_details(api_module, client, monkeypatch):
    def broken(question, results):
        raise api_module.LLMError("secret internal detail")

    monkeypatch.setattr(api_module, "generate_answer", broken)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "secret" not in response.text
