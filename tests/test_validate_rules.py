import pytest

from legalrag import validate
from legalrag.validate import (
    check_arabic_text,
    check_citation_format,
    check_gaps,
    check_length,
    check_numbering,
    check_repealed_flag,
)


def make(number=1, **overrides):
    """Build one valid record; tests override only the field they break."""
    record = {
        "article_number": number, "part": "p", "book": "b", "chapter": "c",
        "section": "s", "topic": "t", "text_ar": "نص عربي صالح للاختبار",
        "text_en": "text", "is_repealed": False, "source_page": 1,
        "citation": f"Egyptian Civil Code, Article {number}",
    }
    record.update(overrides)
    return record


@pytest.fixture
def small_code(monkeypatch):
    # The real LAST_ARTICLE is 1149; shrink it so tiny test lists can pass.
    monkeypatch.setattr(validate, "LAST_ARTICLE", 3)


def test_numbering_passes_for_unique_ints(small_code):
    assert check_numbering([make(1), make(2), make(3)]).passed


def test_numbering_detects_duplicate(small_code):
    assert not check_numbering([make(1), make(1)]).passed


def test_numbering_rejects_string_number(small_code):
    assert not check_numbering([make("2")]).passed


def test_numbering_rejects_out_of_range(small_code):
    assert not check_numbering([make(99999)]).passed


def test_gaps_detects_missing_article(small_code):
    check = check_gaps([make(1), make(3)])
    assert not check.passed
    assert "2" in check.details[0]


def test_gaps_passes_when_complete(small_code):
    assert check_gaps([make(1), make(2), make(3)]).passed


def test_empty_arabic_fails_for_normal_article():
    assert not check_arabic_text([make(text_ar="   ")]).passed


def test_empty_arabic_is_allowed_for_repealed_article():
    assert check_arabic_text([make(text_ar="", is_repealed=True)]).passed


def test_length_too_short_and_too_long_fail():
    assert not check_length([make(text_ar="ab")]).passed
    assert not check_length([make(text_ar="x" * 3001)]).passed


def test_length_normal_passes():
    assert check_length([make()]).passed


def test_repealed_flag_must_be_bool():
    assert not check_repealed_flag([make(is_repealed="false")]).passed
    assert check_repealed_flag([make(is_repealed=True)]).passed


def test_citation_must_match_article_number():
    assert check_citation_format([make(5)]).passed
    assert not check_citation_format([make(5, citation="Egyptian Civil Code, Article 6")]).passed