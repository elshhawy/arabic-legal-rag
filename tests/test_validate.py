from legalrag.validate import check_schema

GOOD = {
    "article_number": 1,
    "part": "p",
    "book": "b",
    "chapter": "c",
    "section": "s",
    "topic": "t",
    "text_ar": "text",
    "text_en": "text",
    "is_repealed": False,
    "source_page": 1,
    "citation": "Egyptian Civil Code, Article 1",
}


def test_schema_passes_for_complete_record():
    assert check_schema([GOOD]).passed


def test_schema_fails_when_field_missing():
    bad = {k: v for k, v in GOOD.items() if k != "citation"}
    check = check_schema([GOOD, bad])
    assert not check.passed
    assert "citation" in check.details[0]
