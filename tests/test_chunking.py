import pytest

from legalrag.chunking import chunk_text

TEXT = "aa bb cc dd ee ff"
# one (start, end) pair per word, like a tokenizer's offset mapping
OFFSETS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]


def test_zero_chunk_size_keeps_whole_article():
    assert chunk_text(TEXT, OFFSETS, chunk_size=0, overlap=0) == [TEXT]


def test_short_article_is_not_split():
    assert chunk_text(TEXT, OFFSETS, chunk_size=10, overlap=2) == [TEXT]


def test_split_with_overlap():
    pieces = chunk_text(TEXT, OFFSETS, chunk_size=4, overlap=2)
    assert pieces == ["aa bb cc dd", "cc dd ee ff"]


def test_split_without_overlap_covers_everything():
    pieces = chunk_text(TEXT, OFFSETS, chunk_size=4, overlap=0)
    assert pieces == ["aa bb cc dd", "ee ff"]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(TEXT, OFFSETS, chunk_size=2, overlap=2)