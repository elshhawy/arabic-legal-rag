import pytest

from legalrag.arabic import (
    fix_visual_line,
    is_arabic_article_header,
    normalize_paragraph_markers,
    to_western_digits,
)


def test_arabic_indic_digits_become_int():
    result = to_western_digits("١٤٧")
    assert result == 147
    assert isinstance(result, int)  # must be int, not the string "147"


@pytest.mark.parametrize("raw, expected", [("١", 1), ("٣٠", 30), ("١١٤٩", 1149), ("147", 147)])
def test_to_western_digits_values(raw, expected):
    assert to_western_digits(raw) == expected


def test_to_western_digits_rejects_garbage():
    # Garbage must fail loudly instead of producing a wrong article number.
    with pytest.raises(ValueError):
        to_western_digits("abc")


def test_header_detected_with_arabic_digits():
    assert is_arabic_article_header("مادة ١٤٧") == 147


def test_header_detected_with_brackets():
    assert is_arabic_article_header("مادة (147)") == 147


def test_normal_sentence_is_not_a_header():
    assert is_arabic_article_header("مادة العقد شريعة المتعاقدين") is None


@pytest.mark.parametrize("raw", [")١(", "(١(", ")١)", "(١)"])
def test_paragraph_markers_become_standard(raw):
    # The source PDF stores brackets inconsistently; all must become "(١)".
    assert normalize_paragraph_markers(raw) == "(١)"


def test_visual_line_is_reversed_to_reading_order():
    # The PDF stores Arabic in visual (reversed) order.
    assert fix_visual_line("دقعلا") == "العقد"


def test_numbers_keep_their_order_after_reversal():
    # Numbers are written left-to-right even inside Arabic text.
    assert fix_visual_line("١٤٧") == "١٤٧"


def test_ligature_placeholder_is_restored():
    # "\ue000" is the private-use placeholder for the lam-alef ligature.
    assert fix_visual_line("\ue000") == "لا"