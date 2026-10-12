import re

# The PDF draws "لا" (and لأ لإ لآ) as ONE ligature glyph. Before reversing a
# line we hide each ligature behind a private-use character, then restore it.
LIGATURES = {
    "\ue000": "لا",
    "\ue001": "لأ",
    "\ue002": "لإ",
    "\ue003": "لآ",
}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Reversing a line also mirrors the brackets, so we swap them back.
MIRROR_BRACKETS = str.maketrans("()", ")(")


def to_western_digits(text):
    """Turn Arabic-Indic digits into an int: '٣٠' -> 30."""
    return int(text.translate(ARABIC_DIGITS))


def fix_visual_line(line):
    """Convert one Arabic line from visual order to reading order."""
    reversed_line = line[::-1].translate(MIRROR_BRACKETS)
    # Numbers are written left-to-right even in Arabic, so flip them back.
    fixed_numbers = re.sub(r"[0-9٠-٩]+", lambda m: m.group(0)[::-1], reversed_line)
    for placeholder, letters in LIGATURES.items():
        fixed_numbers = fixed_numbers.replace(placeholder, letters)
    return fixed_numbers


def normalize_paragraph_markers(text):
    """Rewrite any ')N(' / '(N(' / ')N)' paragraph marker as '(N)'.

    The source PDF is inconsistent: some brackets are stored mirrored and
    some are not, so we cannot fix them by reversing alone.
    """
    return re.sub(r"[()]\s*([0-9٠-٩]+)\s*[()]", r"(\1)", text)


def is_arabic_article_header(line):
    """Return the article number if the line is an Arabic header, else None."""
    match = re.match(r"^مادة\s*[()]?\s*([0-9٠-٩]+)\s*[()]?\s*$", line)
    if match:
        return to_western_digits(match.group(1))
    return None