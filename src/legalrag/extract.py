"""Step 0: turn the Egyptian Civil Code PDF into one JSON record per article.

The PDF is one big two-column table (English left, Arabic right), one article
per row. We treat the whole book as ONE stream of rows, so an article that
continues on the next page is completed automatically.
"""

import json
import re
from pathlib import Path

import pdfplumber

from legalrag.arabic import (
    LIGATURES,
    fix_visual_line,
    is_arabic_article_header,
    normalize_paragraph_markers,
)

PDF_PATH = Path("data/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("data/processed/civil_code.json")
REPORT_PATH = Path("reports/extraction_report.json")

COLUMN_SPLIT_X = 298  # left of this = English, right of this = Arabic
LAST_ARTICLE = 1149
MAX_JUMP = 50  # largest allowed gap between two headers (repealed ranges make big gaps)

# Typos that exist in the source PDF itself (checked visually on page 147).
SOURCE_TYPOS = {"العاقر": "العقار"}  # "barren" -> "the real estate"

REPEAL_RANGE = re.compile(r"Articles?\s+(\d+)\s*(?:-|to)\s*(\d+)", re.IGNORECASE)
NUMBERED_TOPIC = re.compile(r"^\d+\s*[.\-\u2013]\s*(.+)$")  # "1. Individuals"
ROMAN = r"[IVXLC]+"
LEVEL_PATTERNS = [
    ("part", re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\s+PART$", re.I)),
    ("book", re.compile(rf"^BOOK\s+({ROMAN})$", re.I)),
    ("chapter", re.compile(rf"^CHAPTER\s+({ROMAN})$", re.I)),
    ("section", re.compile(rf"^SECTION\s+({ROMAN})$", re.I)),
]
LEVELS = ["part", "book", "chapter", "section", "topic"]
# Words a heading title is still allowed to contain in lowercase, e.g.
# "Rules Applicable to Estates that have not been Wound Up".
SMALL_WORDS = {
    "of", "or", "the", "and", "in", "to", "for", "by", "a", "an", "on",
    "that", "have", "has", "had", "not", "which", "who", "with", "as",
    "from", "at", "after", "before", "upon", "into", "been", "is", "are",
}


# ------------------------------------------------------------------ helpers --
def protect_ligatures(page):
    """Swap each ligature glyph for its placeholder before extracting text."""
    placeholder_for = {letters: mark for mark, letters in LIGATURES.items()}
    for char in page.chars:
        mark = placeholder_for.get(char["text"])
        if mark:
            char["text"] = mark


def is_article_header(line):
    """If the line looks like an article header, return (number, rest_of_line).

    Accepts source-PDF typos: 'rticle 452' (missing A) and 'Article1022'
    (missing space), and article text that starts on the same line.
    """
    match = re.match(r"^\w*rticle\s*(\d{1,4})\b\s*(.*)$", line.strip())
    if match:
        return int(match.group(1)), match.group(2)
    return None


def is_new_article(number, rest, last_number):
    """Check this is a real header, not a cross-reference inside an article."""
    # 'Article 901 has been delivered...' is a reference: real headers are
    # followed by nothing or by a new sentence starting with a capital letter.
    if rest and not rest[0].isupper():
        return False
    return last_number < number <= last_number + MAX_JUMP


def smart_title(text):
    """'OBLIGATIONS OR PERSONAL RIGHTS' -> 'Obligations or Personal Rights'."""
    if not text.isupper():
        return text
    words = []
    for i, word in enumerate(text.lower().split()):
        if re.fullmatch(ROMAN.lower(), word):
            words.append(word.upper())
        elif i > 0 and word in SMALL_WORDS:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def looks_like_label(lines):
    """A short sub-heading such as 'Consent:' or 'Rules Applicable to...'."""
    if not lines or len(lines) > 2:
        return False
    for line in lines:
        if len(line) > 60 or line.endswith(".") or not line[0].isupper():
            return False
    if lines[-1].endswith(":"):
        return True
    words = re.findall(r"[A-Za-z]+", " ".join(lines))
    return all(w[0].isupper() for w in words if len(w) > 3 and w.lower() not in SMALL_WORDS)


def match_level(line):
    """If the line opens a new hierarchy level, return (level, label)."""
    for level, pattern in LEVEL_PATTERNS:
        match = pattern.match(line)
        if match:
            word = match.group(1)
            if level == "part":
                return level, f"{word.title()} Part"
            return level, f"{level.capitalize()} {word.upper()}"
    match = NUMBERED_TOPIC.match(line)
    if match:
        return "topic", match.group(1).strip()
    return None


class Hierarchy:
    """Remembers where we are: part > book > chapter > section > topic."""

    def __init__(self):
        self.values = {level: "" for level in LEVELS}
        self.subtopic = ""

    def open_level(self, level, label):
        for lower in LEVELS[LEVELS.index(level):]:
            self.values[lower] = ""
        self.values[level] = label
        self.subtopic = ""

    def add_title(self, level, title):
        """Attach the title line that follows e.g. 'BOOK I'."""
        self.values[level] = f"{self.values[level]}: {smart_title(title)}"

    def append_text(self, level, text):
        self.values[level] = f"{self.values[level]} {smart_title(text)}".strip()

    def add_subtopic(self, text):
        self.subtopic = f"{self.subtopic} > {text}" if self.subtopic else text

    def snapshot(self):
        data = dict(self.values)
        if self.subtopic:
            data["topic"] = (
                f"{data['topic']} > {self.subtopic}" if data["topic"] else self.subtopic
            )
        return data


# ------------------------------------------------------------ reading rows --
def row_bands(page):
    """Cut a page into table rows using the horizontal lines of the left cells."""
    ys = sorted(
        {
            round(edge["top"], 1)
            for edge in page.edges
            if edge["orientation"] == "h"
            and edge["x0"] <= 40
            and 250 <= edge["x1"] <= COLUMN_SPLIT_X + 2
            and edge["x1"] - edge["x0"] > 200
        }
    )
    merged = []
    for y in ys:
        if not merged or y - merged[-1] > 2.5:
            merged.append(y)
    return list(zip(merged, merged[1:]))


def read_side(page, x0, x1, top, bottom):
    box = page.within_bbox((x0, top - 0.5, x1, bottom + 0.5))
    text = box.extract_text() or ""
    return [line.strip() for line in text.split("\n") if line.strip()]


# --------------------------------------------------------------- the parser --
class CorpusBuilder:
    """Reads rows one after another and builds the article records."""

    def __init__(self):
        self.hierarchy = Hierarchy()
        self.articles = {}
        self.current_en = None  # article receiving English continuation lines
        self.last_ar = None  # article receiving Arabic continuation lines
        self.last_number = 0
        self.repeal_notes = []

    def _start_article(self, number, page_no, rest):
        record = {"number": number, "page": page_no, "en": [], "ar": []}
        record.update(self.hierarchy.snapshot())
        if rest:
            record["en"].append(rest)
        self.articles[number] = record
        self.current_en = number
        self.last_number = number

    def _consume_headings(self, lines):
        """Consume heading lines at the start of a row; return how many."""
        used, active, title_done = 0, None, False
        while used < len(lines):
            line = lines[used]
            header = is_article_header(line)
            if header and is_new_article(*header, self.last_number):
                break
            hit = match_level(line)
            if hit:
                self.hierarchy.open_level(*hit)
                active, title_done = hit[0], False
            elif active is None:
                break
            elif active == "topic":
                previous = lines[used - 1]
                if line.endswith(":") or (looks_like_label([line]) and len(previous) < 45):
                    self.hierarchy.add_subtopic(line.rstrip(":").strip())
                elif looks_like_label([line]):
                    self.hierarchy.append_text("topic", line)
                else:
                    break
            elif not title_done:
                self.hierarchy.add_title(active, line)
                title_done = True
            elif looks_like_label([line]):
                self.hierarchy.append_text(active, line)
            else:
                break
            used += 1
        return used

    def add_row(self, page_no, left, right):
        if not left and not right:
            return

        first = is_article_header(left[0]) if left else None
        first_is_header = bool(first and is_new_article(*first, self.last_number))

        # A stand-alone row such as 'Articles 389-417 repealed'.
        if left and not first_is_header:
            joined = " ".join(left)
            match = REPEAL_RANGE.search(joined)
            if match and "repealed" in joined.lower():
                self.repeal_notes.append(
                    {
                        "first": int(match.group(1)),
                        "last": int(match.group(2)),
                        "text_en": joined,
                        "right": right,
                        "page": page_no,
                    }
                )
                return

        consumed = self._consume_headings(left)
        if consumed == 0 and left and not first_is_header:
            short_arabic = len(right) == 1 and len(right[0]) <= 45
            if looks_like_label(left) and (
                left[-1].endswith(":") or not right or short_arabic
            ):
                self.hierarchy.subtopic = " ".join(left).rstrip(":").strip()
                consumed = len(left)
        had_heading = consumed > 0

        started_here = None
        for line in left[consumed:]:
            header = is_article_header(line)
            if header and is_new_article(*header, self.last_number):
                self._start_article(header[0], page_no, header[1])
                started_here = started_here or header[0]
            elif self.current_en is not None:
                self.articles[self.current_en]["en"].append(line)

        self._add_arabic(right, started_here, had_heading)

    def _add_arabic(self, right, started_here, had_heading):
        """Split the Arabic side by its own 'مادة N' headers and file each block."""
        blocks = [(None, [])]
        for raw in right:
            number = is_arabic_article_header(fix_visual_line(raw))
            if number is not None:
                blocks.append((number, []))
            else:
                blocks[-1][1].append(raw)

        for index, (number, lines) in enumerate(blocks):
            if not lines:
                if number is not None:
                    self.last_ar = number
                continue
            if number is None:
                # Text before any Arabic header: either the Arabic version of
                # a heading (drop it) or the continuation of the previous article.
                if had_heading and (started_here is None or index == 0):
                    continue
                number = started_here if started_here is not None else self.last_ar
            else:
                self.last_ar = number
            if number in self.articles:
                self.articles[number]["ar"].extend(lines)


# ------------------------------------------------------------ final records --
def build_records(builder):
    """Turn raw pieces into clean records (with repealed articles added)."""
    records = {}
    for number, raw in builder.articles.items():
        text_ar = " ".join(fix_visual_line(line) for line in raw["ar"])
        record = {
            "article_number": number,
            "part": raw["part"],
            "book": raw["book"],
            "chapter": raw["chapter"],
            "section": raw["section"],
            "topic": raw["topic"],
            "text_ar": normalize_paragraph_markers(text_ar).strip(),
            "text_en": " ".join(raw["en"]).strip(),
            "is_repealed": False,
            "source_page": raw["page"],
            "citation": f"Egyptian Civil Code, Article {number}",
        }
        for wrong, right in SOURCE_TYPOS.items():
            record["text_ar"] = record["text_ar"].replace(wrong, right)
        records[number] = record

    # Repealed ranges: either a note inside an article (54-80) or a row (389-417).
    ranges = []
    for number, record in list(records.items()):
        match = REPEAL_RANGE.search(record["text_en"])
        if match and "repealed" in record["text_en"].lower():
            ranges.append((int(match.group(1)), int(match.group(2)), record["text_en"],
                           record["text_ar"], record["source_page"]))
    for note in builder.repeal_notes:
        text_ar = normalize_paragraph_markers(
            " ".join(fix_visual_line(line) for line in note["right"])
        )
        ranges.append((note["first"], note["last"], note["text_en"], text_ar, note["page"]))

    for first, last, text_en, text_ar, page in ranges:
        for number in range(first, last + 1):
            earlier = [n for n in records if n < number]
            base = records.get(number) or dict(records[max(earlier)])
            base.update(
                {
                    "article_number": number,
                    "source_page": page,
                    "text_en": f"Article {number} has been repealed. ({text_en.lstrip('* ').strip()})",
                    "text_ar": text_ar or "ملغاة",
                    "is_repealed": True,
                    "citation": f"Egyptian Civil Code, Article {number}",
                }
            )
            records[number] = base
    return records


def repair_article_1022(records):
    """Fix a defect in the source PDF: Arabic of article 1022 sits in the 1021 cell.

    On page 147 the Arabic paragraphs (2) and (3) of the 1021 cell are really
    the Arabic text of article 1022 (its own Arabic cell is empty).
    """
    first, second = records.get(1021), records.get(1022)
    if not first or not second or second["text_ar"]:
        return False
    text = first["text_ar"]
    if "(٢)" not in text or "(٣)" not in text:
        return False
    cut = text.index("(٢)")
    first["text_ar"] = text[:cut].replace("(١)", "").strip()
    second["text_ar"] = text[cut:].replace("(٢)", "(١)", 1).replace("(٣)", "(٢)", 1).strip()
    return True


def main():
    builder = CorpusBuilder()
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            protect_ligatures(page)
            for top, bottom in row_bands(page):
                left = read_side(page, 0, COLUMN_SPLIT_X, top, bottom)
                right = read_side(page, COLUMN_SPLIT_X, page.width, top, bottom)
                builder.add_row(page_no, left, right)

    records = build_records(builder)
    repaired_1022 = repair_article_1022(records)
    articles = [records[n] for n in sorted(records)]

    numbers = {a["article_number"] for a in articles}
    report = {
        "articles_total": len(articles),
        "repealed": [a["article_number"] for a in articles if a["is_repealed"]],
        "missing": sorted(set(range(1, LAST_ARTICLE + 1)) - numbers),
        "without_english": [a["article_number"] for a in articles if not a["text_en"]],
        "without_arabic": [a["article_number"] for a in articles if not a["text_ar"]],
        "manual_repairs": ["1022: Arabic moved out of the 1021 cell"] if repaired_1022 else [],
        "known_source_issues": [
            "The source PDF itself contains scattered Arabic typos (e.g. "
            "'يكبت' for 'يثبت', 'كير' for 'غير', 'يترل' for 'ينزل') found "
            "during a 20-article random spot check. These are left as-is; "
            "only the 'العاقر'->'العقار' typo (10 occurrences) was corrected "
            "because it was verified visually against the PDF page image."
        ],
    }

    for path, payload in ((OUTPUT_PATH, articles), (REPORT_PATH, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Total articles:", report["articles_total"])
    print("Repealed:", len(report["repealed"]))
    print("Missing:", report["missing"])
    print("Without English:", report["without_english"])
    print("Without Arabic:", report["without_arabic"])
    print("Saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()