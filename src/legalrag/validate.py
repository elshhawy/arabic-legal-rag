"""Validate the extracted civil code corpus before trusting it.

Run it after every extraction:
    python -m legalrag.validate

Exits with status 1 if any FAIL check is found, so this can be used as a
CI quality gate later (Session 2).
"""

import json
import sys
from pathlib import Path

CORPUS_PATH = Path("data/processed/civil_code.json")
LAST_ARTICLE = 1149

REQUIRED_FIELDS = {
    "article_number", "part", "book", "chapter", "section", "topic",
    "text_ar", "text_en", "is_repealed", "source_page", "citation",
}

# Reasonable Arabic text length bounds (characters), based on the
# distribution measured across the corpus (median ~206, max ~1413).
MIN_LENGTH = 5
MAX_LENGTH = 3000


class Check:
    """One named validation check with a pass/fail result and details."""

    def __init__(self, name):
        self.name = name
        self.passed = True
        self.details = []

    def fail(self, message):
        self.passed = False
        self.details.append(message)

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"[{status}] {self.name}")
        for detail in self.details[:10]:  # cap noisy output
            print(f"       - {detail}")
        if len(self.details) > 10:
            print(f"       ... and {len(self.details) - 10} more")


def check_schema(articles):
    check = Check("All records have the required fields")
    for article in articles:
        missing = REQUIRED_FIELDS - set(article.keys())
        if missing:
            check.fail(f"Article {article.get('article_number', '?')}: missing {missing}")
    return check


def check_numbering(articles):
    check = Check("Article numbers are unique integers in range")
    seen = set()
    for article in articles:
        number = article["article_number"]
        if not isinstance(number, int):
            check.fail(f"article_number is not an int: {number!r}")
            continue
        if not (1 <= number <= LAST_ARTICLE):
            check.fail(f"Article {number} is outside the valid range")
        if number in seen:
            check.fail(f"Article {number} appears more than once (duplicate)")
        seen.add(number)
    return check


def check_gaps(articles):
    check = Check("No unexplained gaps in article numbers")
    present = {a["article_number"] for a in articles}
    missing = sorted(set(range(1, LAST_ARTICLE + 1)) - present)
    if missing:
        check.fail(f"Missing article numbers: {missing}")
    return check


def check_arabic_text(articles):
    check = Check("Every non-repealed article has Arabic text")
    for article in articles:
        if article["is_repealed"]:
            continue
        if not article["text_ar"].strip():
            check.fail(f"Article {article['article_number']} has empty text_ar")
    return check


def check_length(articles):
    check = Check("Arabic text length is within a plausible range")
    for article in articles:
        if article["is_repealed"]:
            continue
        length = len(article["text_ar"])
        if length < MIN_LENGTH:
            check.fail(f"Article {article['article_number']}: text_ar too short ({length} chars)")
        if length > MAX_LENGTH:
            check.fail(f"Article {article['article_number']}: text_ar too long ({length} chars)")
    return check


def check_repealed_flag(articles):
    check = Check("is_repealed is always a boolean")
    for article in articles:
        if not isinstance(article["is_repealed"], bool):
            check.fail(
                f"Article {article['article_number']}: is_repealed is "
                f"{type(article['is_repealed']).__name__}, not bool"
            )
    return check


def check_citation_format(articles):
    check = Check("citation matches 'Egyptian Civil Code, Article N'")
    for article in articles:
        expected = f"Egyptian Civil Code, Article {article['article_number']}"
        if article["citation"] != expected:
            check.fail(f"Article {article['article_number']}: citation is {article['citation']!r}")
    return check


def main():
    if not CORPUS_PATH.exists():
        print(f"ERROR: {CORPUS_PATH} not found. Run extraction first:")
        print("  python src/legalrag/extract.py")
        sys.exit(1)

    articles = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    checks = [
        check_schema(articles),
        check_numbering(articles),
        check_gaps(articles),
        check_arabic_text(articles),
        check_length(articles),
        check_repealed_flag(articles),
        check_citation_format(articles),
    ]

    print(f"Validating {len(articles)} articles from {CORPUS_PATH}\n")
    for check in checks:
        check.report()

    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")

    if failed:
        print("\nCorpus FAILED validation. Do not proceed to embedding.")
        sys.exit(1)

    print("\nCorpus PASSED validation.")
    sys.exit(0)


if __name__ == "__main__":
    main()