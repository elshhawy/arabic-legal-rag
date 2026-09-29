"""Print 20 random articles so you can check them by eye.

Put this file anywhere inside the project (project root is best) and run:
    python spot_check.py

It finds data/processed/civil_code.json by itself, prints the articles, and
also saves them to spot_check_report.txt (open that file in VS Code: Arabic
always displays correctly there, even if the terminal shows it badly).
"""

import json
import random
import sys
from pathlib import Path

JSON_RELATIVE = Path("data/processed/civil_code.json")
SAMPLE_SIZE = 20
SEED = 7  # fixed seed = the same 20 articles every run


def find_json():
    """Look for the JSON in this folder and up to two folders above it."""
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, here.parent.parent, Path.cwd()):
        candidate = base / JSON_RELATIVE
        if candidate.exists():
            return candidate
    return None


def main():
    json_path = find_json()
    if json_path is None:
        print("Could not find data/processed/civil_code.json")
        print("Run this first, from the project root:")
        print("    python src/legalrag/extract.py")
        sys.exit(1)

    articles = json.loads(json_path.read_text(encoding="utf-8"))
    normal = [a for a in articles if not a["is_repealed"]]

    random.seed(SEED)
    sample = sorted(random.sample(normal, SAMPLE_SIZE), key=lambda a: a["article_number"])

    lines = []
    for a in sample:
        lines.append("=" * 60)
        lines.append(f"Article {a['article_number']} | page {a['source_page']}")
        lines.append(f"Where: {a['book']} > {a['chapter']} > {a['section']} > {a['topic']}")
        lines.append(f"EN: {a['text_en']}")
        lines.append(f"AR: {a['text_ar']}")
    report = "\n".join(lines)

    report_path = json_path.parent.parent.parent / "spot_check_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print("\nSaved to", report_path)


if __name__ == "__main__":
    main()