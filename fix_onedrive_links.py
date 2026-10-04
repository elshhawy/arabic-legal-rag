"""Rewrite every project file as a plain file, removing OneDrive reparse points.

Run once after moving a project out of a OneDrive-synced folder.
"""

from pathlib import Path

# Folders Docker never needs to read; skip them entirely.
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}

root = Path(".")
fixed = []

for path in root.rglob("*"):
    if not path.is_file():
        continue
    if any(part in SKIP_DIRS for part in path.parts):
        continue

    try:
        data = path.read_bytes()
    except OSError as e:
        print(f"SKIP (could not read): {path} -- {e}")
        continue

    path.unlink()
    path.write_bytes(data)
    fixed.append(path)

print(f"\nRewrote {len(fixed)} files as plain local files.")