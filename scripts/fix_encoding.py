"""One-shot script: add encoding="utf-8" to text-mode open()/read_text()/write_text() calls.

Idempotent: re-running on already-fixed files is a no-op.
Skips binary modes ("rb", "wb", "ab", "rb+", etc.).
Run from repo root: `venv/Scripts/python.exe scripts/fix_encoding.py`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "tests", ROOT / "src"]

# open(arg, "r") -> open(arg, "r", encoding="utf-8")  [only if no encoding present]
RE_OPEN_R = re.compile(r'\bopen\(([^,()]+),\s*"r"\)')
# open(arg, "w") -> open(arg, "w", encoding="utf-8")
RE_OPEN_W = re.compile(r'\bopen\(([^,()]+),\s*"w"\)')
# open(arg) used in `with open(arg) as f:` style (default mode is "r")
RE_OPEN_BARE = re.compile(r'\bopen\(([^,()]+)\)(\s*as\s)')
# .read_text() -> .read_text(encoding="utf-8")
RE_READ_TEXT = re.compile(r'\.read_text\(\)')
# .write_text(X) where X is a single arg without encoding
RE_WRITE_TEXT = re.compile(r'\.write_text\(([^()]+)\)(?!\s*\.)')


def fix_content(text: str) -> tuple[str, int]:
    changes = 0

    def sub_r(m: re.Match) -> str:
        nonlocal changes
        changes += 1
        return f'open({m.group(1)}, "r", encoding="utf-8")'

    def sub_w(m: re.Match) -> str:
        nonlocal changes
        changes += 1
        return f'open({m.group(1)}, "w", encoding="utf-8")'

    def sub_bare(m: re.Match) -> str:
        nonlocal changes
        changes += 1
        return f'open({m.group(1)}, encoding="utf-8"){m.group(2)}'

    def sub_read(m: re.Match) -> str:
        nonlocal changes
        changes += 1
        return '.read_text(encoding="utf-8")'

    def sub_write(m: re.Match) -> str:
        nonlocal changes
        arg = m.group(1)
        # Don't double-add if encoding already inside arg
        if "encoding=" in arg:
            return m.group(0)
        return f'.write_text({arg}, encoding="utf-8")'

    text = RE_OPEN_R.sub(sub_r, text)
    text = RE_OPEN_W.sub(sub_w, text)
    text = RE_OPEN_BARE.sub(sub_bare, text)
    text = RE_READ_TEXT.sub(sub_read, text)
    text = RE_WRITE_TEXT.sub(sub_write, text)
    return text, changes


def process_file(path: Path) -> int:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Source file itself isn't utf-8 — skip
        return 0
    new, changes = fix_content(original)
    if changes and new != original:
        path.write_text(new, encoding="utf-8")
        return changes
    return 0


def main() -> int:
    total_files = 0
    total_changes = 0
    for target in TARGETS:
        if not target.is_dir():
            continue
        for py in sorted(target.rglob("*.py")):
            n = process_file(py)
            if n:
                total_files += 1
                total_changes += n
                print(f"  {py.relative_to(ROOT)}: +{n}")
    print(f"\nFixed {total_changes} call sites across {total_files} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
