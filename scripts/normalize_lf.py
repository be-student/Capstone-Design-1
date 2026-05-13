"""Re-write tracked text files with LF line endings.

The encoding fix run on Windows accidentally wrote CRLF; the upstream repo
uses LF only. This restores LF without touching content. Idempotent.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only touch files git already considers modified, so we don't disturb
# unrelated working-tree state.
result = subprocess.run(
    ["git", "diff", "--name-only"],
    capture_output=True, text=True, cwd=ROOT, check=True,
)
files = [ROOT / p for p in result.stdout.splitlines() if p.endswith((".py", ".yaml", ".yml", ".md", ".txt", ".json"))]

changed = 0
for path in files:
    if not path.is_file():
        continue
    raw = path.read_bytes()
    if b"\r\n" not in raw:
        continue
    fixed = raw.replace(b"\r\n", b"\n")
    path.write_bytes(fixed)
    changed += 1

print(f"Normalized {changed} files to LF.")
