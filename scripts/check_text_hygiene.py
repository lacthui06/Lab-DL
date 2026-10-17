"""Blocks commits that would add encoding damage or forbidden characters.

Why this exists: an earlier session edited a UTF-8 Vietnamese document with
`Get-Content | Set-Content` in Windows PowerShell. That pipeline decodes the
file with the ANSI codepage, replaces every undefined byte with U+FFFD, and
writes the result back. The damage is not reversible, and because the files were
untracked there was no copy to restore. A hook cannot undo that class of
mistake, but it can stop the damaged file from being committed, which keeps the
last good version in Git.

Checks, applied to text files staged in this commit only:
1. Valid UTF-8 (a file that does not decode is already broken).
2. No U+FFFD replacement character.
3. No mojibake sequences (UTF-8 read as a Western codepage, then re-encoded).
4. No emoji, box-drawing, block, or geometric-shape characters, which the
   project style rules forbid in source, docs, and notebook cells.

Exit code 1 blocks the commit.
"""

import subprocess
import sys
import unicodedata

TEXT_EXT = (
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".ps1", ".csv", ".ipynb", ".html", ".css", ".js", ".ts",
)
TEXT_NAMES = {".gitignore", ".gitattributes"}

MOJIBAKE = (
    "\u00c3",        # A-tilde: UTF-8 lead byte decoded as Latin-1
    "\u00c4\u2018",  # A-dieresis + quote: Vietnamese đ mis-decoded
    "\u00e2\u20ac",  # a-circumflex + euro: UTF-8 punctuation mis-decoded
    "\u00e1\u00ba",  # a-acute + ordmasculine: Vietnamese lowercase mis-decoded
    "\u00e1\u00bb",
)

FORBIDDEN_RANGES = (
    (0x1F300, 0x1FAFF),  # emoji and pictographs
    (0x2500, 0x257F),    # box drawing
    (0x2580, 0x259F),    # block elements
    (0x25A0, 0x25FF),    # geometric shapes
)
FORBIDDEN_SINGLE = {0x2705, 0x274C, 0x2B50, 0x270C, 0xFE0F}


def staged_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return [p for p in result.stdout.splitlines() if p.strip()]


def is_text_file(path):
    lower = path.lower()
    return lower.endswith(TEXT_EXT) or lower.rsplit("/", 1)[-1] in TEXT_NAMES


def staged_bytes(path):
    result = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def inspect(path, raw):
    """Returns a list of (kind, detail) problems for one file."""
    problems = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [("not valid UTF-8", str(exc)[:80])]

    count = text.count("\ufffd")
    if count:
        problems.append(("U+FFFD replacement char", f"x{count}"))

    for sig in MOJIBAKE:
        n = text.count(sig)
        if n:
            problems.append(("mojibake", f"{sig!r} x{n}"))
            break

    found = {}
    for ch in set(text):
        cp = ord(ch)
        if cp in FORBIDDEN_SINGLE or any(lo <= cp <= hi for lo, hi in FORBIDDEN_RANGES):
            found[ch] = text.count(ch)
    if found:
        detail = ", ".join(
            f"U+{ord(c):04X} {unicodedata.name(c, '?')[:28]} x{n}"
            for c, n in sorted(found.items(), key=lambda kv: -kv[1])[:3]
        )
        problems.append(("emoji/box-drawing", detail))

    return problems


def main():
    files = [p for p in staged_files() if is_text_file(p)]
    failures = []
    for path in files:
        raw = staged_bytes(path)
        if raw is None:
            continue
        for kind, detail in inspect(path, raw):
            failures.append((path, kind, detail))

    if not failures:
        return 0

    print("Commit blocked: text hygiene check failed.\n")
    for path, kind, detail in failures:
        print(f"  {kind}")
        print(f"    file:   {path}")
        print(f"    detail: {detail}\n")
    print("Fix the file, re-stage it, and commit again.")
    print("Do not repair a damaged file by round-tripping it through a shell;")
    print("that is what causes this damage. Restore the last good version from")
    print("Git instead: git checkout HEAD -- <path>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
