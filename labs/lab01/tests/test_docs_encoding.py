"""Guards the repository text files against encoding damage.

These tests exist because an earlier session destroyed two Vietnamese documents
by round-tripping them through `Get-Content | Set-Content` in Windows PowerShell:
that pipeline decodes UTF-8 with the ANSI codepage, replaces undefined bytes with
U+FFFD, and writes the result back. The damage is irreversible. The files were
untracked, so there was no copy to restore.

A pre-commit hook blocks this at commit time. This module blocks it at test time
as well, so the failure surfaces during development rather than at commit.
"""

import io
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".ps1", ".csv", ".ipynb",
}
TEXT_NAMES = {".gitignore", ".gitattributes"}

# UTF-8 bytes decoded as a Western codepage and re-encoded. These sequences do
# not occur in correct Vietnamese text.
MOJIBAKE_SEQUENCES = (
    "\u00c3",        # A-tilde
    "\u00c4\u2018",  # A-dieresis + quote
    "\u00e2\u20ac",  # a-circumflex + euro
    "\u00e1\u00ba",
    "\u00e1\u00bb",
)

FORBIDDEN_RANGES = (
    (0x1F300, 0x1FAFF),  # emoji
    (0x2500, 0x257F),    # box drawing
    (0x2580, 0x259F),    # block elements
    (0x25A0, 0x25FF),    # geometric shapes
)


def _tracked_text_files():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        pytest.skip("not a Git checkout; cannot enumerate tracked files")

    files = []
    for rel in result.stdout.splitlines():
        if not rel.strip():
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return files


@pytest.mark.parametrize(
    "path", _tracked_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_tracked_text_file_is_clean_utf8(path):
    """Every tracked text file decodes as UTF-8 with no replacement character."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{path.name} is not valid UTF-8: {exc}")

    assert "\ufffd" not in text, (
        f"{path.name} contains U+FFFD, which means the file was decoded with the "
        f"wrong codepage and re-saved. Restore it from Git; do not try to repair "
        f"it by round-tripping through a shell."
    )


@pytest.mark.parametrize(
    "path", _tracked_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_tracked_text_file_has_no_mojibake(path):
    """No file contains the signature of UTF-8 read as a Western codepage."""
    text = io.open(path, encoding="utf-8").read()
    for sequence in MOJIBAKE_SEQUENCES:
        assert sequence not in text, (
            f"{path.name} contains mojibake sequence {sequence!r}. The text was "
            f"decoded with the wrong codepage at some point."
        )


@pytest.mark.parametrize(
    "path", _tracked_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_tracked_text_file_has_no_forbidden_characters(path):
    """No emoji, box-drawing, block, or geometric-shape characters."""
    text = io.open(path, encoding="utf-8").read()
    for char in set(text):
        codepoint = ord(char)
        for low, high in FORBIDDEN_RANGES:
            assert not (low <= codepoint <= high), (
                f"{path.name} contains U+{codepoint:04X}, which the project style "
                f"rules forbid in source, docs, and notebook cells."
            )


@pytest.mark.parametrize("name", ["eda-image.md", "feature-image.md"])
def test_image_flow_docs_stay_short(name):
    """Each shared flow document must remain readable in one sitting.

    A flow document that grows past roughly 200 lines is a sign that measured
    results or lab-specific detail leaked into it. Those belong in the lab's
    walkthrough, not in the shared procedure.
    """
    path = REPO_ROOT / "docs" / name
    assert path.is_file(), f"docs/{name} is missing"
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= 200, (
        f"docs/{name} has {lines} lines. Keep the shared flow under 200 lines and "
        f"move lab-specific numbers into walkthrough_lab_NN.md."
    )
