"""Installs the repository Git hooks.

Hooks are not committed, because `.git/hooks/` is local to each clone. This
script copies the tracked hook into place so a fresh clone can enable it with
one command:

    python scripts/install_hooks.py

Running it twice is safe: the hook is overwritten with the tracked copy.
"""

import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS_SRC = ROOT / "scripts" / "pre-commit"
HOOKS_DST = ROOT / ".git" / "hooks" / "pre-commit"


def main() -> int:
    if not (ROOT / ".git").is_dir():
        print(f"Not a Git repository: {ROOT}")
        return 1
    if not HOOKS_SRC.is_file():
        print(f"Missing hook source: {HOOKS_SRC}")
        return 1

    HOOKS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HOOKS_SRC, HOOKS_DST)

    # Git only runs the hook if it is executable. On Windows this is advisory,
    # but on Linux and on Kaggle it is required.
    mode = HOOKS_DST.stat().st_mode
    HOOKS_DST.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Installed {HOOKS_DST}")
    print("The hook runs on every commit and blocks:")
    print("  - files that are not valid UTF-8")
    print("  - U+FFFD replacement characters")
    print("  - mojibake sequences from mis-decoded UTF-8")
    print("  - emoji, box-drawing, block and geometric-shape characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
