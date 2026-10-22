"""Unique run folders, durable logs and checksum manifests."""
import contextlib
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .helpers import OUTPUTS_DIR, collect_environment_snapshot


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.log.write(text)
        self.log.flush()
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            self.stream.write(text.encode(encoding, errors="backslashreplace").decode(encoding))
    def flush(self):
        self.stream.flush()
        self.log.flush()


class Run:
    """Capture stdout/stderr even on failure; never overwrite a previous run."""
    def __init__(self, name, config=None, root=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root or OUTPUTS_DIR / "runs") / f"{stamp}_{name}_{uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config or {}
        self.started_at = datetime.now(timezone.utc).isoformat()
    def __enter__(self):
        write_json(self.path / "config.json", self.config)
        collect_environment_snapshot(self.path / "environment_snapshot.json")
        write_json(self.path / "status.json", {"status": "running", "started_at": self.started_at})
        self.log = (self.path / "console.log").open("w", encoding="utf-8")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, self.log)))
        self.stack.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, self.log)))
        print("Run:", self.path)
        return self
    def __exit__(self, typ, value, tb):
        if typ and typ.__name__ == "TrialPruned":
            print("Trial pruned by the search policy; partial artifacts retained.")
        elif typ:
            traceback.print_exception(typ, value, tb)
        self.stack.close()
        self.log.close()
        write_json(self.path / "status.json", {"status": ("pruned" if typ and typ.__name__ == "TrialPruned" else "failed") if typ else "completed",
                   "error": str(value) if typ else None, "started_at": self.started_at,
                   "finished_at": datetime.now(timezone.utc).isoformat()})
        manifest = []
        for p in sorted(self.path.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                manifest.append({"path": str(p.relative_to(self.path)), "bytes": p.stat().st_size,
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        write_json(self.path / "manifest.json", manifest)
        return False
