"""Publish report evidence from explicitly selected runs without training.

Run from any directory. Original runs remain untouched. Replaced publications,
console logs and exclusion inventories are retained in a local audit run.
"""
import argparse
import hashlib
import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LAB = Path(__file__).resolve().parents[1]
TEXT = {".json", ".csv", ".txt"}
IMAGES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
PRIVATE = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY-----|(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*[\"']?[A-Za-z0-9_/-]{16,})", re.I)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize(text):
    # Keep provenance portable while removing the original user's local paths.
    text = re.sub(r"[A-Za-z]:\\[^\"\n`]+", lambda match: match.group().replace("\\", "/"), text)
    text = re.sub(r"[A-Za-z]:/Users/[^/\"\n]+/Documents/Lab DL/labs/lab01", "<LAB>", text, flags=re.I)
    text = re.sub(r"[A-Za-z]:/Users/[^/\"\n]+", "<USER_HOME>", text, flags=re.I)
    text = re.sub(r"/(?:home|Users)/[^/\"\n]+", "<USER_HOME>", text)
    if PRIVATE.search(text):
        raise ValueError("Potential credential detected; publication stopped. Inspect local audit log.")
    return text


def clean_value(value):
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    return value


def selected_runs(config, extra):
    result = {}
    for key, value in config["runs"].items():
        if isinstance(value, dict):
            for name, path in value.items():
                result[f"models/{name}"] = path
        else:
            result[key] = value
    result.update(extra.get("extra_runs", {}))
    return result


def exclusion(path):
    if path.name in {"manifest.json", "console.log"}:
        return "original manifest or raw console log"
    if path.name in {"train_features.csv", "val_features.csv", "test_features.csv", "spatial_features.csv"}:
        return "per-image preparation table; distribute with full artifact archive"
    if path.suffix.lower() in IMAGES | TEXT:
        return None
    if path.suffix.lower() == ".npz" and ("prediction" in path.name or path.name == "split_indices.npz"):
        return None
    return "checkpoint, prepared data, database, code snapshot, or other non-report file"


def publish(lab=LAB):
    config = read_json(lab / "config/workflow.json")
    extra = read_json(lab / "config/publication.json")
    sources = selected_runs(config, extra)
    audit = lab / "outputs/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_publication_" + uuid4().hex[:8])
    audit.mkdir(parents=True)
    log = logging.getLogger("publication")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(audit / "console.log", encoding="utf-8")
    log.addHandler(handler)
    target = lab / "outputs/published"
    staging = Path(tempfile.mkdtemp(prefix="publication_staging_", dir=audit))
    entries, omitted, mapping = [], [], {}
    try:
        for alias, source in sources.items():
            origin = (lab / source).resolve()
            if not origin.is_relative_to((lab / "outputs/runs").resolve()):
                raise ValueError("Source must be inside outputs/runs")
            status = read_json(origin / "status.json")
            if status["status"] != "completed":
                raise ValueError(f"Selected run not completed: {origin.name}")
            mapping[source.replace("\\", "/")] = "outputs/published/" + alias
            copied = []
            for file in sorted(origin.rglob("*")):
                if not file.is_file():
                    continue
                if file.is_symlink():
                    raise ValueError("Symlink artifacts are not supported")
                relative = file.relative_to(origin)
                reason = exclusion(file)
                if reason:
                    omitted.append({"source_run": origin.name, "path": relative.as_posix(), "reason": reason})
                    continue
                dest = staging / alias / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                if file.suffix.lower() == ".json":
                    write_json(dest, clean_value(read_json(file)))
                elif file.suffix.lower() in TEXT or file.suffix.lower() == ".svg":
                    dest.write_text(sanitize(file.read_text(encoding="utf-8")), encoding="utf-8")
                else:
                    shutil.copyfile(file, dest)
                item = {"path": f"{alias}/{relative.as_posix()}", "bytes": dest.stat().st_size,
                        "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
                copied.append(item)
                entries.append(item)
            write_json(staging / alias / "provenance.json", {"source_run": origin.name, "files": copied})
            log.info("%s: %d artifacts", alias, len(copied))
        # Validate the single source report without creating a duplicate.
        report = (lab / "walkthrough_lab_01.md").read_text(encoding="utf-8")
        for old, new in mapping.items():
            report = report.replace(old, new)
        write_json(staging / "workflow_config.json", clean_value(config))
        write_json(staging / "index.json", {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "sources": sources, "files": entries, "excluded_count": len(omitted),
                   "note": "Report evidence only; original run paths in configs are provenance, not executable dataset paths."})
        (staging / "README.md").write_text("# Published Lab01 evidence\n\nRead [the walkthrough](../../walkthrough_lab_01.md) for the full report. All images from selected completed runs are included. Each section contains tables, metrics and provenance. index.json inventories published files and SHA-256 checksums.\n\nNo training is needed to read this evidence. Training/view code still uses the original runs and prepared data; obtain the full artifact archive to execute it. Raw console logs, checkpoints, prepared data and Optuna databases are local-only. Text exports redact local user paths.\n", encoding="utf-8")
        write_json(audit / "excluded.json", omitted)
        # Resolve every report link before replacing the previous publication.
        for link in re.findall(r"\]\(([^)]+)\)", report):
            if "://" in link or link.startswith("#"):
                continue
            candidate = link.split("#")[0]
            if candidate.startswith("outputs/published/"):
                resolved = staging / candidate.removeprefix("outputs/published/")
            else:
                resolved = (lab / candidate.removeprefix("../../"))
            if not resolved.exists():
                raise FileNotFoundError(f"Broken report link: {link}")
        if target.exists():
            target.rename(audit / "previous_publication")
        try:
            staging.rename(target)
        except Exception:
            backup = audit / "previous_publication"
            if backup.exists():
                backup.rename(target)
            raise
        write_json(audit / "status.json", {"status": "completed", "files": len(entries)})
        print(f"Published {len(entries)} artifacts ({sum(x['bytes'] for x in entries)/1048576:.2f} MiB) to {target}")
        print(f"Local audit and exclusions: {audit}")
        return target
    except Exception:
        log.exception("Publication failed; existing publication preserved")
        write_json(audit / "status.json", {"status": "failed"})
        raise
    finally:
        handler.close()
        log.removeHandler(handler)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    publish()
