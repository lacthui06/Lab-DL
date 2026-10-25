"""Finish packaging and execute the report notebook after training completes."""
from pathlib import Path
import json
import subprocess
import sys
import time
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.data import file_hash
from src.experiment import write_json

run = Path(sys.argv[1]).resolve()
deadline = time.monotonic() + 14400
while True:
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "completed":
        break
    if status["status"] == "failed":
        raise RuntimeError(status)
    if time.monotonic() > deadline:
        raise TimeoutError("Training not completed within four hours")
    time.sleep(5)
with (root / "outputs" / "delivery_validation.log").open("w", encoding="utf-8") as log:
    test = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', 'tests', '-q'], cwd=root,
                          text=True, encoding='utf-8', capture_output=True)
    (run / 'pytest_validation.log').write_text(test.stdout + test.stderr, encoding='utf-8')
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    for script, arguments in [('build_notebook.py', []), ('finalize.py', []),
                              ('execute_notebook.py', []),
                              ('execute_notebook.py', ['01_cifar10_eda_architecture.ipynb'])]:
        result = subprocess.run([sys.executable, '-X', 'utf8', str(root / "scripts" / script), *arguments], cwd=root,
                                text=True, encoding="utf-8", capture_output=True)
        log.write(result.stdout + result.stderr)
        log.flush()
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError(f"{script} failed")
    result = subprocess.run([sys.executable, str(root / "scripts" / "infer.py"),
        "--checkpoint", str(run / "resnet18" / "finetune" / "best.pt"), "--sample-index", "0"],
        cwd=root, text=True, encoding="utf-8", capture_output=True, check=True)
    log.write(result.stdout + result.stderr)
    write_json(run / "cli_inference_example.json", json.loads(result.stdout))
notebook = root / "notebooks" / "lab_02_transfer_learning.ipynb"
nb = json.loads(notebook.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert all(c["execution_count"] is not None for c in code_cells)
assert not any(o.get("output_type") == "error" for c in code_cells for o in c["outputs"])
verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
eda_notebook = root / 'notebooks' / '01_cifar10_eda_architecture.ipynb'
eda = json.loads(eda_notebook.read_text(encoding='utf-8'))
eda_cells = [c for c in eda['cells'] if c['cell_type'] == 'code']
assert all(c['execution_count'] is not None for c in eda_cells)
assert not any(o.get('output_type') == 'error' for c in eda_cells for o in c['outputs'])
verification.update({"notebook_execution": "passed", "executed_code_cells": len(code_cells),
                     "notebook_sha256": file_hash(notebook), "cli_inference": "passed",
                     'eda_notebook_execution': 'passed', 'eda_executed_code_cells': len(eda_cells),
                     'eda_notebook_sha256': file_hash(eda_notebook),
                     'tests': test.stdout.strip()})
write_json(run / "verification.json", verification)
write_json(run / "manifest.json", {str(p.relative_to(run)): file_hash(p)
    for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"})
print("DELIVERY VERIFIED", flush=True)
print(run / "report_vi.html", flush=True)
