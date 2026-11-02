"""Thực thi notebook view và lưu toàn bộ cell output trong artifact run riêng."""
from pathlib import Path
import argparse
import ast
import sys
import nbformat
from nbclient import NotebookClient
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.utils.runs import ROOT, run_context, save_json, hash_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    all_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    # Template GPU không được chạy vô tình khi render báo cáo offline.
    paths = [p for p in all_paths if nbformat.read(p, as_version=4).metadata.get("lab03", {}).get("execution_mode") != "train"]
    skipped = [p.name for p in all_paths if p not in paths]
    config = {"mode":"view", "kernel":args.kernel, "notebooks":[p.name for p in paths],
              "purpose":"persist all executed notebook outputs; no GPU training", "skipped_gpu_templates":skipped}
    with run_context("notebook_render", config) as out:
        results = []
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            modes = []
            for cell in notebook.cells:
                if cell.cell_type != "code": continue
                for statement in ast.walk(ast.parse(cell.source)):
                    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODE" for t in statement.targets):
                        modes.append(ast.literal_eval(statement.value))
            if modes != ["view"]:
                raise ValueError("Renderer requires MODE=view; run training separately")
            NotebookClient(notebook, timeout=180, kernel_name=args.kernel,
                resources={"metadata":{"path":str(ROOT / "notebooks")}}).execute()
            nbformat.validate(notebook)
            errors = sum(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
            if errors: raise AssertionError(f"Notebook failed: {path.name}")
            nbformat.write(notebook,path)
            result = {"notebook":path.name,"mode":"view","executed_code_cells":sum(c.cell_type == "code" for c in notebook.cells),
                      "errors":errors,"training_executed":False,"artifact":path.relative_to(ROOT).as_posix(),
                      "notebook_sha256":hash_file(path),
                      "inline_pngs":sum("image/png" in o.get("data", {}) for c in notebook.cells if c.cell_type=="code" for o in c.outputs)}
            results.append(result); print("Executed and saved:",path.name,flush=True)
        save_json(out / "execution_summary.json", results)
        save_json(ROOT / "validation/notebook_execution.json", {"artifact_run":out.name,"notebooks":results})
    print("Complete notebook artifacts:",out)
