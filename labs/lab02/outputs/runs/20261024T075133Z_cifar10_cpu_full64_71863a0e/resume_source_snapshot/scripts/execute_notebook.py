"""Execute the view notebook with the project's interpreter; preserve outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import json
root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / (sys.argv[1] if len(sys.argv) > 1 else "lab_02_transfer_learning.ipynb")
notebook = nbformat.read(path, as_version=4)
kernels = root / ".jupyter" / "kernels"
spec = kernels / "lab02"
spec.mkdir(parents=True, exist_ok=True)
(spec / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Lab 02", "language": "python"}), encoding="utf-8")
manager = KernelManager(kernel_name="lab02",
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
client = NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(root)}},
                        kernel_name="lab02", km=manager)
client.extra_arguments = ["--HistoryManager.enabled=False"]
client.execute()
nbformat.write(notebook, path)
print("Executed successfully:", path)
