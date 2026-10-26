# Lab03 Kaggle: copy từng block # %% vào một cell Code.
# Bật GPU và Internet trước khi chạy.

# %%
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from IPython.display import display, Image, Markdown, FileLink

REPO_URL = "https://github.com/lacthui06/Lab-DL.git"
BRANCH = "lab03"
REPO = Path("/kaggle/working/Lab-DL")
ENV = Path("/kaggle/working/lab03-py312")
PYTHON = str(ENV / "bin/python")

if shutil.which("nvidia-smi") is None:
    raise RuntimeError("Hãy bật Settings > Accelerator > GPU và chạy lại từ cell đầu.")
subprocess.run(["nvidia-smi"], check=True)

if not REPO.exists():
    subprocess.run(["git", "clone", "--single-branch", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
else:
    print("Repo đã tồn tại; giữ nguyên file và kết quả hiện có.")
LAB = REPO / "labs/lab03"
assert (LAB / "scripts/run_study.py").is_file(), "Repo chưa có bản 10 epoch; cập nhật branch lab03 trước khi chạy."
print("Code revision:")
subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.9.5"], check=True)
def uv(*args):
    subprocess.run([sys.executable, "-m", "uv", *map(str, args)], check=True)
uv("python", "install", "3.12")
if not Path(PYTHON).is_file():
    uv("venv", "--python", "3.12", ENV)
uv("pip", "install", "--python", PYTHON, "--link-mode", "copy", "-r", LAB / "requirements.txt")
uv("pip", "check", "--python", PYTHON)

RUN_ENV = os.environ.copy()
RUN_ENV.update(CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLCONFIGDIR="/kaggle/working/.mpl-lab03")
def run_python(*args, env=None):
    return subprocess.run([PYTHON, "-u", *map(str, args)], cwd=REPO,
                          env=RUN_ENV if env is None else env, check=True)
run_python("-c", "import sys, torch, transformers; print(sys.version); print('torch', torch.__version__); print('transformers', transformers.__version__); assert torch.cuda.is_available(), 'GPU chưa khả dụng'; print(torch.cuda.get_device_name(0))")
free_gib = shutil.disk_usage("/kaggle/working").free / 1024**3
print(f"Dung lượng trống: {free_gib:.1f} GiB")
if free_gib < 6:
    raise RuntimeError("Cần ít nhất 6 GiB trống cho candidate checkpoints và ZIP. Hãy dùng session sạch.")

# %%
from datetime import datetime, timezone
validation_dir = LAB / "outputs/preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
validation_dir.mkdir(parents=True, exist_ok=False)
TEST_ENV = RUN_ENV.copy()
TEST_ENV["CUDA_VISIBLE_DEVICES"] = ""
with (validation_dir / "pytest.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen([PYTHON, "-u", "-m", "pytest", "labs/lab03/tests", "-q",
                                f"--junitxml={validation_dir / 'pytest.xml'}"],
                               cwd=REPO, env=TEST_ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end=""); log.write(line)
    if process.wait() != 0:
        raise RuntimeError("Preflight tests failed. Chưa chạy training; xem pytest.log.")
print("Preflight passed:", validation_dir)

# %%
run_python("labs/lab03/scripts/run_exercise1.py")
run_python("-c", "from labs.lab03.src.analysis import ensure_exercise1_analysis; from labs.lab03.src.utils.runs import ROOT; import json; s=json.loads((ROOT/'config/workflow.json').read_text()); print(ensure_exercise1_analysis(ROOT/'outputs/runs'/s['active_runs']['exercise1']))")
state = json.loads((LAB / "config/workflow.json").read_text())
exercise1_run = LAB / "outputs/runs" / state["active_runs"]["exercise1"]
display(json.loads((exercise1_run / "predictions.json").read_text()))
exercise1_analysis = LAB / "outputs/runs" / state["active_runs"]["exercise1_analysis"]
for image_path in sorted(exercise1_analysis.glob("*.png")):
    display(Markdown(f"**{image_path.name}**"))
    display(Image(filename=str(image_path)))

# %%
config = json.loads((LAB / "config/config.json").read_text())
study_settings = json.loads((LAB / "config/study.json").read_text())
assert config["exercise2"]["num_train_epochs"] == 10
assert study_settings["epochs_per_candidate"] == 10
print(json.dumps(study_settings, indent=2))

# Chạy cell này đúng một lần cho mỗi thí nghiệm mới.
# Chạy lại sẽ tạo study và candidate mới, không ghi đè hoặc resume run cũ.
run_python("labs/lab03/scripts/run_study.py")
state = json.loads((LAB / "config/workflow.json").read_text())
STUDY = LAB / "outputs/runs" / state["active_runs"]["study"]
print("Study đã hoàn tất:", STUDY)

# %%
import importlib.util
spec = importlib.util.spec_from_file_location("lab03_display_study", LAB / "scripts/display_study.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)
shown_figures = viewer.display_study(STUDY, root=LAB)
assert len(shown_figures) >= 10, "Thiếu hình trực quan; kiểm tra artifacts trước khi đóng gói."

# %%
run_python("labs/lab03/scripts/audit_study.py", "--study-run", STUDY)
run_python("labs/lab03/scripts/package_study.py", "--study-run", STUDY)
bundle = json.loads((LAB / "outputs/bundles/latest_bundle.json").read_text())
for key in ["review", "checkpoint"]:
    if not bundle.get(key):
        continue
    source = Path(bundle[key])
    download = Path("/kaggle/working") / source.name
    shutil.copy2(source, download)
    print(key, ":", download, f"({download.stat().st_size / 1024**2:.1f} MB)")
    display(FileLink(str(download.relative_to(Path.cwd())) if download.is_relative_to(Path.cwd()) else str(download)))
print("SHA256 review:", bundle["review_sha256"])
print("Nếu panel Output chưa hiện ZIP: nhấn refresh cạnh /kaggle/working.")
