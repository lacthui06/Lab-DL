# Copy mỗi block # %% vào một cell Code Kaggle, đúng 10 cell. Bật GPU + Internet.

# %%
import os
import sys
import json
import shutil
import subprocess
import importlib.util
from pathlib import Path
from IPython.display import display, Image, Markdown, FileLink

REPO_URL = "https://github.com/lacthui06/Lab-DL.git"
BRANCH = "lab03"
REPO = Path("/kaggle/working/Lab-DL")
LAB = REPO / "labs/lab03"
ENV = Path("/kaggle/working/lab03-py312")
PYTHON = str(ENV / "bin/python")

assert shutil.which("nvidia-smi"), "Hãy bật Settings > Accelerator > GPU."
subprocess.run(["nvidia-smi"], check=True)
if not REPO.exists():
    subprocess.run(["git", "clone", "--single-branch", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
else:
    print("Giữ repo và các kết quả hiện có.")
assert (LAB / "scripts/preview_dataset.py").is_file(), "Hãy cập nhật code lab03 phiên bản 10 cell."
subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.9.5"], check=True)
def uv(*args):
    subprocess.run([sys.executable, "-m", "uv", *map(str,args)], check=True)
uv("python", "install", "3.12")
if not Path(PYTHON).is_file(): uv("venv", "--python", "3.12", ENV)
uv("pip", "install", "--python", PYTHON, "--link-mode", "copy", "-r", LAB / "requirements.txt")
uv("pip", "check", "--python", PYTHON)

RUN_ENV = os.environ.copy()
RUN_ENV.update(CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLCONFIGDIR="/kaggle/working/.mpl-lab03")
def run_python(*args, env=None):
    return subprocess.run([PYTHON,"-u",*map(str,args)], cwd=REPO,
                          env=RUN_ENV if env is None else env, check=True)
def selected_folder(kind):
    state=json.loads((LAB / "config/workflow.json").read_text())
    folder=(LAB / "outputs/runs" / state["active_runs"][kind]).resolve()
    assert folder.parent==(LAB / "outputs/runs").resolve(), "Run path không hợp lệ."
    return folder

def show_figures(folder, names=None):
    folder=Path(folder)
    paths=[folder/name for name in names] if names else sorted(folder.glob("*.png"))
    assert paths, f"Không có PNG trong {folder}"
    for path in paths:
        assert path.is_file(), f"Thiếu ảnh {path}"
        display(Markdown(f"**{path.name}**"))
        display(Image(filename=str(path)))

run_python("-c", "import sys,torch,transformers; print(sys.version); print(torch.__version__,transformers.__version__); assert torch.cuda.is_available(), 'GPU chưa khả dụng'; print(torch.cuda.get_device_name(0))")
free_gib=shutil.disk_usage("/kaggle/working").free/1024**3
print(f"Dung lượng trống: {free_gib:.1f} GiB")
assert free_gib>=6, "Cần ít nhất 6 GiB trống; hãy dùng session sạch."

# %%
import pandas as pd
CONFIG_FILE = LAB / "config/augmented.json"
STUDY_FILE = LAB / "config/augmented_study.json"
config=json.loads(CONFIG_FILE.read_text())
study_settings=json.loads(STUDY_FILE.read_text())
assert config["exercise2"]["num_train_epochs"]==10
assert study_settings["epochs_per_candidate"]==10
assert config["exercise2"]["eval_strategy"]=="epoch"
display(pd.DataFrame([{"seed":config["seed"], "backbone":config["exercise2"]["model_name"],
    "epochs_each":10, "learning_rates":str(study_settings["learning_rates"]),
    "batch_size":config["exercise2"]["batch_size"], "max_length":config["exercise2"]["max_length"],
    "weight_decay":config["exercise2"]["weight_decay"]}]))
display(config["exercise2"]["model_overrides"])
display(config["exercise2"]["augmentation"])

# %%
from datetime import datetime, timezone
validation_dir=LAB / "outputs/preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
validation_dir.mkdir(parents=True,exist_ok=False)
TEST_ENV=RUN_ENV.copy(); TEST_ENV["CUDA_VISIBLE_DEVICES"]=""
with (validation_dir/"pytest.log").open("w",encoding="utf-8") as log:
    process=subprocess.Popen([PYTHON,"-u","-m","pytest","labs/lab03/tests","-q",
        f"--junitxml={validation_dir/'pytest.xml'}"],cwd=REPO,env=TEST_ENV,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end="");log.write(line)
    assert process.wait()==0, "Preflight failed; xem pytest.log, chưa chạy training."
print("Preflight PASSED:",validation_dir)

# %%
run_python("labs/lab03/scripts/run_exercise1.py")
EXERCISE1=selected_folder("exercise1")
EXERCISE1_ANALYSIS=selected_folder("exercise1_analysis")
display(pd.DataFrame(json.loads((EXERCISE1/"predictions.json").read_text())["manual"]))
display(json.loads((EXERCISE1/"tokenizer_breakdown.json").read_text()))
show_figures(EXERCISE1_ANALYSIS)

# %%
run_python("labs/lab03/scripts/preview_dataset.py","--config",CONFIG_FILE)
EDA=selected_folder("dataset_preview")
eda=json.loads((EDA/"data_preview.json").read_text())
display(pd.DataFrame([{"split":k,"samples":v,"negative":eda["label_counts"][k]["0"],
                      "positive":eda["label_counts"][k]["1"]} for k,v in eda["split_sizes"].items()]))
display(pd.DataFrame([eda["train_token_lengths"]]))
show_figures(EDA)

# %%
run_python("labs/lab03/scripts/preview_augmentation.py","--config",CONFIG_FILE)
PREVIEW=selected_folder("augmentation_preview")
augmentation=json.loads((PREVIEW/"augmentation.json").read_text())
display(pd.DataFrame([{"official_train":augmentation["original_train_size"],
    "synthetic_added":augmentation["added_count"], "effective_train":augmentation["effective_train_size"],
    "validation":eda["split_sizes"]["validation"], "test":eda["split_sizes"]["test"]}]))
display(pd.DataFrame(augmentation["records"]).head(8))
show_figures(PREVIEW)

# %%
run_python("labs/lab03/scripts/run_study.py","--config",CONFIG_FILE,"--settings",STUDY_FILE)
STUDY=selected_folder("study")
candidates=json.loads((STUDY/"candidate_comparison.json").read_text())
for candidate in candidates:
    folder=LAB / "outputs/runs" / candidate["run_id"]
    display(Markdown(f"### LR={candidate['learning_rate']:g} — đủ 10 epoch"))
    epoch_rows=json.loads((folder/"epoch_history.json").read_text())
    assert [r["epoch"] for r in epoch_rows]==list(range(1,11))
    display(pd.DataFrame(epoch_rows))
    show_figures(folder,["learning_curves.png"])
print("Study:",STUDY)

# %%
selection=json.loads((STUDY/"frozen_selection.json").read_text())
policy=json.loads((STUDY/"decision_threshold.json").read_text())
display(pd.DataFrame(candidates))
display(selection)
display(pd.DataFrame([{"policy":"Reference 0.5",**policy["baseline_metrics"]},
                      {"policy":"Validation selected",**policy["selected_metrics"]}]))
WINNER=LAB / "outputs/runs" / selection["winner"]["run_id"]
show_figures(STUDY,["candidate_validation_comparison.png"])
show_figures(WINNER,["validation_threshold/threshold_curves.png"])

# %%
display(pd.DataFrame(json.loads((STUDY/"test_comparison.json").read_text())))
display(pd.DataFrame([json.loads((STUDY/"test_runtime_metrics.json").read_text())]))
display(json.loads((STUDY/"probability_diagnostics.json").read_text()))
show_figures(STUDY,["baseline_model_comparison.png","confusion_matrix.png",
    "test_threshold/confusion_matrix.png","metrics_bar.png","roc_pr_curves.png",
    "calibration_confidence.png","sample_predictions.png","inference_predictions.png"])
display(pd.DataFrame(json.loads((STUDY/"sample_predictions.json").read_text())))
display(pd.DataFrame(json.loads((STUDY/"inference_predictions.json").read_text())))

# %%
run_python("labs/lab03/scripts/audit_study.py","--study-run",STUDY)
run_python("labs/lab03/scripts/package_study.py","--study-run",STUDY)
bundle=json.loads((LAB / "outputs/bundles/latest_bundle.json").read_text())
for kind in ["review","checkpoint"]:
    if not bundle.get(kind):continue
    source=Path(bundle[kind]);download=Path("/kaggle/working")/source.name
    shutil.copy2(source,download)
    print(kind,":",download,f"({download.stat().st_size/1024**2:.1f} MB)")
    display(FileLink(str(download.relative_to(Path.cwd())) if download.is_relative_to(Path.cwd()) else str(download)))
print("SHA256 review:",bundle["review_sha256"])
print("Tải review ZIP để kiểm chứng/report; giữ riêng checkpoint ZIP.")
