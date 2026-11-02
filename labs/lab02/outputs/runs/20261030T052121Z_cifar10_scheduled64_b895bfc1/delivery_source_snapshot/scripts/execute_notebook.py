"""Execute a notebook in-place; used by README and delivery.complete_delivery."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import sys

ROOT=Path(__file__).resolve().parents[1]


def scheduled_notebook(nbformat):
    """Build a run-driven presentation, without prose or metrics from an older run."""
    cells = []
    def section(markdown, code):
        cells.extend([nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)])
    section('# Lab02 — Scheduled transfer learning\n\nKết quả đọc từ run được chọn; không train lại khi thực thi notebook. Train học trọng số, validation chọn checkpoint, test đánh giá sau selection.',
        'from pathlib import Path\nimport sys, json\nimport pandas as pd\nsys.path.insert(0, str(Path.cwd()))\nfrom src.utils import notebook_views as view\nROOT, RUN, OUT = view.context()\nprint("Run:", RUN.name)\ndisplay(pd.Series(json.loads((RUN / "config.json").read_text()), name="Config"))')
    section('## 1. Kiến trúc và freeze\n\nResNet: layer4.1 + fc. DenseNet: block3 layer11–12,18–24; transition3; block4 layer1–4,13–16; norm5; classifier. Trainable <=50% mỗi model. BatchNorm running statistics cố định; weight/bias ở phần mở vẫn học. Dropout 0.2 bật khi train, tắt khi eval.',
        'view.architecture_visual(RUN, OUT)')
    section('## 2. Lịch huấn luyện\n\nMột recipe/model, batch 128 cố định; không grid search. Head 12 epoch, LR 1e-3. Fine-tune tối đa 20 epoch, backbone LR 1e-4 và head LR 1e-3. Cosine giảm từng nhóm còn 10%; LR cuối lịch đạt sau bước scheduler cuối. Early stopping loss: min_epochs=10, patience=7, min_delta=1e-4. Checkpoint chọn theo val accuracy, hòa xét loss.\n\nHead dùng feature cache không augmentation; fine-tune train crop32/padding reflect4 và flip0.5 trước resize64/Normalize ImageNet. Validation/test không augmentation.',
        'for name in pd.read_csv(RUN / "comparison.csv").model.unique():\n    print(name)\n    display(pd.read_csv(RUN / name / "search_results.csv"))\n    display(pd.read_csv(RUN / name / "finetune_search_results.csv"))')
    section('## 3. Learning curves\n\nTrain và validation có điều kiện dropout/augmentation khác nhau. Xem số epoch thực chạy và stop_reason; không mặc định đã hội tụ.',
        'view.epoch_budget(RUN, ROOT)\nview.learning_curves(RUN, OUT)')
    section('## 4. Kết quả TEST\n\nAccuracy, macro precision/recall/F1 càng cao càng tốt; loss càng thấp càng tốt. Macro F1 là trung bình F1 từng lớp. Không dùng test chọn cấu hình.',
        'view.quality_cost(RUN, OUT)')
    section('## 5. Phân tích lỗi TEST — fine-tune\n\nP/R/F1 cao tốt; FPR thấp tốt. Bảng ngắn lấy 2 lớp F1 thấp nhất/model; số trong bảng là tỷ lệ 0–1, heatmap dùng %. Confusion matrix: hàng thật, cột dự đoán. Lưới ảnh minh họa không đại diện accuracy.',
        'view.errors(RUN, OUT)\nview.evidence(RUN, "confusion_matrix")\nview.evidence(RUN, "prediction_examples")')
    section('## 6. ROC / PR — TEST\n\nOne-vs-rest qua nhiều ngưỡng. ROC AUC và PR không phải accuracy; khác metric argmax.',
        'view.roc_pr(RUN)')
    section('## 7. Chi phí và artifact\n\nMở xen kẽ DenseNet vẫn cần backward về block3 layer11. Tỷ lệ tham số không trực tiếp quyết định chi phí. Kết quả chỉ đại diện seed/run này; chưa cô lập riêng tác động freeze và scheduler.',
        'view.efficiency(RUN, OUT)\nview.monitoring_status(RUN)')
    return nbformat.v4.new_notebook(cells=cells)


def execute(notebook_name):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    path=ROOT/'notebooks'/notebook_name
    selected = Path(os.environ.get('LAB02_RUN') or
                    json.loads((ROOT/'outputs/selected_run.json').read_text())['run'])
    selected = selected if selected.is_absolute() else ROOT/selected
    content = path.read_bytes()
    # Keep every overwritten notebook version; content hash makes snapshots immutable.
    backup = selected/'notebook_snapshots'/hashlib.sha256(content).hexdigest()/path.name
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():
        backup.write_bytes(content)
    notebook=nbformat.read(path,as_version=4)
    config=json.loads((selected/'config.json').read_text())
    if notebook_name=='02_lab02_modeling.ipynb' and config.get('profile')=='cifar10_scheduled64':
        notebook=scheduled_notebook(nbformat)
    kernels=ROOT/'.jupyter/kernels'
    spec=kernels/'lab02'
    spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Lab 02',language='python')),encoding='utf-8')
    manager=KernelManager(kernel_name='lab02',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client=NotebookClient(notebook,timeout=600,resources={'metadata':{'path':str(ROOT)}},kernel_name='lab02',km=manager)
    client.extra_arguments=['--HistoryManager.enabled=False']
    client.execute()
    nbformat.write(notebook,path)
    completed=path.read_bytes()
    archived=selected/'notebook_snapshots'/hashlib.sha256(completed).hexdigest()/path.name
    archived.parent.mkdir(parents=True,exist_ok=True)
    if not archived.exists():
        archived.write_bytes(completed)
    print('Executed successfully:',path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',nargs='?',default='02_lab02_modeling.ipynb')
    execute(parser.parse_args(argv).notebook)


if __name__=='__main__':
    main()
