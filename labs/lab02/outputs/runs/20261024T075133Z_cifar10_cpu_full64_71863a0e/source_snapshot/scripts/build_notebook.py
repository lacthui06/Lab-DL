"""Build the teaching notebook without maintaining duplicated training code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
                  "outputs": [], "execution_count": None})


md("""# Lab 02 — Pre-trained networks với PyTorch

**Bài toán:** phân loại 10 lớp FashionMNIST bằng ResNet18 và DenseNet121 pre-trained ImageNet.

**Câu hỏi:** hai kiến trúc khác nhau thế nào; head-only và fine-tune block cuối cho kết quả/chi phí ra sao; LR và batch size ảnh hưởng thế nào?

| Bước | Giai đoạn | Nội dung | Module | Bằng chứng |
|---|---|---|---|---|
| 1 | Setup | Runtime, MODE, cấu hình | src.experiment | environment/config |
| 2 | Dữ liệu | Split stratified, RGB, preprocessing theo weights | src.data | indices, data_summary |
| 3 | Kiến trúc | Xem lớp và tham số, thay classifier | src.models | architecture, parameters |
| 4 | Huấn luyện | Cache features, tìm LR/batch, fine-tune block cuối | src.experiment | search, selection, checkpoints |
| 5 | TensorBoard | Loss, accuracy, LR, ảnh và cấu hình | SummaryWriter | event files |
| 6 | So sánh | Learning curves, validation và test metrics | src.reporting | CSV, figures |
| 7 | Phân tích lỗi | Confusion matrix và mẫu đúng/sai | src.reporting | probabilities, report |
| 8 | Save/load | Tải vào instance mới, so xác suất | src.experiment | roundtrip.json |
| 9 | Kết luận | Phạm vi thật, giới hạn và tái lập | src.reporting | report_vi.md, manifest |

`MODE='view'` đọc run đã chọn. `MODE='train'` tạo run mới ở bước 4; bước này chạy trọn suite và giữ artifact ở từng giai đoạn. Các mục sau trình bày bằng chứng đã lưu. Profile CPU là pilot trên dữ liệu thật, không đại diện benchmark đầy đủ.
""")
md("## 1. Setup")
code("""from pathlib import Path
import os, sys, json, inspect
import pandas as pd
from IPython.display import display, Markdown, Image

MODE = 'view'  # 'train' để chạy mới
PROFILE = 'cpu_pilot'  # 'full' cho 54000/6000/10000; nên dùng GPU
if MODE not in {'view', 'train'}:
    raise ValueError('MODE phải là view hoặc train')
candidates = [Path(os.environ.get('LAB02_ROOT', '.')).resolve(), *Path.cwd().resolve().parents]
ROOT = next((p for p in candidates if (p / 'src' / 'experiment.py').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Đặt LAB02_ROOT đến thư mục lab2 chứa src/.')
sys.path.insert(0, str(ROOT))
WORK = Path('/kaggle/working/lab2') if Path('/kaggle/working').exists() else ROOT
CONFIG = ROOT / 'config' / f'{PROFILE}.json'
cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
display(cfg)
""")
md("""## 2. Dữ liệu và tiền xử lý

Split train/val stratified từ 60.000 ảnh train; test chính thức giữ riêng. Chưa có indices gốc của Lab1 để xác nhận cùng split. Pilot chọn tập con cân bằng trong mỗi split, không lấy validation từ test.

Ảnh một kênh được lặp thành ba kênh, resize/crop/normalize theo `IMAGENET1K_V1`. Không dùng mean/std của Lab1 cho pre-trained ImageNet. Không augmentation trong comparison này để giữ đầu vào cố định cho cache feature và fine-tune.
""")
code("""from src.data import image_transform
for name in cfg['models']:
    print(name, image_transform(name))
""")
md("""## 3. Khám phá và điều chỉnh kiến trúc

ResNet18 thay `fc`; DenseNet121 thay `classifier`; cả hai có 10 logits. Head-only chỉ cập nhật lớp mới. Fine-tune mở `layer4` hoặc `denseblock4 + norm5`, giữ BatchNorm running statistics để tránh thay đổi ngầm các phần đóng băng.
""")
code("""from src.models import build_model, configure_trainable, parameter_counts
print(inspect.getsource(build_model))
print(inspect.getsource(configure_trainable))
for name in cfg['models']:
    demo = build_model(name, pretrained=False)  # khảo sát cấu trúc; train dùng pretrained=True
    for stage in ['head', 'finetune']:
        configure_trainable(demo, name, stage)
        print(name, stage, parameter_counts(demo))
    del demo
""")
md("""## 4. Head search và fine-tune

Candidate dùng cùng khởi tạo classifier và seed. Grid LR×batch chỉ chọn bằng validation; backbone cache ở eval mode nên tương đương forward cố định. Fine-tune tiếp tục từ head được chọn và có ngân sách bổ sung; comparison chưa cô lập tác động mở lớp ở cùng tổng epoch.

Cell dưới chạy mới chỉ ở MODE=train. Khi view, registry chọn run rõ ràng thay vì lấy thư mục mới nhất. Có thể đặt `LAB02_RUN` để xem một run khác.
""")
code("""if MODE == 'train':
    from src.experiment import run_suite
    RUN = run_suite(CONFIG, WORK)
else:
    override = os.environ.get('LAB02_RUN')
    if override:
        RUN = Path(override)
    else:
        registry = WORK / 'outputs' / 'selected_run.json'
        if not registry.exists():
            raise FileNotFoundError('Chưa có run. Chạy scripts/run.py hoặc đặt MODE=train.')
        RUN = Path(json.loads(registry.read_text(encoding='utf-8'))['run'])
status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
assert status['status'] == 'completed', status
print('Run:', RUN)
display(json.loads((RUN / 'environment.json').read_text(encoding='utf-8')))
display(json.loads((RUN / 'data_summary.json').read_text(encoding='utf-8')))
run_cfg = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
for name in run_cfg['models']:
    print('\\nArchitecture:', name)
    print((RUN / name / 'architecture.txt').read_text(encoding='utf-8'))
    display(pd.read_csv(RUN / name / 'parameters.csv').head(12))
    display(pd.read_csv(RUN / name / 'search_results.csv'))
    display(json.loads((RUN / name / 'selection.json').read_text(encoding='utf-8')))
""")
md("""## 5. TensorBoard

Mỗi candidate và lượt fine-tune ghi train/validation loss, accuracy, LR từng nhóm tham số và hparams. Ảnh đầu vào cùng kiến trúc nằm trong input_preview. Bật cell dưới để theo dõi giao diện tương tác; event files đã được tạo khi train.
""")
code("""events = list(RUN.rglob('events.out.tfevents.*'))
print('TensorBoard event files:', len(events))
print('Log directory:', RUN)
START_TENSORBOARD = False
if START_TENSORBOARD:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', f'--logdir "{RUN}"')
""")
md("## 6. Kết quả và chi phí")
code("""results = pd.read_csv(RUN / 'comparison.csv')
display(results)
display(Image(filename=str(RUN / 'comparison.png')))
for name in run_cfg['models']:
    display(Image(filename=str(RUN / name / 'learning_curves.png')))
    display(json.loads((RUN / name / 'head_selected' / 'training_summary.json').read_text(encoding='utf-8')))
""")
md("""## 7. Confusion matrix và dự đoán

Mọi ô confusion matrix có count; màu là tỷ lệ theo lớp thật. Lưới ảnh gồm tối đa 8 đúng và 8 sai để khảo sát lỗi, không đại diện accuracy toàn test. Softmax confidence có thể cao khi dự đoán sai.
""")
code("""for name in run_cfg['models']:
    for stage in ['head_selected', 'finetune']:
        folder = RUN / name / stage
        print(name, stage)
        display(pd.read_csv(folder / 'classification_report.csv', index_col=0))
        display(Image(filename=str(folder / 'confusion_matrix.png')))
        display(Image(filename=str(folder / 'prediction_examples.png')))
""")
md("""## 8. Save / Load model

Mỗi condition đã được load vào instance mới trước khi đánh giá test. Checkpoint chứa state_dict và metadata kiến trúc. Phép roundtrip so xác suất của hai instance trên cùng ảnh ở eval mode. Cell sau minh họa tải model đã chọn để dùng lại, không tải weights ImageNet lần nữa.
""")
code("""import torch
name = run_cfg['models'][0]
saved = torch.load(RUN / name / 'finetune' / 'best.pt', map_location='cpu', weights_only=True)
LOADED_MODEL = build_model(name, pretrained=False)
LOADED_MODEL.load_state_dict(saved['state_dict'])
LOADED_MODEL.eval()
for model_name in run_cfg['models']:
    for stage in ['head_selected', 'finetune']:
        check = json.loads((RUN / model_name / stage / 'roundtrip.json').read_text(encoding='utf-8'))
        assert check['passed'], check
        print(model_name, stage, check)
print('LOADED_MODEL:', name)
""")
md("## 9. Kết luận và bằng chứng")
code("""text = (RUN / 'report_vi.md').read_text(encoding='utf-8')
# Figures are displayed above with resolved paths; keep report text portable here.
text = '\\n'.join(line for line in text.splitlines() if not line.startswith('!['))
display(Markdown(text))
manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
print('Files recorded in manifest:', len(manifest))
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (Lab 02)",
            "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"}},
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell['id'] = f'lab02-{i:02d}'
dest = ROOT / "notebooks" / "lab_02_transfer_learning.ipynb"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(dest)
