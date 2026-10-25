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

**Bài toán:** phân loại 10 lớp CIFAR-10 RGB bằng ResNet18 và DenseNet121 pre-trained ImageNet.

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

`MODE='view'` đọc run đã chọn. `MODE='train'` tạo run mới ở bước 4; bước này chạy trọn suite và giữ artifact ở từng giai đoạn. Profile cpu_full dùng toàn bộ dữ liệu, đầu vào 64×64 và ngân sách epoch CPU. Profile full dùng 224×224 cho GPU; không suy diễn kết quả của profile chưa chạy.
""")
md("## 1. Setup")
code("""from pathlib import Path
import os, sys, json, inspect
import pandas as pd
from IPython.display import display, Markdown, Image

MODE = 'view'  # 'train' để chạy mới
PROFILE = 'cpu_full'  # toàn bộ 45000/5000/10000, ảnh 64x64; 'full' dùng 224x224
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
print('Cấu hình mặc định cho lượt chạy mới; cấu hình run thực tế hiển thị ở bước 4:')
display(cfg)
""")
md("""## 2. Dữ liệu và tiền xử lý

Split train/val stratified 45.000/5.000 từ 50.000 ảnh train; test chính thức 10.000 giữ riêng. Lưu indices và kiểm tra không trùng train/validation. Notebook 01 trình bày EDA trên ảnh RGB gốc và bảng phân bố lớp.

Giữ RGB, normalize theo `IMAGENET1K_V1`. Profile CPU resize 64×64; profile GPU dùng đầy đủ transforms 224×224 của weights. Đây là hai điều kiện khác nhau. Không augmentation để giữ đầu vào cố định cho cache feature và fine-tune.
""")
code("""from src.data import image_transform
for name in cfg['models']:
    print(name, image_transform(name, cfg['image_size']))
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
if not RUN.is_absolute():
    RUN = ROOT / RUN
status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
assert status['status'] == 'completed', status
print('Run:', RUN)
display(json.loads((RUN / 'environment.json').read_text(encoding='utf-8')))
display(json.loads((RUN / 'data_summary.json').read_text(encoding='utf-8')))
run_cfg = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
assert run_cfg.get('dataset') == 'CIFAR10', 'Run cũ khác dữ liệu; chọn run CIFAR-10'
display(run_cfg)
if (RUN / 'resume_config.json').exists():
    print('Cấu hình tiếp tục tại ranh giới kiến trúc:')
    display(json.loads((RUN / 'resume_config.json').read_text(encoding='utf-8')))
for name in run_cfg['models']:
    print('\\nArchitecture:', name)
    print((RUN / name / 'architecture.txt').read_text(encoding='utf-8'))
    display(pd.read_csv(RUN / name / 'parameters.csv').head(12))
    display(pd.read_csv(RUN / name / 'search_results.csv'))
    display(json.loads((RUN / name / 'selection.json').read_text(encoding='utf-8')))
    if (RUN / name / 'prefix_equivalence.json').exists():
        print('Kiểm chứng cache prefix đóng băng trên ảnh thật:')
        display(json.loads((RUN / name / 'prefix_equivalence.json').read_text(encoding='utf-8')))
from src.experiment import pass_epoch
print('Vòng lặp train/validation được gọi bởi fit:')
print(inspect.getsource(pass_epoch))
from src.models import CachedTail
print('Tối ưu thực thi tương đương khi prefix đóng băng, BN cố định và không augmentation:')
print(inspect.getsource(CachedTail))
""")
md("""## 5. TensorBoard

Mỗi candidate và lượt fine-tune ghi train/validation loss, accuracy, LR từng nhóm tham số và hparams. Ảnh đầu vào cùng kiến trúc nằm trong input_preview. Bật cell dưới để theo dõi giao diện tương tác; event files đã được tạo khi train.
""")
code("""events = list(RUN.rglob('events.out.tfevents.*'))
print('TensorBoard event files:', len(events))
print('Log directory:', RUN)
if (RUN / 'tensorboard_evidence.png').exists():
    display(Image(filename=str(RUN / 'tensorboard_evidence.png')))
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
    display(Image(filename=str(RUN / name / 'hyperparameter_search.png')))
    display(json.loads((RUN / name / 'head_selected' / 'training_summary.json').read_text(encoding='utf-8')))
if (RUN / 'cost_comparison.csv').exists():
    display(pd.read_csv(RUN / 'cost_comparison.csv'))
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
        if (folder / 'roc_pr.png').exists():
            display(Image(filename=str(folder / 'roc_pr.png')))
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

# Companion EDA notebook: reads evidence from the explicitly selected CIFAR-10 run.
cells = []
md('''# Lab 02 — 01. CIFAR-10: EDA và kiến trúc

Khảo sát ảnh RGB gốc và điều kiện đầu vào trước khi đọc notebook modeling.

| Bước | Nội dung | Hàm/file thực thi | Khi train | Khi view |
|---|---|---|---|---|
| 1 | Chọn run và môi trường | config, registry | run_suite tạo run mới | Đọc run được chỉ định |
| 2 | Kiểm chứng split/lớp | prepare_indices | Lưu indices | Đọc NPZ, kiểm tra đối chiếu dataset |
| 3 | Ảnh và thống kê | export_eda | Tính trên train | Bảng RGB và ảnh train |
| 4 | Tiền xử lý | image_transform | Resize/normalize trong loader | Gọi transform trên ảnh thật |
| 5 | Kiến trúc và freeze | build_model, configure_trainable | Nạp pretrained và thay head | Khảo sát cấu trúc, counts và tham số |

EDA không fit scaler trên validation/test. Normalization mô hình dùng thống kê ImageNet của pretrained weights; thống kê CIFAR-10 dưới đây phục vụ mô tả dữ liệu.
''')
md('## 1. Run và môi trường')
code('''from pathlib import Path
import os, sys, json, inspect
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image, Markdown
candidates = [Path(os.environ.get('LAB02_ROOT', '.')).resolve(), *Path.cwd().resolve().parents]
ROOT = next(p for p in candidates if (p / 'src' / 'experiment.py').exists())
sys.path.insert(0, str(ROOT))
override = os.environ.get('LAB02_RUN')
RUN = Path(override) if override else Path(json.loads((ROOT / 'outputs' / 'selected_run.json').read_text(encoding='utf-8'))['run'])
if not RUN.is_absolute():
    RUN = ROOT / RUN
cfg = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
assert cfg.get('dataset') == 'CIFAR10', 'Hãy chọn run CIFAR-10'
assert json.loads((RUN / 'status.json').read_text())['status'] == 'completed'
print('Run:', RUN)
display(cfg)
display(json.loads((RUN / 'environment.json').read_text(encoding='utf-8')))
''')
md('## 2. Kiểm chứng dữ liệu và split')
code('''from torchvision.datasets import CIFAR10
from src.data import prepare_indices, image_transform, IndexedImages
raw = CIFAR10(str(ROOT / 'data'), train=True, download=False)
official_test = CIFAR10(str(ROOT / 'data'), train=False, download=False)
full = dict(np.load(RUN / 'split_indices.npz'))
selected = dict(np.load(RUN / 'selected_indices.npz'))
assert len(raw) == 50000 and len(official_test) == 10000
assert not np.intersect1d(full['train'], full['val']).size
np.testing.assert_array_equal(np.union1d(full['train'], full['val']), np.arange(50000))
assert [len(full[s]) for s in ['train', 'val', 'test']] == [45000, 5000, 10000]
display(pd.read_csv(RUN / 'eda' / 'class_counts.csv', index_col=0))
print('Original image shape:', raw.data.shape[1:], 'dtype:', raw.data.dtype)
print(inspect.getsource(prepare_indices))
''')
md('## 3. Ảnh train và thống kê RGB')
code('''summary = json.loads((RUN / 'eda' / 'summary.json').read_text(encoding='utf-8'))
display(pd.DataFrame({'channel': ['R', 'G', 'B'], 'mean_train': summary['train_mean_rgb'], 'std_train': summary['train_std_rgb']}))
display(Image(filename=str(RUN / 'eda' / 'training_examples.png')))
print(summary['normalization_policy'])
''')
md('''## 4. Kiểm chứng preprocessing trên ảnh thật

Giữ màu RGB. CPU dùng resize trực tiếp 64×64; GPU dùng trọn transforms chính thức 224×224. Resize không khôi phục chi tiết mới. Không augmentation để cache feature nhất quán với fine-tuning. Có thể mở rộng augmentation sau như một thí nghiệm riêng.
''')
code('''from src.models import SPECS
import matplotlib.pyplot as plt
print(inspect.getsource(image_transform))
index = int(selected['train'][0])
image, label = raw[index]
fig, axes = plt.subplots(1, 1 + len(cfg['models']), figsize=(11, 4))
axes[0].imshow(image)
axes[0].set_title(f'Original: {raw.classes[label]} (32x32)')
for ax, name in zip(axes[1:], cfg['models']):
    transform = image_transform(name, cfg['image_size'])
    processed = transform(image)
    dataset = IndexedImages(ROOT / 'data', 'train', [index], name, cfg['image_size'])
    loaded, actual, original_index = dataset[0]
    torch.testing.assert_close(processed, loaded)
    assert actual == label and original_index == index
    spec = SPECS[name][1].transforms()
    restored = processed * torch.tensor(spec.std)[:, None, None] + torch.tensor(spec.mean)[:, None, None]
    ax.imshow(restored.permute(1, 2, 0).clamp(0, 1))
    ax.set_title(f'{name}: {tuple(processed.shape)}')
    print(name, transform)
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
''')
md('''## 5. Kiến trúc và tham số được học

Training nạp ImageNet `IMAGENET1K_V1`. Cell này dựng cùng kiến trúc với `pretrained=False` để khảo sát mà không tải weights; kiến trúc training thực tế đã được lưu trong artifact. Mỗi classifier xuất 10 logits, CrossEntropyLoss nhận logits trực tiếp.
''')
code('''from src.models import build_model, configure_trainable, parameter_counts
print(inspect.getsource(build_model))
print(inspect.getsource(configure_trainable))
rows = []
for name in cfg['models']:
    print((RUN / name / 'architecture.txt').read_text(encoding='utf-8'))
    model = build_model(name, pretrained=False)
    for stage in ['head', 'finetune']:
        configure_trainable(model, name, stage)
        rows.append({'model': name, 'stage': stage, **parameter_counts(model)})
        print(name, stage, 'trainable layers:')
        print([n for n, p in model.named_parameters() if p.requires_grad])
    display(pd.read_csv(RUN / name / 'parameters.csv'))
    del model
display(pd.DataFrame(rows))
''')
md('''Tiếp tục với `lab_02_transfer_learning.ipynb` để xem LR/batch search, fine-tuning, TensorBoard, đánh giá và save/load. Kết quả CPU 64×64 áp dụng cho chính cấu hình đó; cấu hình GPU 224×224 được cung cấp để chạy lại.
''')
companion = {**notebook, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'cifar10-eda-{i:02d}'
eda_dest = ROOT / 'notebooks' / '01_cifar10_eda_architecture.ipynb'
eda_dest.write_text(json.dumps(companion, ensure_ascii=False, indent=1), encoding='utf-8')
print(eda_dest)
