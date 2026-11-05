"""Execute a notebook in-place; used by README and delivery.complete_delivery."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import sys

ROOT=Path(__file__).resolve().parents[1]


def scheduled_notebook(nbformat):
    """Build a roadmap presentation from saved artifacts; never start training."""
    cells = []
    def section(markdown, code):
        cells.extend([nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)])
    cells.append(nbformat.v4.new_markdown_cell(
        '# Lab02: Modeling, evaluation và inference một ảnh\n\n'
        'Notebook xem artifact của run được chọn; không train hoặc benchmark lại khi Run All. '
        'Train học trọng số, validation chọn checkpoint, test đánh giá sau lựa chọn. '
        '[Báo cáo](../walkthrough_lab_02.md).\n\n'
        '## Bảng lộ trình triển khai (Pipeline Execution Roadmap)\n\n'
        '| Bước | Giai đoạn | Module / hàm | Output hiển thị |\n'
        '|---|---|---|---|\n'
        '| 1 | Setup và data contract | [notebook_views.context](../src/utils/notebook_views.py) | Run ID, config, split và tensor |\n'
        '| 2 | Model và freeze | [models](../src/models/models.py), architecture_visual | Kiến trúc, tham số và phần mở |\n'
        '| 3 | Training | [experiment.fit](../src/training/experiment.py), learning_curves | Epoch thực chạy, loss/accuracy |\n'
        '| 4 | Evaluation | quality_cost, errors, evidence, roc_pr | Metric, confusion, ROC/PR và ảnh dự đoán |\n'
        '| 5 | Selection | lr_search, fine_search | Trial đã đo; validation selection |\n'
        '| 6 | Training/test cost | efficiency, monitoring_status | Thời gian train và test theo batch |\n'
        '| 7 | Save/load | roundtrip | Bằng chứng kiểm tra sau load đã lưu |\n'
        '| 8 | Loaded-model inference | single_image_benchmark; [benchmark CLI](../scripts/benchmark_inference.py) | Latency một ảnh, RAM RSS và VRAM |\n'
        '| 8b | Training batch 1 | training_benchmark; benchmark_training.py | Training step, RAM và VRAM; khác inference |\n'
        '| 9 | Bàn giao | Artifact provenance | Run, nguồn bằng chứng và giới hạn |'))
    section('## 1. Setup và data contract\n\n'
        'Chọn run qua LAB02_RUN hoặc registry, không tự lấy run mới nhất. '
        'CIFAR-10: 45.000 train/5.000 validation/10.000 test; RGB uint8 HWC → float32 BCHW → 10 logits. '
        'Cấu hình dưới đây là của run thực tế, không phải giá trị mặc định cho mọi run.',
        "from pathlib import Path\nimport os, sys, json\nimport pandas as pd\ntry:\n    ROOT = Path(os.environ['LAB02_ROOT']).expanduser() if os.getenv('LAB02_ROOT') else next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/utils/notebook_views.py').is_file())\nexcept (OSError, StopIteration) as exc:\n    raise RuntimeError('Cannot resolve Lab02. Set LAB02_ROOT to an existing absolute Lab02 directory.') from exc\nif not ROOT.is_absolute() or not (ROOT / 'src/utils/notebook_views.py').is_file():\n    raise RuntimeError('Set LAB02_ROOT to an existing absolute Lab02 directory.')\nos.chdir(ROOT)\nsys.path.insert(0, str(ROOT))\nfrom src.utils import notebook_views as view\nROOT, RUN, OUT = view.context()\nfrom IPython import get_ipython; get_ipython().run_line_magic('matplotlib', 'inline')\nprint('Run:', RUN.name)\ndisplay(pd.Series(json.loads((RUN / 'config.json').read_text()), name='Recorded config'))")
    section('## 2. Kiến trúc, freeze và sanity\n\n'
        'ResNet18 dùng residual addition; DenseNet121 dùng nối feature maps để tái sử dụng đặc trưng. '
        'Hai kiểu kết nối giúp khảo sát transfer learning; số tham số ít hơn không bảo đảm nhanh hơn. '
        'Backbone ImageNet được thay classifier 10 lớp. Đọc phần mở và tỷ lệ trainable từ metadata của run. '
        'BatchNorm running statistics cố định; affine weights ở phần mở vẫn học. '
        'Sanity overfit thuộc pipeline training; biểu đồ này chỉ đọc artifact, không chạy lại sanity. '
        '[Thiết kế](../walkthrough_lab_02.md#24-thay-classifier-freeze-và-fine-tune).',
        'view.architecture_visual(RUN, OUT)')
    section('## 3. Training: phương pháp và đường học\n\n'
        'Head dùng pooled feature cache. Với scheduled recipe, fine-tune train dùng crop/flip, '
        'dropout và full-image forward; validation/test không augmentation. '
        'Checkpoint chọn val accuracy, hòa xét loss rồi epoch sớm hơn. Early stopping theo val loss. '
        'Đọc actual epochs và stop reason; không mặc định đủ ngân sách nghĩa là hội tụ. '
        'Train/validation có điều kiện augmentation/dropout khác nhau.',
        'view.epoch_budget(RUN, ROOT)\nview.learning_curves(RUN, OUT)')
    section('## 4. Đánh giá và so sánh model\n\n'
        '### 4.1. Metric tổng thể trên test\n\n'
        'Accuracy, macro precision/recall/F1 cao hơn tốt hơn; loss thấp hơn tốt hơn. '
        'Macro F1 là trung bình F1 từng lớp. Test không chọn cấu hình.',
        'view.quality_cost(RUN, OUT)')
    section('### 4.2. Metric theo lớp và confusion matrix\n\n'
        'P/R/F1 cao tốt; FPR thấp tốt. Recall chính là TPR. Bảng ngắn lấy hai lớp F1 thấp nhất/model; '
        'bảng dùng fraction, heatmap dùng %. Confusion: hàng thật, cột dự đoán; '
        'số trong ô là count, màu chuẩn hóa theo hàng.',
        'view.errors(RUN, OUT)\nview.evidence(RUN, "confusion_matrix")')
    section('### 4.3. ROC và Precision–Recall\n\n'
        'One-vs-rest qua nhiều ngưỡng: ROC X=FPR/Y=TPR; PR X=recall/Y=precision. '
        'ROC legend là AUC; PR legend là AP, không phải F1. Không đồng nhất với accuracy argmax.',
        'view.roc_pr(RUN)')
    section('### 4.4. Dự đoán đúng/sai\n\n'
        'Mỗi ảnh ghi nhãn thật, nhãn dự đoán và softmax probability. '
        'Lưới tối đa 8 đúng+8 sai không đại diện accuracy; confidence cao vẫn có thể sai.',
        'view.evidence(RUN, "prediction_examples")')
    section('## 5. Search và lựa chọn bằng validation\n\n'
        'Chỉ trình bày trial thực có. Scheduled recipe có một cấu hình/model, '
        'không phải grid search hay optimizer ablation. Candidate chưa chạy phải ghi rõ trạng thái.',
        'view.lr_search(RUN, OUT)\nview.fine_search(RUN, OUT, ROOT)')
    section('## 6. Thời gian huấn luyện và đánh giá test\n\n'
        'Tách extraction, search và selected recipe; không cộng lại selected trial hai lần. '
        'Test time là pass theo batch trên toàn tập test, không phải latency một ảnh. '
        'So sánh train time cần cùng thiết bị và cache policy. TensorBoard bên dưới là '
        'bằng chứng đã ghi, không xác nhận server đang chạy.',
        'view.efficiency(RUN, OUT)\nview.monitoring_status(RUN)')
    section('## 7. Save/load: bằng chứng sau tải\n\n'
        'Bảng đọc roundtrip.json đã lưu theo model/stage, không nạp lại checkpoint khi xem. '
        'Số mẫu và sai số là phạm vi kiểm chứng; roundtrip tám mẫu không phải full-test sau load.',
        'view.roundtrip(RUN)')
    section('## 8. Dự đoán một ảnh: processing time và RAM/VRAM\n\n'
        '**Model đã nạp sẵn**, batch 1, FP32. Đọc artifact benchmark đã thực thi, '
        'không chạy lại GPU. Đường dẫn benchmark chọn tường minh và phải khớp training run đang xem. '
        'Nếu chuyển máy, copy bộ evidence published cùng source.\n\n'
        '- Total: đọc/giải mã PNG → preprocessing → transfer → forward → softmax/CPU → JSON; loại load model.\n'
        '- Thời gian: mean/median/P95 sau 20 warm-up, 100 lượt; request đầu tách riêng.\n'
        '- RAM: RSS sau load, baseline trước memory pass, peak và tăng thêm; đơn vị MiB.\n'
        '- VRAM: allocated/reserved peak của PyTorch; không phải toàn bộ CUDA context.\n'
        '- Stage diagnostics và memory dùng lượt đo riêng. RSS lấy mẫu có thể bỏ lỡ đỉnh ngắn.\n\n'
        'Một ảnh lặp có OS cache; không suy ra accuracy, batch throughput hoặc phần cứng tối thiểu. '
        '[Báo cáo](../walkthrough_lab_02.md#39-benchmark-một-ảnh-khi-model-đã-nạp-sẵn).',
        "BENCHMARK = ROOT / 'outputs/published/single_image_benchmark/20261102T184034Z_kaggle'\ntry:\n    benchmark_tables = view.single_image_benchmark(RUN, BENCHMARK, OUT)\nexcept FileNotFoundError as exc:\n    print('Benchmark evidence unavailable. Copy the complete published bundle:', exc)")
    section('## 8b. Training trên đúng một mẫu\n\n'
        'Đọc evidence CPU batch 1; Run All không train hoặc benchmark lại. '
        'Forward → CrossEntropyLoss → backward → optimizer.step. Mẫu train và preprocessing cố định, '
        'không augmentation; dropout hoạt động, BatchNorm running statistics cố định. '
        'Full-image forward cho cả hai model; freeze theo checkpoint. Optimizer mới dùng LR đầu recipe, '
        'không resume optimizer/scheduler. Memory đo bước đầu, timing riêng sau 3 warmup với 10 lượt. '
        'RAM gồm runtime; CPU không có VRAM đo được. Không thay fit time hoặc inference T4. '
        '[Phương pháp](../walkthrough_lab_02.md#211-benchmark-training-batch-size-1).',
        "TRAINING_BENCHMARK = ROOT / 'outputs/published/training_benchmark/scheduled64_cpu_batch1'\ntry:\n    view.training_benchmark(RUN, TRAINING_BENCHMARK, OUT)\nexcept FileNotFoundError as exc:\n    print('Training benchmark unavailable; run scripts/benchmark_training.py explicitly:', exc)")
    section('## 9. Bàn giao và giới hạn\n\n'
        'Mỗi bước hiển thị output ngay bên dưới như roadmap Lab01. Run/config/source snapshot '
        'giữ điều kiện đã thực thi; biểu đồ trình bày ở outputs/review không sửa metric training. '
        'Benchmark GPU dùng checkpoint của run CPU; không ghép thời gian CPU/GPU thành một phép đo. '
        'Một seed và một ảnh benchmark chưa xác lập ưu thế phổ quát. '
        'Notebook benchmark gốc được giữ làm bằng chứng thực thi; kết quả sử dụng nằm tại mục 8.',
        "print('Modeling run:', RUN.name)\nprint('Presentation output:', OUT)\nprint('Benchmark evidence:', BENCHMARK)\nprint('Report:', ROOT / 'walkthrough_lab_02.md')")
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
    if notebook_name in ('02_lab02_modeling.ipynb', '02_lab02_modeling-kaggle.ipynb') and config.get('profile')=='cifar10_scheduled64':
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
