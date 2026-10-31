"""Export metrics and figures from saved probabilities, without retraining."""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, log_loss)

from src.datasets.eda import CLASSES


def save_figure(fig, path, **kwargs):
    """Replace a PNG atomically, keeping open previews away from partial writes."""
    from io import BytesIO
    from pathlib import Path
    from uuid import uuid4
    path = Path(path)
    temporary = path.with_name(f'{path.stem}.{uuid4().hex}.tmp')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', **kwargs)
    try:
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_evaluation(folder, probabilities, labels, indices, data_dir):
    from torchvision.datasets import CIFAR10
    predictions = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(10), average="macro", zero_division=0)
    metrics = {"test_samples": len(labels), "accuracy": accuracy_score(labels, predictions),
               "macro_f1": f1, "macro_precision": precision, "macro_recall": recall,
               "loss": log_loss(labels, probabilities, labels=np.arange(10))}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(folder / "test_predictions.npz", probabilities=probabilities,
                        labels=labels, indices=indices)
    report = classification_report(labels, predictions, labels=np.arange(10),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / "classification_report.csv")
    counts = confusion_matrix(labels, predictions, labels=np.arange(10))
    normalized = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(folder / "confusion_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    color="white" if normalized[i, j] > .55 else "black", fontsize=8)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title=f"{folder.parent.name} / {folder.name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Fraction within actual class")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    raw = CIFAR10(str(data_dir), train=False)
    correct = np.flatnonzero(predictions == labels)[:8]
    wrong = np.flatnonzero(predictions != labels)[:8]
    chosen = np.concatenate([correct, wrong])
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    records = []
    for ax in axes.flat:
        ax.axis("off")
    for ax, j in zip(axes.flat, chosen):
        ax.imshow(raw.data[int(indices[j])])
        ax.set_title(f"True: {CLASSES[labels[j]]}\nPred: {CLASSES[predictions[j]]}\np={probabilities[j, predictions[j]]:.3f}",
                     color="green" if predictions[j] == labels[j] else "firebrick", fontsize=9)
        records.append({"index": int(indices[j]), "true": int(labels[j]),
                        "predicted": int(predictions[j]), "probability": float(probabilities[j, predictions[j]])})
    fig.suptitle("Up to 8 correct + 8 incorrect examples; selected grid does not represent accuracy")
    fig.tight_layout()
    fig.savefig(folder / "prediction_examples.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(records).to_csv(folder / "prediction_examples.csv", index=False)
    return metrics


def plot_learning_curves(run, output=None, show=False, per_model_folders=False, cfg=None):
    """Separate head/fine rows; show measured epochs, checkpoint and actual stop evidence."""
    from pathlib import Path
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    run = Path(run)
    output = Path(output or run)
    output.mkdir(parents=True, exist_ok=True)
    cfg = cfg if cfg is not None else json.loads((run/'config.json').read_text(encoding='utf-8'))
    paths = []
    for name in pd.read_csv(run/'comparison.csv').model.unique():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        notes = []
        for row, (stage, folder, key) in enumerate([
                ('HEAD ONLY', 'head_selected', 'head_epochs'),
                ('FINE-TUNE', 'finetune', 'finetune_epochs')]):
            history = pd.read_csv(run/name/folder/'history.csv')
            summary_path = run/name/folder/'training_summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            planned = cfg.get(key, 'not logged')
            reason = summary.get('stop_reason', 'configured budget completed' if len(history)==planned else 'not logged')
            notes.append(f'{stage}: {len(history)}/{planned} epochs; stop: {reason}')
            best_epoch = summary.get('best_epoch')
            for ax, metric in zip(axes[row], ['loss', 'accuracy']):
                ax.plot(history.epoch, history[f'train_{metric}'], '-o', color='#2563eb', label='Train')
                ax.plot(history.epoch, history[f'val_{metric}'], '--s', color='#d97706', label='Validation')
                if best_epoch in history.epoch.values:
                    value = history.loc[history.epoch==best_epoch, f'val_{metric}'].iloc[0]
                    ax.scatter([best_epoch], [value], marker='*', s=140, color='#111827',
                               zorder=5, label='Selected checkpoint')
                ax.set(title=f'{stage} | {metric.upper()} | {len(history)}/{planned} epochs',
                       xlabel='Epoch within this stage')
                ax.set_facecolor('#eff6ff' if row==0 else '#ecfdf5')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlim(.8, max(history.epoch)+.2)
                if metric=='accuracy': ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
        fig.suptitle(f'{name}: HEAD and FINE-TUNE are separate stages', fontsize=13)
        fig.text(.02,.015, '\n'.join(notes)+'\nConvergence is not established by the epoch count alone.', fontsize=8)
        fig.tight_layout(rect=(0,.09,1,.96))
        destination = run/name/'learning_curves.png' if per_model_folders else output/f'learning_curves_{name}.png'
        save_figure(fig, destination, dpi=130, bbox_inches='tight')
        if show:
            from IPython.display import display
            display(fig)
        plt.close(fig)
        paths.append(destination)
    return paths


def summarize_suite(run, cfg, selected):
    comparison = pd.read_csv(run / "comparison.csv")
    from src.utils.costs import cost_table, validation_recommendation
    cost_table(run, comparison).to_csv(run/'cost_comparison.csv', index=False)
    (run/'validation_recommendation.json').write_text(
        json.dumps(validation_recommendation(run,comparison,cfg), indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = comparison["model"] + "\n" + comparison["stage"]
    for ax, col, title in zip(axes, ["accuracy", "macro_f1"], ["Test accuracy", "Test Macro F1"]):
        bars = ax.bar(names, comparison[col], color=["#2563eb", "#0d9488"] * len(cfg["models"]))
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "comparison.png", dpi=150)
    plt.close(fig)
    plot_learning_curves(run, per_model_folders=True, cfg=cfg)
    table = ["| Model | Stage | Val acc | Test acc | Macro F1 | Trainable | Train (s) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in comparison.iterrows():
        table.append(f"| {row.model} | {row.stage} | {row.best_val_accuracy:.2%} | {row.accuracy:.2%} | {row.macro_f1:.4f} | {int(row.trainable):,} | {row.train_seconds:.1f} |")
    results = []
    for name in cfg["models"]:
        a = comparison[(comparison.model == name) & (comparison.stage == "head")].iloc[0]
        b = comparison[(comparison.model == name) & (comparison.stage == "finetune")].iloc[0]
        results.append(f"- {name}: fine-tune thay đổi test accuracy {(b.accuracy-a.accuracy)*100:+.2f} điểm phần trăm và Macro F1 {b.macro_f1-a.macro_f1:+.4f} so với head-only.")
    full = all(cfg[f"{s}_limit"] == 0 for s in ("train", "val", "test"))
    scope = "Toàn bộ split CIFAR-10; ngân sách epoch và độ phân giải được ghi rõ bên dưới." if full else "Đây là thực nghiệm pilot trên tập con cân bằng của CIFAR10 thật; chưa phải benchmark toàn bộ dữ liệu."
    policies=[]
    for name in cfg['models']:
        record=json.loads((run/name/'selection.json').read_text(encoding='utf-8'))
        chosen=record.get('finetune',{})
        policies.append(f"- {name}: {chosen.get('optimizer','adamw')}; depth={chosen.get('unfreeze','last_block')}; backbone_lr={chosen.get('backbone_lr',cfg['finetune_backbone_lr'])}; head_lr={chosen.get('head_lr',cfg['finetune_head_lr'])}; scheduler={chosen.get('scheduler','none')}")
    fine_policy='\n'.join(policies)
    augmentation = ('Fine-tune train dùng RandomCrop(32,padding=4,reflect), flip ngang p=0.5; '
                    'head cache và validation/test không augmentation.' if cfg.get('train_augmentation') else
                    'Không augmentation; preprocessing cố định.')
    report = f"""# Lab 02 — Transfer Learning trên CIFAR10

## 1. Mục tiêu
Khảo sát ResNet18 và DenseNet121 pre-trained ImageNet, thay classifier thành 10 lớp, so sánh đóng băng backbone và fine-tune các module được cấu hình. Điều chỉnh learning rate/batch size, theo dõi bằng TensorBoard, đánh giá và lưu/tải model.

## 2. Dữ liệu và phạm vi thực nghiệm
{scope}
Profile: `{cfg['profile']}`. Train/validation/test thực tế: {len(selected['train']):,}/{len(selected['val']):,}/{len(selected['test']):,}. Seed: {cfg['seed']}.
Split gốc 45.000/5.000 được tạo stratified; test giữ nguồn test chính thức. Khi chạy pilot, mỗi split được lấy tập con stratified riêng. Mỗi lớp có 4.500 train / 500 validation / 1.000 test khi dùng toàn bộ dữ liệu.
Ảnh RGB gốc 32×32 được giữ màu. Đầu vào thực nghiệm là {cfg["image_size"]}×{cfg["image_size"]}; normalization dùng mean/std của weights IMAGENET1K_V1. Với 224×224, dùng trọn transforms chính thức; với 64×64, resize trực tiếp để phù hợp ngân sách CPU, không phải preprocessing mặc định của weights. {augmentation} Dropout classifier p={cfg.get('dropout', 0.)}.

## 3. Phương pháp
Mỗi mô hình thay classifier bằng Linear 10 đầu ra. Head-only đóng băng backbone và cache feature train/val ở eval mode. Cache tương đương forward backbone cho đầu vào cố định; thời gian trích xuất được ghi riêng.
Head dùng LR {cfg['learning_rates']} và batch size {cfg['batch_sizes']}, {cfg['head_epochs']} epoch/candidate với {cfg.get('head_optimizer','adamw')}, weight_decay={cfg['weight_decay']}; scheduler={cfg.get('head_scheduler','none')}. Một tổ hợp là một recipe, không tìm nhiều cấu hình. Các candidate dùng cùng initial head và shuffle seed. Policy chọn candidate: {cfg.get('selection_policy', {'accuracy_tolerance':0,'loss_tolerance':0})}; accuracy band → validation loss band → fit time. Checkpoint trong mỗi trial chọn accuracy cao nhất, hòa chọn loss thấp hơn rồi epoch sớm hơn.
Fine-tune: mọi candidate khởi tạo lại từ cùng selected head; cấu hình optimizer/LR/depth/scheduler được chọn chỉ bằng validation. Ngân sách tối đa {cfg['finetune_epochs']} epoch; stopping policy={cfg.get('early_stopping', 'fixed epoch budget')}. Số epoch thực chạy và stop_reason nằm trong training_summary.json; không suy ra hội tụ từ ngân sách. BatchNorm running statistics giữ cố định. Depth và các LR thực chạy được ghi bên dưới; scheduled_tail của DenseNet mở block3 layer11–12,18–24, transition3, block4 layer1–4,13–16, norm5; ResNet mở layer4.1; cả hai mở classifier. Cosine_ratio giảm từng nhóm còn {cfg.get('scheduler_min_ratio',.1):.0%} LR đầu. Cache prefix chỉ hỗ trợ last block.
{fine_policy}
Toàn bộ fine-search: `finetune_search_results.csv`; xem `cost_comparison.csv` để phân biệt chi phí toàn search và selected recipe.

CrossEntropyLoss nhận logits. Checkpoint chọn theo validation. Test không tham gia search hoặc lựa chọn checkpoint; hai điều kiện head/fine-tune được xác định trước và đánh giá sau khi khóa lựa chọn. Mỗi condition được tải vào model instance mới trước đánh giá.

## 4. Kết quả đo được
{chr(10).join(table)}

{chr(10).join(results)}

![So sánh](comparison.png)

Train (s) chỉ là thời gian fit candidate được chọn hoặc fine-tune, chưa bao gồm feature extraction/search. `training_summary.json` của head_selected ghi riêng thời gian trích xuất và tổng search. Vì fine-tune tiếp tục từ head, chi phí toàn pipeline phải cộng các giai đoạn.

## 5. Phân tích và giới hạn
Kết quả chỉ đại diện cho seed và phạm vi dữ liệu đã ghi. Fine-tune có thêm epoch và được khởi tạo từ head đã học; chênh lệch phản ánh chiến lược hai giai đoạn, chưa cô lập tác động unfreeze ở cùng ngân sách tính toán.
Không mặc định pre-trained hoặc fine-tune phải tốt hơn. Tập nguồn ImageNet khác phân phối ảnh CIFAR-10 độ phân giải thấp. Phóng ảnh không bổ sung thông tin chi tiết mới.
Không so sánh trực tiếp với Lab1 vì FashionMNIST và CIFAR-10 là hai bài toán khác nhau. Cấu hình CPU 64×64 và GPU 224×224 cũng phải được báo cáo riêng; không suy diễn kết quả của cấu hình chưa chạy.

## 6. Bằng chứng và tái lập
`config.json`, `environment.json`, `source_hashes.json`, hai NPZ indices, `console.log`, `status.json` và SHA256 manifest ghi điều kiện chạy. Từng kiến trúc có architecture, parameters, preprocessing, search_results, selection, TensorBoard event và learning curves. Mỗi condition có best.pt, history, classification report, confusion counts, probabilities, ảnh đúng/sai và roundtrip.json.
Lưới ảnh chọn tối đa 8 đúng/8 sai để minh họa; xác suất softmax không bảo đảm dự đoán đúng. Số trong confusion matrix là count, màu chuẩn hóa theo lớp thật.

## 7. Tài liệu tham khảo
- [Torchvision models và weights](https://docs.pytorch.org/vision/main/models)
- [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
- [TensorBoard với PyTorch](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial)
"""
    (run / "report_vi.md").write_text(report, encoding="utf-8")


def write_final_report(run, cfg, selected, diagnostic_notes):
    """One report writer, called by delivery after validation; rebuild idempotently.

    summarize_suite owns the base narrative and comparison figures. Delivery
    supplies verified diagnostic notes; Markdown and HTML share this same content.
    """
    import base64
    import re
    import markdown

    summarize_suite(run,cfg,selected)
    diagnostic_notes=list(diagnostic_notes)
    diagnostic_notes.extend(['### Diễn giải chỉ số và thiết kế',
        'Accuracy là số ảnh dự đoán đúng chia tổng số ảnh. Macro F1 tính F1 riêng cho từng lớp rồi lấy trung bình không trọng số; vì vậy mỗi lớp đóng góp ngang nhau. Macro Precision/Recall và cross-entropy test được lưu trong metrics.json. ROC-AUC và average precision tính one-vs-rest từng lớp, không phải cùng một đại lượng.',
        'ResNet dùng residual connections để cộng tín hiệu qua các block; DenseNet nối các feature maps từ nhiều lớp trong dense block. Thực nghiệm thay head và giữ cấu trúc backbone gốc, không thiết kế lại CNN từ đầu.',
        f"Profile {cfg['profile']}: ngân sách tối đa {cfg['finetune_epochs']} epoch fine-tune, ảnh {cfg['image_size']}x{cfg['image_size']}, seed {cfg['seed']}. Kết quả chỉ đại diện cấu hình và ngân sách này; một seed chưa chứng minh ưu thế thống kê."])
    path=run/'report_vi.md'
    text=path.read_text(encoding='utf-8')
    text+='\n## 8. Khảo sát lỗi và hình bổ sung\n\n'+'\n\n'.join(diagnostic_notes)
    path.write_text(text,encoding='utf-8')

    def embed_image(match):
        content=base64.b64encode((run/match.group(2)).read_bytes()).decode('ascii')
        return f'<img alt="{match.group(1)}" src="data:image/png;base64,{content}">'

    embedded=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',embed_image,text)
    body=markdown.markdown(embedded,extensions=['tables','fenced_code'])
    html='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Lab 02</title>
<style>body{font:16px/1.65 system-ui;color:#172033;max-width:1120px;margin:40px auto;padding:0 24px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dbe2ef;padding:8px;text-align:left}
th{background:#edf3fc}img{max-width:100%;height:auto}h2{margin-top:32px;border-bottom:1px solid #dbe2ef}
code{background:#edf2f7;padding:2px 4px}</style><body>'''+body+'</body></html>'
    (run/'report_vi.html').write_text(html,encoding='utf-8')
