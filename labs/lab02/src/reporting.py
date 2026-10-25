"""Export metrics and figures from saved probabilities, without retraining."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, log_loss)
from torchvision.datasets import CIFAR10

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def export_evaluation(folder, probabilities, labels, indices, data_dir):
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


def summarize_suite(run, cfg, selected):
    comparison = pd.read_csv(run / "comparison.csv")
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
    for name in cfg["models"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for stage, sub in [("head", "head_selected"), ("finetune", "finetune")]:
            frame = pd.read_csv(run / name / sub / "history.csv")
            for ax, metric in zip(axes.flat, ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]):
                ax.plot(frame.epoch, frame[metric], marker="o", label=stage)
                ax.set(title=metric, xlabel="Epoch")
                ax.grid(alpha=.2)
                ax.legend()
        fig.suptitle(f"{name}: fine-tune starts from selected head checkpoint")
        fig.tight_layout()
        fig.savefig(run / name / "learning_curves.png", dpi=150)
        plt.close(fig)
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
    report = f"""# Lab 02 — Transfer Learning trên CIFAR10

## 1. Mục tiêu
Khảo sát ResNet18 và DenseNet121 pre-trained ImageNet, thay classifier thành 10 lớp, so sánh đóng băng backbone và fine-tune block cuối. Điều chỉnh learning rate/batch size, theo dõi bằng TensorBoard, đánh giá và lưu/tải model.

## 2. Dữ liệu và phạm vi thực nghiệm
{scope}
Profile: `{cfg['profile']}`. Train/validation/test thực tế: {len(selected['train']):,}/{len(selected['val']):,}/{len(selected['test']):,}. Seed: {cfg['seed']}.
Split gốc 45.000/5.000 được tạo stratified; test giữ nguồn test chính thức. Khi chạy pilot, mỗi split được lấy tập con stratified riêng. Mỗi lớp có 4.500 train / 500 validation / 1.000 test khi dùng toàn bộ dữ liệu.
Ảnh RGB gốc 32×32 được giữ màu. Đầu vào thực nghiệm là {cfg["image_size"]}×{cfg["image_size"]}; normalization dùng mean/std của weights IMAGENET1K_V1. Với 224×224, dùng trọn transforms chính thức; với 64×64, resize trực tiếp để phù hợp ngân sách CPU, không phải preprocessing mặc định của weights. Không augmentation trong thí nghiệm này để tái sử dụng feature đóng băng và giữ tiền xử lý nhất quán.

## 3. Phương pháp
Mỗi mô hình thay classifier bằng Linear 10 đầu ra. Head-only đóng băng backbone và cache feature train/val ở eval mode. Cache tương đương forward backbone cho đầu vào cố định; thời gian trích xuất được ghi riêng.
Tìm learning rate {cfg['learning_rates']} và batch size {cfg['batch_sizes']}, {cfg['head_epochs']} epoch/candidate với AdamW, weight_decay={cfg['weight_decay']}. Các candidate dùng cùng initial head và shuffle seed. Chọn bằng best validation accuracy; hòa chọn trial sớm nhất.
Fine-tune bắt đầu từ head đã chọn; mở layer4 của ResNet18 hoặc denseblock4+norm5 của DenseNet121. Backbone LR={cfg['finetune_backbone_lr']}, head LR={cfg['finetune_head_lr']}, {cfg['finetune_epochs']} epoch. BatchNorm giữ running statistics; các affine weights trong block được mở vẫn có gradient.
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
