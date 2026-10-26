"""Xuất hình/bảng từ kết quả đã lưu, không tải model hay train lại."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import selected_run
from labs.lab03.src.analysis import ensure_analysis, ensure_exercise1_analysis
from labs.lab03.src.threshold_analysis import ensure_threshold_analysis
if __name__ == "__main__":
    print("Exercise 1 analysis:", ensure_exercise1_analysis(selected_run("exercise1")))
    print("Fine-tuning analysis:", ensure_analysis(selected_run("finetuning")))
    print("Threshold/checkpoint comparison:", ensure_threshold_analysis(selected_run("finetuning")))
