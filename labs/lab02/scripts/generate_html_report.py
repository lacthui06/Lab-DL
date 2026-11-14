"""Dựng HTML Lab02 từ artifact đã lưu, không chạy training."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.html_report import generate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Run cụ thể; mặc định outputs/selected_run.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/walkthrough_lab_02.html')
    args = parser.parse_args()
    print(generate_report(ROOT, args.output, args.run))


if __name__ == '__main__':
    main()
