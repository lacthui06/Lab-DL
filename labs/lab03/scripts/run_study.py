"""Điểm vào CLI cho thí nghiệm GPU 10 epoch mỗi cấu hình."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.study import run_study
from labs.lab03.src.workflow import read_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--settings', type=Path)
    args = parser.parse_args()
    import json
    settings = json.loads(args.settings.read_text()) if args.settings else None
    print('Completed study:', run_study(read_config(args.config), settings))
