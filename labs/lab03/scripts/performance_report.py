"""Xuất performance từ artifact đã có; không tải model hoặc train."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.performance import ensure_performance_report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    print(ensure_performance_report(args.run))
