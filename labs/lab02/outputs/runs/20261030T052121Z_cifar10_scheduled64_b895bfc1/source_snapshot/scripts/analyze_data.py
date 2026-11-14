"""Measure raw CIFAR-10 EDA without loading a neural network."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eda import build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Folder containing cifar-10-batches-py')
    args = parser.parse_args()
    print(build(ROOT, args.data_dir))
