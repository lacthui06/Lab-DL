"""CLI for the delivery workflow in src.delivery; see ../README.md."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.delivery import finalize_run, complete_delivery

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Run directory; defaults to selected_run.json')
    parser.add_argument('--all', action='store_true', help='Also wait, run tests/notebooks and check recommended-model inference')
    args=parser.parse_args()
    if args.all:
        complete_delivery(args.run)
    else:
        finalize_run(args.run)
