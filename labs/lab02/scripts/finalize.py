"""CLI for the delivery workflow in src.delivery; see ../README.md."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.delivery import finalize_run

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Run directory; defaults to selected_run.json')
    finalize_run(parser.parse_args().run)
