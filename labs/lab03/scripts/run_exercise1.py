from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.workflow import exercise1
if __name__ == "__main__":
    out = exercise1()
    from labs.lab03.src.analysis import ensure_exercise1_analysis
    print("Exercise 1 run:", out)
    print("Visual analysis:", ensure_exercise1_analysis(out))
