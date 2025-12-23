
# scripts/utils.py
import json
import psutil
import time
from pathlib import Path
from typing import List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def read_prompts(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def save_jsonl(path: str, rows: List[dict]):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_csv(path: str, headers: List[str], row: List):
    p = Path(path)
    exists = p.exists()
    with p.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(headers) + "\n")
        f.write(",".join(map(str, row)) + "\n")

def rss_mb() -> float:
    """Resident set size (MB) of current process."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.perf_counter()
