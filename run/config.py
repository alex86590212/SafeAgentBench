"""Dataset paths relative to SafeAgentBench repo root."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "safe_detailed": os.path.join(REPO_ROOT, "dataset", "safe_detailed_1009.jsonl"),
    "unsafe_detailed": os.path.join(REPO_ROOT, "dataset", "unsafe_detailed_1009.jsonl"),
    "abstract": os.path.join(REPO_ROOT, "dataset", "abstract_1009.jsonl"),
    "long_horizon": os.path.join(REPO_ROOT, "dataset", "long_horizon_1009.jsonl"),
}

# Paper counts (abstract file has 100 tasks -> 400 rows after L1–L4 expansion in load_dataset)
COUNTS = {
    "safe_detailed": 300,
    "unsafe_detailed": 300,
    "abstract": 100,   # raw tasks; expanded to 400 when using methods.utils.load_dataset
    "long_horizon": 50,
}
