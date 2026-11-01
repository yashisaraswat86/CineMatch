from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / 'artifacts'
TOP_N = 12
# Initial weights; use scripts/tune_weights.py after offline evaluation to tune these.
WEIGHTS = {'svd': 0.35, 'item_cf': 0.25, 'user_cf': 0.15, 'content': 0.25}
