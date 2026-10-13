import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

collect_ignore_glob = ["scripts/*", "web/*"]
