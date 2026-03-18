from pathlib import Path
import sys

import matplotlib

# Force a non-interactive backend for CI/headless test runs.
matplotlib.use("Agg")

# Ensure project root is on sys.path so `src` imports resolve under pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
