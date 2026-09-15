"""Backend test configuration — ensures lunar_core is importable."""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend/ to PYTHONPATH so `from lunar_core.io...` resolves
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
