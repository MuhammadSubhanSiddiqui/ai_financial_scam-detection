# Optional copy of helpers tailored for the project root.
# If you already have a helpers.py in the notebook root, you can remove this copy or keep it for app imports.

from pathlib import Path
from importlib import import_module

try:
    # Try to import the workspace-level helpers if available
    from ..helpers import get_distilbert_score, extract_suspicious_claims, verify_with_gemini
except Exception:
    # Fallback: attempt local import (safe if you copy the main helpers here)
    try:
        from helpers import get_distilbert_score, extract_suspicious_claims, verify_with_gemini
    except Exception:
        # No-op placeholder; the real helpers live in the notebook workspace.
        def get_distilbert_score(*a, **k):
            raise RuntimeError("helpers not installed. Copy helpers.py into this folder or adjust PYTHONPATH.")
        def extract_suspicious_claims(*a, **k):
            raise RuntimeError("helpers not installed. Copy helpers.py into this folder or adjust PYTHONPATH.")
        def verify_with_gemini(*a, **k):
            raise RuntimeError("helpers not installed. Copy helpers.py into this folder or adjust PYTHONPATH.")
