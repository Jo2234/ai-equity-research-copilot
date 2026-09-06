from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("AIERC_DATA_DIR", tempfile.mkdtemp(prefix="aierc-tests-"))
os.environ["AIERC_LLM_PROVIDER"] = "local"
