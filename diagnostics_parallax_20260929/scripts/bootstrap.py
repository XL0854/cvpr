"""Paths and cache isolation for the parallax diagnostic."""
import os
import sys
from pathlib import Path

DIAG = Path(__file__).resolve().parents[1]
ROOT = DIAG.parent
sys.dont_write_bytecode = True
for key, relative in {
    "TORCH_HOME": "cache/torch",
    "XDG_CACHE_HOME": "cache",
    "MPLCONFIGDIR": "cache/matplotlib",
    "TRITON_CACHE_DIR": "cache/triton",
    "CUDA_CACHE_PATH": "cache/cuda",
}.items():
    os.environ[key] = str(DIAG / relative)
sys.path.insert(0, str(ROOT / "wCoefNet" / "Codes"))
sys.path.insert(0, str(DIAG / "scripts"))

