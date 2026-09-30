"""Keep all writes/cache inside the diagnostic workspace; source imports are read-only."""
import os
import sys
from pathlib import Path

DIAG = Path(__file__).resolve().parents[1]
ROOT = DIAG.parent
sys.dont_write_bytecode = True
for key, relative in {
    'TORCH_HOME': 'cache/torch', 'XDG_CACHE_HOME': 'cache',
    'HF_HOME': 'cache/huggingface', 'MPLCONFIGDIR': 'cache/matplotlib',
    'TRITON_CACHE_DIR': 'cache/triton', 'CUDA_CACHE_PATH': 'cache/cuda',
}.items():
    os.environ[key] = str(DIAG / relative)
os.environ['WANDB_MODE'] = 'disabled'
for path in [ROOT/'wCoefNet/Codes', DIAG/'third_party/RoMa', DIAG/'third_party/python_deps']:
    sys.path.insert(0, str(path))

