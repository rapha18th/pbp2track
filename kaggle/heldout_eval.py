"""Kaggle script: the held-out evaluation.

Notebook settings: add the build_library notebook's output as an input. Internet on for
the pip install, or bundle this file with tools/bundle_kernel.py for an offline notebook.
Output: /kaggle/working/eval.json and the table in the log.
"""
import glob
import json
import os
import subprocess
import sys

try:
    import pbp2track  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/rapha18th/pbp2track"], check=True)

from pbp2track.evaluate import heldout, table  # noqa: E402

root = os.path.dirname(glob.glob("/kaggle/input/**/possessions.parquet", recursive=True)[0])
summary = heldout(root, n=40, seed=7, include=["0021500438"], max_clips=90_000, log=lambda s: print(s, flush=True))
print(table(summary))
with open("/kaggle/working/eval.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
