"""Make the scripts under each portfolio's src/ importable from the tests.

The ML scripts create an ``imagens/`` folder at import time, relative to the
current directory, so they are imported once from a throwaway directory.
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in ("portfolio-2-busca-otimizacao", "portfolio-3-modelos-probabilisticos", "portfolio-4-machine-learning"):
    sys.path.insert(0, str(ROOT / folder / "src"))

_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp())
try:
    for name in ("projeto_01_supervisionado", "projeto_03_reforco", "projeto_04_deep_learning"):
        importlib.import_module(name)
finally:
    os.chdir(_cwd)
