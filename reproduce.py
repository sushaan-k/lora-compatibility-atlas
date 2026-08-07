#!/usr/bin/env python3
"""Entry point for the released archive."""
import os
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "peft_atlas_lite" / "run_peft_atlas_lite.py"
os.execv(sys.executable, [sys.executable, str(RUNNER), *sys.argv[1:]])
