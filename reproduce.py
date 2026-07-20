#!/usr/bin/env python3
"""Entry point for the released archive.

Forwards all arguments to the public PEFT compatibility runner:

    python reproduce.py --config peft_atlas_lite/configs/multifamily/tinyllama_12x.json --out results/tinyllama_12x
    python reproduce.py --config peft_atlas_lite/configs/smoke.json --out results/smoke --dry-run

One config per panel lives under peft_atlas_lite/configs/.
"""
import os
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "peft_atlas_lite" / "run_peft_atlas_lite.py"
os.execv(sys.executable, [sys.executable, str(RUNNER), *sys.argv[1:]])
