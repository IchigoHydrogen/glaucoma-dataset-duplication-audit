"""Every path used by this archive is resolved here, and nowhere else.

By default the archive expects this layout::

    <archive root>/
        code/          this directory
        dataset/       the three public datasets, downloaded by the user (see README)
        results/       written by the scripts

Override either location with an environment variable if your copies live elsewhere::

    export PAPERB_DATASET=/mnt/data/harvard
    export PAPERB_RESULTS=/scratch/paperB
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('PAPERB_ROOT', Path(__file__).resolve().parent.parent))
DATASET = Path(os.environ.get('PAPERB_DATASET', ROOT / 'dataset'))
RESULTS = Path(os.environ.get('PAPERB_RESULTS', ROOT / 'results'))
RESULTS.mkdir(parents=True, exist_ok=True)
