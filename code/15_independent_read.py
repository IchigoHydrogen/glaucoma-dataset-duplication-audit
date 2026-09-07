"""Stage 15 — Check that the negative result does not depend on our choice of search terms.

What it produces
----------------
``verdicts.csv``  one verdict per stored full text

Why it is done this way
-----------------------
The claim that no citing work reports duplication rests on a list of terms. A work that
discussed the issue in words absent from that list would be missed. To test that, every stored
full text was read a second time by a separate model, asked a single question and given the
passages around each dataset mention. The question distinguishes overlap between these public
datasets from the meanings the term usually carries elsewhere.

Only text already held on disk is used. The model is never asked about a paper we could not
obtain, and it does not make the determination: it looks for a passage an author would then
read. It found none, and no reported number changed.

This is an internal check. The manuscript does not report it, because it altered nothing and
answers a question no reviewer raised. It is kept here so that the check can be repeated.

Restricting the text to windows around each dataset mention is sufficient for the purpose: a
work reporting that these three releases share images must name at least one of them.

Inputs
------
the stored full texts from stages 09, 10 and 14; the ``codex`` command-line tool on PATH

Runtime: about 25 minutes.

Manuscript values reproduced
----------------------------
None. The check confirmed the existing classification and is reported only here.
"""
import csv
import sys
import json
import re
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402  every path is resolved in config.py

OUT = RESULTS / '15_independent_read'
OUT.mkdir(parents=True, exist_ok=True)

DIRS = [RESULTS / '09_citation_classify' / 'fulltext',
        RESULTS / '10_fulltext_sweep' / 'text',
        RESULTS / '14_fulltext_sweep3' / 'text']
NAME = re.compile(r'Harvard[-\s]?GDP\b|Harvard\s+Glaucoma\s+Detection\s+and\s+Progression|'
                  r'Harvard[-\s]?GF\b|Harvard\s+Glaucoma\s+Fairness|'
                  r'Harvard[-\s]?FairVision|\bFairVision\b', re.I)

PROMPT = """You are auditing a paper for one specific fact. Answer only from the text given.

Question: does this text state that two or more PUBLIC GLAUCOMA DATASETS (Harvard-GDP,
Harvard-GF, Harvard-FairVision) share patients, share eyes, contain duplicated or identical
images, or otherwise are not independent of one another?

This is NOT what I am asking about, and each must be answered "no":
  - leakage or overlap inside the authors' own train/test split
  - "information leakage" of a sensitive attribute in a representation
  - distribution overlap of embeddings, classes or scatter plots
  - Dice or IoU overlap between a prediction and a ground truth
  - a dataset whose NAME contains the word "overlap"
  - patient overlap discussed for some other dataset (MIMIC, CheXpert, ...)

Reply with one line of JSON and nothing else:
{"reports_cross_dataset_duplication": true|false, "quote": "<verbatim sentence, or empty>"}
"""


def windows(text, span=1400):
    out, last = [], -1
    for m in NAME.finditer(text):
        a, b = max(0, m.start() - span), min(len(text), m.end() + span)
        if a <= last:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
        last = b
    return ''.join(text[a:b] + '\n…\n' for a, b in out[:12])


rows = []
files = sorted(p for d in DIRS if d.exists() for p in d.glob('*.txt'))
print(f'{len(files)} stored full texts')
for k, p in enumerate(files, 1):
    text = p.read_text(encoding='utf-8', errors='ignore')
    ctx = windows(text)
    if not ctx.strip():
        rows.append(dict(file=p.name, mentions=0, verdict='no dataset mention', quote=''))
        print(f'  {k:>3}/{len(files)}  no dataset named        {p.name[:52]}', flush=True)
        continue
    ctx = ctx[:60000]
    try:
        r = subprocess.run(['codex', 'exec', '--skip-git-repo-check', '-'],
                           input=PROMPT + '\n\n--- TEXT ---\n' + ctx,
                           capture_output=True, text=True, timeout=300)
        raw = r.stdout.strip()
        m = re.search(r'\{[^{}]*reports_cross_dataset_duplication[^{}]*\}', raw, re.S)
        j = json.loads(m.group(0)) if m else {}
        v = j.get('reports_cross_dataset_duplication')
        q = (j.get('quote') or '')[:400]
    except Exception as e:
        v, q = 'ERROR', str(e)[:120]
    rows.append(dict(file=p.name, mentions=len(NAME.findall(text)),
                     verdict=('REPORTS' if v is True else 'no' if v is False else str(v)), quote=q))
    print(f'  {k:>3}/{len(files)}  {rows[-1]["verdict"]:20s} mentions {rows[-1]["mentions"]:>3}  {p.name[:44]}',
          flush=True)

with (OUT / 'verdicts.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

pos = [r for r in rows if r['verdict'] == 'REPORTS']
err = [r for r in rows if r['verdict'] not in ('REPORTS', 'no', 'no dataset mention')]
print(f'\n=== verdicts from the independent reader ===')
print(f'  works naming a dataset       : {sum(1 for r in rows if r["mentions"])}')
print(f'  judged to report duplication : {len(pos)}')
for r in pos:
    print(f'    ★ {r["file"]}\n       {r["quote"][:220]}')
print(f'  no verdict                   : {len(err)}')
print(f'\nWrote {OUT/"verdicts.csv"}')
