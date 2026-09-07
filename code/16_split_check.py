"""Stage 16 — Count the works that split their own data by patient to prevent leakage.

What it produces
----------------
``split_verdicts.csv``  per work: whether it uses one of the datasets, and whether it splits by
                        patient, with the sentence that shows it

Why it is done this way
-----------------------
The manuscript observes that this precaution is already established in the literature and that
it cannot detect overlap between the datasets themselves. The observation is only worth making
if the count behind it is right, and a search for phrases is not enough: averaging metrics per
patient reads like a patient-level split and is not one.

Each work is therefore examined on two questions at once, and every work the stage returns was
then read by an author before being counted. Of the three counted, one splits the Harvard-GF
data by patient, one declares its splits disjoint at the subject level, and one states that its
test set was built with no patient overlap. An earlier count of five had included a systematic
review that recommends the practice without analysing data; recommending and applying are not
the same act, and the manuscript now says applying.

Inputs
------
the stored full texts from stages 09, 10 and 14; the ``codex`` command-line tool on PATH

Runtime: about 25 minutes.

Manuscript values reproduced
----------------------------
Three of the 64 works read in full split their data by patient (eMethods 7, Discussion).
"""
import csv
import sys
import json
import re
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402  every path is resolved in config.py

OUT = RESULTS / '16_split_check'
OUT.mkdir(parents=True, exist_ok=True)
DIRS = [RESULTS / '09_citation_classify' / 'fulltext',
        RESULTS / '10_fulltext_sweep' / 'text',
        RESULTS / '14_fulltext_sweep3' / 'text']
NAME = re.compile(r'Harvard[-\s]?GDP\b|Harvard\s+Glaucoma\s+Detection\s+and\s+Progression|'
                  r'Harvard[-\s]?GF\b|Harvard\s+Glaucoma\s+Fairness|Harvard[-\s]?FairVision|'
                  r'\bFairVision\b', re.I)
REF = re.compile(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b')

PROMPT = """Answer only from the text given. Two questions about ONE paper.

Q1. Does this paper use Harvard-GDP, Harvard-GF or Harvard-FairVision as data in its own
experiments (not merely cite them in the reference list)?

Q2. Does this paper state that it split its data so that the same patient (or subject) does
not appear in both training and evaluation? Phrases such as "patient-level split",
"patient-wise split", "no patient overlap between train and test", "subject-disjoint",
"split by patient" all count. Averaging metrics per patient does NOT count. A split by image
or by eye without a patient constraint does NOT count.

Reply with one line of JSON and nothing else:
{"uses_harvard_dataset": true|false, "patient_level_split": true|false, "quote": "<verbatim sentence for Q2, or empty>"}
"""


def body(t):
    m = list(REF.finditer(t))
    return t[:m[-1].start()] if m and m[-1].start() > len(t) * 0.4 else t


def windows(t, span=1200):
    """Pass only the windows around dataset mentions and around any description of the split."""
    keys = list(NAME.finditer(t)) + list(re.finditer(
        r'split|train(ing)?\s+set|test\s+set|validation\s+set|cross[-\s]?validation|fold', t, re.I))
    spans = sorted((max(0, m.start() - span), min(len(t), m.end() + span)) for m in keys)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return ''.join(t[a:b] + '\n…\n' for a, b in merged)[:60000]


rows = []
files = sorted(p for d in DIRS if d.exists() for p in d.glob('*.txt'))
print(f'judging {len(files)} full texts')
for k, p in enumerate(files, 1):
    b = body(p.read_text(encoding='utf-8', errors='ignore'))
    if not NAME.search(b):
        rows.append(dict(file=p.name, uses='false', split='false', quote='no dataset in body'))
        print(f'  {k:>3}/{len(files)}  no dataset in body      {p.name[:46]}', flush=True)
        continue
    try:
        r = subprocess.run(['codex', 'exec', '--skip-git-repo-check', '-'],
                           input=PROMPT + '\n\n--- TEXT ---\n' + windows(b),
                           capture_output=True, text=True, timeout=300)
        m = re.search(r'\{[^{}]*patient_level_split[^{}]*\}', r.stdout, re.S)
        j = json.loads(m.group(0)) if m else {}
        u, s, q = j.get('uses_harvard_dataset'), j.get('patient_level_split'), (j.get('quote') or '')[:300]
    except Exception as e:
        u, s, q = 'ERROR', 'ERROR', str(e)[:100]
    rows.append(dict(file=p.name, uses=str(u), split=str(s), quote=q))
    print(f'  {k:>3}/{len(files)}  uses={str(u):6s} patient split={str(s):6s}  {p.name[:44]}', flush=True)

with (OUT / 'split_verdicts.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

both = [r for r in rows if r['uses'] == 'True' and r['split'] == 'True']
print(f'\n=== both conditions met: {len(both)}, each then read by an author ===')
for r in both:
    print(f'  {r["file"][:46]:48s} {r["quote"][:110]}')
print(f'\nWrote {OUT/"split_verdicts.csv"}')
