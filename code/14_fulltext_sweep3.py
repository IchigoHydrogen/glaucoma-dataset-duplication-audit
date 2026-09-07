"""Stage 14 — Read the full texts that stage 13 found, under the rules used before.

What it produces
----------------
``sweep3_classified.csv``  one row per work read here, classified exactly as in stages 09 and 10
``text/``                  the extracted text, kept locally and not redistributed

Why it is done this way
-----------------------
The classification rules are copied unchanged from stage 10. Changing them would leave the
works read here judged by a different standard from the works read earlier, and the count of
works reporting duplication is only meaningful if one standard applies throughout.

Ten of the sixteen contain a term related to duplication, overlap or leakage. Every one was
read by an author, and in every case the term carried a different meaning: overlap between
embedding distributions, overlap between a prediction and a ground truth, leakage of a sensitive
attribute, a train-test split inside the work's own data, or a dataset whose name contains the
word. The sentence that triggered each flag is kept in ``dup_evidence`` so that the judgment can
be checked; it is trimmed to one sentence, because these are third-party papers.

The extracted text itself is not redistributed. Rerunning this stage recreates it.

Inputs
------
``availability.csv`` from stage 13; network access to the open-access PDFs

Runtime: about 3 minutes. No GPU.

Manuscript values reproduced
----------------------------
15 further distinct works read in full, of which 4 use one of the datasets; none reports
duplication, patient overlap or leakage between the datasets (eMethods 7).
"""
import csv
import sys
import io
import re
import time
import urllib.request
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402  every path is resolved in config.py

AVAIL = RESULTS / '13_fulltext_probe3' / 'availability.csv'
OUT = RESULTS / '14_fulltext_sweep3'
TEXT = OUT / 'text'
TEXT.mkdir(parents=True, exist_ok=True)
MAIL = 's2422005@stu.musashino-u.ac.jp'
UA = {'User-Agent': f'paperB-audit/1.0 (mailto:{MAIL})'}

NAME_RE = {
    'Harvard-GDP': re.compile(r'Harvard[-\s]?GDP\b|Harvard\s+Glaucoma\s+Detection\s+and\s+Progression', re.I),
    'Harvard-GF': re.compile(r'Harvard[-\s]?GF\b|Harvard\s+Glaucoma\s+Fairness', re.I),
    'Harvard-FairVision': re.compile(r'Harvard[-\s]?FairVision|\bFairVision\b', re.I),
}
USE_RE = re.compile(r'external(ly)?\s+valid|independent\s+(test|validation|cohort|dataset)|'
                    r'pre[-\s]?train|benchmark|fine[-\s]?tun|train(ed|ing)?\s+on|'
                    r'evaluat(ed|ion)\s+on|we\s+use[d]?\s+the|experiments?\s+on', re.I)
DUP_RE = re.compile(r'duplicat|overlap|data\s+leak|leakage|same\s+patients?|shared\s+patients?|'
                    r'patient\s+overlap|identical\s+(image|scan|volume)|contaminat|'
                    r'non[-\s]?independen|not\s+independent', re.I)
REF_HEAD = re.compile(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b')


def sentences(t):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]


def main_body(t):
    """Drop the reference list, so a term inside another work's title is not read as a mention."""
    m = list(REF_HEAD.finditer(t))
    return t[:m[-1].start()] if m and m[-1].start() > len(t) * 0.4 else t


rows = []
avail = [r for r in csv.DictReader(AVAIL.open(encoding='utf-8')) if r['available'] == '1']
print(f'attempting {len(avail)} downloads\n')

for k, r in enumerate(avail, 1):
    try:
        with urllib.request.urlopen(urllib.request.Request(r['url'], headers=UA), timeout=90) as f:
            blob = f.read()
        text = ''
        for page in PdfReader(io.BytesIO(blob)).pages:
            text += (page.extract_text() or '') + '\n'
    except Exception as e:
        print(f'  {k:>3}/{len(avail)}  failed {str(e)[:40]:42s} {r["title"][:50]}', flush=True)
        rows.append(dict(title=r['title'], year=r['year'], venue=r['venue'], doi=r['doi'],
                         route=r['route'], chars=0, in_main='', uses='', n_dup='',
                         dup_evidence='FETCH FAILED'))
        continue

    slug = re.sub(r'\W+', '_', (r['doi'] or r['title'])[:60]).strip('_')
    (TEXT / f'{slug}.txt').write_text(text, encoding='utf-8')

    body = main_body(text)
    in_main = [n for n, rx in NAME_RE.items() if rx.search(body)]
    dup = [s for s in sentences(body) if DUP_RE.search(s)]
    rows.append(dict(title=r['title'], year=r['year'], venue=r['venue'], doi=r['doi'],
                     route=r['route'], chars=len(text), in_main='; '.join(in_main),
                     uses=bool(in_main) and bool(USE_RE.search(body)),
                     n_dup=len(dup), dup_evidence=' || '.join(dup[:6])[:2000]))
    print(f'  {k:>3}/{len(avail)}  {len(text):>7} chars  datasets named {"; ".join(in_main) or "none":34s} '
          f'duplication terms {len(dup):>3}  {r["title"][:44]}', flush=True)
    time.sleep(1.0)

with (OUT / 'sweep3_classified.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

ok = [r for r in rows if r['chars']]
print(f'\n=== newly classified from full text: {len(ok)} ===')
print(f'  datasets named in the body                : {sum(1 for r in ok if r["in_main"])}')
print(f'  used in the work\'s own analysis (automatic): {sum(1 for r in ok if r["uses"] is True)}')
flagged = [r for r in ok if r['n_dup']]
print(f'  contains a duplication-related term, to be read by an author: {len(flagged)}')
print(f'\nWrote {OUT}')
