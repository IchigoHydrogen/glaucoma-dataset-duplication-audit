"""Stage 10 — Retrieve more full texts, and screen what still cannot be read.

What it produces
----------------
``newly_classified.csv``  works whose full text was obtained here and classified as in stage 09
``abstract_screen.csv``   the remainder, screened on title and, where available, abstract

Why it is done this way
-----------------------
Stage 09 could read 32 of the 107, because it looked only where PubMed Central holds full text.
The works it missed are mostly preprints and conference proceedings, which are open access but
indexed elsewhere. Following the open-access locations recorded in OpenAlex and reading the PDFs
directly raises the count to 49. Europe PMC was also tried and held full text for none of the
works PubMed Central had missed, because it indexes neither IEEE nor Springer nor arXiv; that
attempt is kept in ``superseded/fulltext_sweep_europepmc.py``.

The remaining 58 are screened on title, and on abstract where one could be retrieved. These
levels do not carry equal weight and the manuscript does not treat them as though they did. A
full-text search can detect a remark in a Methods section, which is where an observation of this
kind would most probably appear. A title or abstract screen would identify a study devoted to
dataset duplication but would not detect a passing remark. The finding therefore rests on the
49 full texts, and the screen of the remaining 58 is a check and not evidence of absence.

Inputs
------
``citing_population.csv`` from stage 09; network access to OpenAlex and the open-access PDFs

Runtime: about 5 minutes. No GPU.

Manuscript values reproduced
----------------------------
17 further works read in full at this stage, and the remainder screened on title and abstract.
Stages 13 and 14 read 15 more, and stage 17 assembles the counts the manuscript prints.
"""
import io
import sys
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

POP = RESULTS / '09_citation_classify' / 'citing_population.csv'
OUT = RESULTS / '10_fulltext_sweep'
OUT.mkdir(parents=True, exist_ok=True)
PDFS = OUT / 'text'
PDFS.mkdir(exist_ok=True)
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

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def get(url, tries=3, binary=False, timeout=45):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                b = r.read()
            return b if binary else json.loads(b)
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def inverted_to_text(inv):
    if not inv:
        return ''
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return ' '.join(pos[i] for i in sorted(pos))


def pdf_text(url, key):
    f = PDFS / f'{key}.txt'
    if f.exists():
        return f.read_text()
    b = get(url, binary=True, timeout=60)
    if not b or not b[:5].startswith(b'%PDF'):
        return ''
    try:
        rd = PdfReader(io.BytesIO(b))
        t = ' '.join((p.extract_text() or '') for p in rd.pages[:40])
    except Exception:
        return ''
    t = re.sub(r'\s+', ' ', t)
    if len(t) > 2000:
        f.write_text(t)
    return t


def sentences(t):
    return [x.strip() for x in re.split(r'(?<=[.;])\s+(?=[A-Z(])', t) if 8 < len(x) < 600]


def main():
    d = pd.read_csv(POP).fillna('')
    have = d.pmcid.astype(str).str.startswith('PMC')
    need = d[~have].reset_index(drop=True)
    w('=' * 96)
    w(f'sweeping the {len(need)} works without full text, via open-access PDFs and abstracts')
    w('=' * 96)

    # ---------- open-access locations and abstracts, from OpenAlex ----------
    meta = {}
    dois = [x for x in need.doi if x]
    for i in range(0, len(dois), 40):
        chunk = '|'.join(f'https://doi.org/{x}' for x in dois[i:i + 40])
        u = (f'https://api.openalex.org/works?mailto={MAIL}&per-page=50&filter=doi:'
             + urllib.parse.quote(chunk, safe='|:/.'))
        r = get(u)
        for rec in (r or {}).get('results', []):
            k = (rec.get('doi') or '').replace('https://doi.org/', '').lower()
            meta[k] = rec
        time.sleep(0.3)
    w(f'  bibliographic records retrieved from OpenAlex: {len(meta)} / {len(dois)}')

    def oa_pdf(rec):
        cands = []
        b = rec.get('best_oa_location') or {}
        if b.get('pdf_url'):
            cands.append(b['pdf_url'])
        for loc in rec.get('locations') or []:
            if loc.get('pdf_url'):
                cands.append(loc['pdf_url'])
            lp = (loc.get('landing_page_url') or '')
            m = re.search(r'arxiv\.org/abs/([\w.\-/]+)', lp)
            if m:
                cands.append(f'https://arxiv.org/pdf/{m.group(1)}')
        return list(dict.fromkeys(cands))[:2]

    # ---------- [1] fetch full text where an open-access PDF exists ----------
    w('\n[1] full text from open-access PDFs, then classification')
    rows, no_ft = [], []
    for k, (_, r) in enumerate(need.iterrows(), 1):
        rec = meta.get(str(r.doi).lower(), {})
        txt = ''
        for u in oa_pdf(rec):
            txt = pdf_text(u, re.sub(r'[^A-Za-z0-9]', '_', str(r.doi) or str(k))[:60])
            if len(txt) > 3000:
                break
        if len(txt) < 3000:
            no_ft.append((r, rec))
            continue
        m = list(REF_HEAD.finditer(txt))
        main_text = txt[:m[-1].start()] if m else txt
        in_main = [nm for nm, rx in NAME_RE.items() if rx.search(main_text)]
        dup = [s for s in sentences(main_text) if DUP_RE.search(s)]
        rows.append(dict(doi=r.doi, venue=r.venue, year=r.year, title=str(r.title)[:130],
                         in_main='; '.join(in_main), uses_dataset=bool(in_main),
                         uses=bool(in_main) and bool(USE_RE.search(main_text)),
                         n_dup=len(dup), dup_evidence=' || '.join(dup[:5])[:1600]))
        if k % 20 == 0:
            w(f'    ...{k}/{len(need)}  retrieved {len(rows)}')
    n = pd.DataFrame(rows)
    if len(n):
        n.to_csv(OUT / 'newly_classified.csv', index=False)
    w(f'  newly classified from full text: {len(n)}')
    if len(n):
        w(f'    names a dataset in the main text {int(n.uses_dataset.sum())} / used in their own analysis {int(n.uses.sum())}')
        w(f'    contains a duplication-related term {int((n.n_dup > 0).sum())}')
        for _, x in n[n.n_dup > 0].iterrows():
            w(f'\n    [{x.venue[:40]} {x.year}] {str(x.title)[:88]}  uses: {x.in_main or "none named"}')
            for s in str(x.dup_evidence).split(' || ')[:3]:
                w(f'      - {s[:210]}')

    # ---------- [2] screen the remainder on title and, where available, abstract ----------
    w('\n[2] screening the remainder on title and, where available, abstract')
    scr, flagged = [], []
    for r, rec in no_ft:
        ab = inverted_to_text(rec.get('abstract_inverted_index'))
        blob = f'{r.title} {ab}'
        hit = DUP_RE.search(blob)
        scr.append(dict(doi=r.doi, venue=r.venue, year=r.year, title=str(r.title)[:130],
                        has_abstract=bool(ab), dup_term=bool(hit),
                        evidence=(blob[max(0, hit.start()-90):hit.start()+110] if hit else '')))
        if hit:
            flagged.append(scr[-1])
    sc = pd.DataFrame(scr)
    sc.to_csv(OUT / 'abstract_screen.csv', index=False)
    w(f'  screened: {len(sc)} works, of which {int(sc.has_abstract.sum())} had a retrievable abstract')
    w(f'  duplication-related term in title or abstract: {len(flagged)}  <- every one is read')
    for x in flagged:
        w(f'\n    [{str(x["venue"])[:40]} {x["year"]}] {x["title"][:92]}')
        w(f'      …{re.sub(chr(10)," ",x["evidence"])}…')

    total = int(have.sum()) + len(n)
    w('\n' + '=' * 96)
    w(f'  full text classified: {int(have.sum())} -> {total} / {len(d)}'
      f'（{int(have.sum())/len(d):.0%} → {total/len(d):.0%}）')
    w(f'  screened at some level, including title and abstract: {total + int(sc.has_abstract.sum())} / {len(d)}')
    json.dump(dict(population=int(len(d)), had=int(have.sum()), newly=int(len(n)),
                   total_fulltext=total, screened_abstract=int(sc.has_abstract.sum()),
                   flagged_fulltext=int((n.n_dup > 0).sum()) if len(n) else 0,
                   flagged_abstract=len(flagged)),
              open(OUT / 'summary.json', 'w'), indent=2)
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
