"""SUPERSEDED — an attempt to recover missing full texts through Europe PMC.

Europe PMC holds open-access full text from PubMed Central and from several preprint servers, so
it was the natural next place to look for the works stage 09 could not read. It identified 13 of
68 DOIs and held full text for none of the works PubMed Central had already missed, because it
indexes neither IEEE nor Springer nor arXiv.

The attempt is kept because it documents where the missing works actually live, which is what led
to the approach that worked: following the open-access locations recorded in OpenAlex and reading
the PDFs directly.

**Do not use this script for the reported analysis.** The analysis of record is
``10_fulltext_sweep.py``.
"""
import json
import sys
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # config.py lives in code/
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

POP = RESULTS / '09_citation_classify' / 'citing_population.csv'
OUT = RESULTS / 'superseded_fulltext_sweep_europepmc'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'fulltext'
CACHE.mkdir(exist_ok=True)
UA = {'User-Agent': 'paperB-audit/1.0 (mailto:s2422005@stu.musashino-u.ac.jp)'}
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'

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


def get(url, tries=4, raw=False):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                b = r.read()
            return b.decode('utf-8', 'replace') if raw else json.loads(b)
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def search_batch(dois):
    """Query DOIs in batches; one request per work would mean 75 round trips."""
    q = ' OR '.join(f'DOI:"{d}"' for d in dois)
    u = EPMC + 'search?' + urllib.parse.urlencode(
        {'query': q, 'format': 'json', 'pageSize': 100, 'resultType': 'core'})
    d = get(u)
    return (d or {}).get('resultList', {}).get('result', [])


def full_text(src, eid):
    f = CACHE / f'{src}_{eid}.txt'
    if f.exists():
        return f.read_text()
    xml = get(EPMC + f'{src}/{eid}/fullTextXML', raw=True)
    if not xml or '<' not in xml:
        return ''
    body = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', xml))
    f.write_text(body)
    time.sleep(0.2)
    return body


def sentences(t):
    return [x.strip() for x in re.split(r'(?<=[.;])\s+(?=[A-Z(])', t) if 8 < len(x) < 600]


def main():
    w('=' * 96)
    w('SUPERSEDED attempt to recover missing full texts through Europe PMC')
    w('=' * 96)
    d = pd.read_csv(POP).fillna('')
    d['pmid'] = d.pmid.astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')
    have = d.pmcid.astype(str).str.startswith('PMC')
    w(f'  citing population {len(d)} / already have full text {int(have.sum())} / missing {int((~have).sum())}')

    need = d[~have]
    dois = [x for x in need.doi if x]
    w(f'  querying {len(dois)} missing works that carry a DOI, in batches')

    found = {}
    for i in range(0, len(dois), 20):
        for r in search_batch(dois[i:i + 20]):
            doi = (r.get('doi') or '').lower()
            if doi:
                found[doi] = r
        time.sleep(0.4)
    w(f'  identified in Europe PMC: {len(found)}')

    avail = []
    for doi, r in found.items():
        src, eid = r.get('source', ''), r.get('id', '')
        if r.get('inEPMC') == 'Y':
            avail.append((doi, 'PMC', r.get('pmcid', '').replace('PMC', 'PMC'), r))
        elif src == 'PPR':
            avail.append((doi, 'PPR', eid, r))
    w(f'  of which full text may be retrievable: {len(avail)}'
      f'  (PMC {sum(1 for a in avail if a[1]=="PMC")} / preprints {sum(1 for a in avail if a[1]=="PPR")})')

    w('\n[fetch full text and classify]')
    rows = []
    for k, (doi, src, eid, meta) in enumerate(avail, 1):
        body = full_text('PMC' if src == 'PMC' else 'PPR', eid)
        if len(body) < 3000:
            continue
        m = list(REF_HEAD.finditer(body))
        main_text = body[:m[-1].start()] if m else body
        in_main = [nm for nm, rx in NAME_RE.items() if rx.search(main_text)]
        uses = bool(in_main) and bool(USE_RE.search(main_text))
        dup = [s for s in sentences(main_text) if DUP_RE.search(s)]
        rows.append(dict(doi=doi, source=src, eid=eid, title=(meta.get('title') or '')[:150],
                         venue=meta.get('journalTitle') or meta.get('bookOrReportDetails', {}).get('publisher', ''),
                         year=meta.get('pubYear', ''), in_main='; '.join(in_main),
                         uses_dataset=bool(in_main), uses=uses, n_dup=len(dup),
                         dup_evidence=' || '.join(dup[:6])[:1800]))
        if k % 15 == 0:
            w(f'    …{k}/{len(avail)}')
    n = pd.DataFrame(rows)
    n.to_csv(OUT / 'newly_classified.csv', index=False)

    w('\n' + '=' * 96)
    w('result')
    w('=' * 96)
    w(f'  newly classified from full text  {len(n)}')
    if len(n):
        w(f'    names a dataset in the main text  {int(n.uses_dataset.sum())}')
        w(f'    used in their own analysis        {int(n.uses.sum())}')
        w(f'    contains a duplication-related term {int((n.n_dup > 0).sum())}  <- each is read')
        w('\n' + '-' * 96)
        w('the sentences that triggered the flag, in full')
        w('-' * 96)
        for _, r in n[n.n_dup > 0].iterrows():
            w(f'\n  [{r.source} {r.eid}] {str(r.venue)[:46]} ({r.year})  uses: {r.in_main or "none named"}')
            w(f'    {str(r.title)[:104]}')
            for s in str(r.dup_evidence).split(' || ')[:4]:
                w(f'      - {s[:220]}')
    total_ft = int(have.sum()) + len(n)
    w('\n' + '=' * 96)
    w(f'  full text classified: {int(have.sum())} -> {total_ft} / {len(d)}'
      f'（{int(have.sum())/len(d):.0%} → {total_ft/len(d):.0%}）')
    w(f'  still not obtainable: {len(d) - total_ft}')
    json.dump(dict(population=int(len(d)), had_fulltext=int(have.sum()),
                   newly_retrieved=int(len(n)), total_fulltext=total_ft,
                   new_users=int(n.uses.sum()) if len(n) else 0,
                   new_flagged=int((n.n_dup > 0).sum()) if len(n) else 0),
              open(OUT / 'summary.json', 'w'), indent=2)
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
