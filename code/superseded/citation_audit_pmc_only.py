"""SUPERSEDED — the first audit of downstream use, which searched PubMed Central alone.

This version is kept because the manuscript reports that its first audit was inadequate, and a
reader should be able to check that claim rather than accept it.

PubMed Central carries no full text from JAMA Network, Elsevier or IEEE journals, nor from
preprint servers. Two of the three studies named in the manuscript's own Results were therefore
invisible to it, and the population it returned was roughly half the size of the union that five
indexes return. The gaps are not random: they fall on exactly the venues where this literature
appears.

**Do not use this script for the reported analysis.** The analysis of record is
``08_citation_union.py`` together with stages 09 and 10.
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

PREFIX = Path(__file__).stem
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'fulltext'
CACHE.mkdir(exist_ok=True)
BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SLEEP = 0.4

# How the datasets are referred to, allowing for spelling variants.
DATASETS = {
    'Harvard-GDP': ['"Harvard-GDP"', '"Harvard GDP"', '"Glaucoma Detection and Progression"'],
    'Harvard-GF': ['"Harvard-GF"', '"Harvard GF"', '"Harvard Glaucoma Fairness"'],
    'Harvard-FairVision': ['"Harvard-FairVision"', '"FairVision"', '"Harvard FairVision"'],
}
# "glaucoma detection and progression" is also ordinary prose, so it counts as a dataset name
# only when "Harvard" is adjacent; the false positives were measured and this was corrected.
NAME_RE = {
    'Harvard-GDP': re.compile(r'Harvard[-\s]?GDP\b|Harvard\s+Glaucoma\s+Detection\s+and\s+Progression', re.I),
    'Harvard-GF': re.compile(r'Harvard[-\s]?GF\b|Harvard\s+Glaucoma\s+Fairness', re.I),
    'Harvard-FairVision': re.compile(r'Harvard[-\s]?FairVision|\bFairVision\b', re.I),
}
# Separate a work that merely cites a dataset from one that used it.
# A name appearing only in the reference list counts as a mention, not as use.
REF_HEAD = re.compile(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b')
# use verbs
USE_RE = re.compile(r'external(ly)?\s+valid|independent\s+(test|validation|cohort|dataset)|'
                    r'pre[-\s]?train|benchmark|fine[-\s]?tun|train(ed|ing)?\s+on|evaluat(ed|ion)\s+on', re.I)
# terms that would signal a report of duplication or leakage
DUP_RE = re.compile(r'duplicat|overlap|data\s+leak|leakage|same\s+patients?|shared\s+patients?|'
                    r'patient\s+overlap|identical\s+(image|scan|volume)|contaminat|'
                    r'non[-\s]?independen|not\s+independent', re.I)

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def _get(url, tries=6):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'could not fetch: {url[:110]} ... {last}')


def call(ep, pairs):
    out = _get(BASE + ep + '?' + urllib.parse.urlencode(pairs))
    time.sleep(SLEEP)
    return out


def esearch(term, db='pubmed', retmax=500):
    d = json.loads(call('esearch.fcgi', [('db', db), ('term', term),
                                         ('retmax', retmax), ('retmode', 'json')]))
    r = d['esearchresult']
    return int(r['count']), r.get('idlist', [])


def sentences(t):
    return [x.strip() for x in re.split(r'(?<=[.;])\s+(?=[A-Z(])', t) if 8 < len(x) < 600]


def main():
    w('=' * 100)
    w('SUPERSEDED audit of downstream use, searching PubMed Central full text only')
    w('=' * 100)

    # ---------- 1. count how often the datasets are used ----------
    w('\n[1] how often the datasets appear (PubMed Central full-text search)')
    counts = {}
    for name, terms in DATASETS.items():
        q = ' OR '.join(terms)
        n, _ = esearch(q, db='pmc', retmax=0)
        counts[name] = n
        w(f'  {name:22s} {n:5d}')
    any_q = ' OR '.join(t for ts in DATASETS.values() for t in ts)
    n_any, _ = esearch(any_q, db='pmc', retmax=0)
    w(f'  {"any of the three":22s} {n_any:5d}')
    n_two, _ = esearch(f'({" OR ".join(DATASETS["Harvard-GF"])}) AND '
                       f'({" OR ".join(DATASETS["Harvard-FairVision"])})', db='pmc', retmax=0)
    w(f'  {"both GF and FairVision":22s} {n_two:5d}  <- most directly exposed')

    # ---------- 2. fetch full text and classify ----------
    w('\n[2] fetch full text and classify')
    n_all, ids = esearch(any_q, db='pmc', retmax=400)
    w(f'  records to fetch: {len(ids)}')
    rows = []
    for k, pmc in enumerate(ids, 1):
        f = CACHE / f'PMC{pmc}.txt'
        if f.exists():
            body = f.read_text()
        else:
            try:
                xml = call('efetch.fcgi', [('db', 'pmc'), ('id', pmc), ('retmode', 'xml')])
            except Exception:
                continue
            body = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', xml))
            f.write_text(body)
        if len(body) < 3000:
            continue
        used = [nm for nm, rx in NAME_RE.items() if rx.search(body)]
        if not used:
            continue
        # Use is judged from the main text, that is, everything before the reference list.
        m = list(REF_HEAD.finditer(body))
        main_text = body[:m[-1].start()] if m else body
        in_main = [nm for nm in used if NAME_RE[nm].search(main_text)]
        # Mentions of duplication are picked up two ways: within a sentence that also names a dataset,
        # and anywhere in the main text, because the two can appear in different sentences.
        dup_sent = [s for s in sentences(body)
                    if DUP_RE.search(s) and any(NAME_RE[nm].search(s) for nm in used)]
        dup_doc = [s for s in sentences(main_text)
                   if DUP_RE.search(s)] if in_main else []
        # Use is judged at document level; requiring name and verb in one sentence misses too much.
        use_doc = bool(in_main) and bool(USE_RE.search(main_text))
        use_ev = [s for s in sentences(main_text)
                  if USE_RE.search(s) and any(NAME_RE[nm].search(s) for nm in in_main)]
        rows.append(dict(pmcid=f'PMC{pmc}', chars=len(body),
                         datasets='; '.join(used), n_datasets=len(used),
                         in_main_text='; '.join(in_main), uses_dataset=bool(in_main),
                         uses_as_benchmark=use_doc,
                         flags_duplication=bool(dup_sent),
                         dup_doc_mentions=len(dup_doc),
                         dup_evidence=' || '.join((dup_sent or dup_doc[:3]))[:1500],
                         use_evidence=' || '.join(use_ev[:2])[:600]))
        if k % 40 == 0:
            w(f'    ...{k}/{len(ids)}')
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'citation_audit.csv', index=False)

    w('\n' + '-' * 100)
    w(f'works classified from full text: {len(d)}')
    w(f'  names a dataset in the main text: {int(d.uses_dataset.sum())}'
      f'  (the other {int((~d.uses_dataset).sum())} only in the reference list)')
    w(f'  mentions two or more datasets: {int((d.n_datasets >= 2).sum())}')
    w(f'  used for external validation, pretraining or benchmarking: '
      f'{int(d.uses_as_benchmark.sum())}')
    w(f'  flagged automatically as mentioning duplication or leakage: {int(d.flags_duplication.sum())}')

    w('\n' + '-' * 100)
    w('works whose main text contains a duplication-related term, wherever it appears')
    dd = d[(d.dup_doc_mentions > 0) & d.uses_dataset]
    w(f'  {len(dd)}. every triggering sentence follows, to be read by hand')
    for _, r in dd.iterrows():
        w(f'\n  {r.pmcid}  [{r.in_main_text}]')
        for x in str(r.dup_evidence).split(' || ')[:3]:
            w(f'    {x[:240]}')

    w('\n' + '-' * 100)
    w('sentences that triggered the flag, in full; a negative result needs reading, not matching')
    fl = d[d.flags_duplication]
    if len(fl) == 0:
        w('  none')
    for _, r in fl.iterrows():
        w(f'\n  {r.pmcid}  [{r.datasets}]')
        for s in str(r.dup_evidence).split(' || '):
            w(f'    {s[:260]}')

    w('\n' + '-' * 100)
    w('counts by combination of datasets mentioned')
    for combo, c in d.datasets.value_counts().items():
        w(f'  {c:4d}  {combo}')

    (OUT / 'summary.json').write_text(json.dumps({
        'pmc_counts': counts, 'pmc_any': n_any, 'pmc_gf_and_fairvision': n_two,
        'fulltext_judged': int(len(d)),
        'two_or_more_datasets': int((d.n_datasets >= 2).sum()),
        'uses_as_benchmark': int(d.uses_as_benchmark.sum()),
        'flags_duplication_auto': int(d.flags_duplication.sum()),
        'note': 'automated flags; every flagged work must be read by hand',
    }, indent=2, ensure_ascii=False))
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
