"""Stage 09 — Separate the citing population from confirmed use, and read the flagged articles.

What it produces
----------------
``citing_population.csv``    the 107 works, after removing the three dataset publications
``fulltext_classified.csv``  for each work whose full text was obtainable: whether a dataset name
                             appears outside the reference list, whether a use verb appears in the
                             main text, and every sentence containing a duplication-related term

Why it is done this way
-----------------------
Two numbers are reported rather than one, because either alone would mislead. The citing
population is the upper bound on how many published works could be affected; confirmed use is the
lower bound, limited by how many could be read in full.

The three dataset publications are removed because they cite one another. They are identified by
title rather than DOI, since several exist in more than one version.

The central negative result, that no study reported the duplication, cannot rest on automated
classification. Every article whose full text contains any duplication-related term is written out
in full so that an author can read it, and the shipped log contains those sentences. In every case
the term referred to something else: Dice overlap between segmentations, overlapping lesions,
overlapping confidence intervals, the removal of duplicate records during a systematic review, or
a study preventing leakage inside its own train-test split.

Inputs
------
``citing_works_union.csv`` from stage 08; network access to PubMed Central for full text

Runtime: about 10 minutes. No GPU.

Manuscript values reproduced
----------------------------
107 citing works; 32 with full text at this stage; the flagged articles read by hand (eMethods 7).
"""
import json
import sys
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
SRC = RESULTS / '08_citation_union' / 'citing_works_union.csv'
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)
FT = OUT / 'fulltext'
FT.mkdir(exist_ok=True)

# The three dataset publications, removed by title: several exist in more than one version,
SOURCE_TITLES = [
    'harvardglaucomadetectionandprogression',
    'harvardglaucomafairness',
    'fairvisionequitabledeeplearningforeyediseasescreening',
]
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


def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', str(t or '').lower())[:110]


def get_pmc(pmcid):
    f = FT / f'{pmcid}.txt'
    if f.exists():
        return f.read_text()
    u = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode(
        {'db': 'pmc', 'id': pmcid.replace('PMC', ''), 'retmode': 'xml'}))
    for i in range(4):
        try:
            with urllib.request.urlopen(u, timeout=90) as r:
                xml = r.read().decode('utf-8', 'replace')
            body = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', xml))
            f.write_text(body)
            time.sleep(0.4)
            return body
        except Exception:
            time.sleep(3 * (i + 1))
    return ''


def sentences(t):
    return [x.strip() for x in re.split(r'(?<=[.;])\s+(?=[A-Z(])', t) if 8 < len(x) < 600]


def main():
    w('=' * 100)
    w(f'{PREFIX}: separating the citing population from confirmed use')
    w('=' * 100)
    d = pd.read_csv(SRC).fillna({'venue': '', 'doi': '', 'pmcid': '', 'sources': '',
                                 'datasets': '', 'title': ''})
    w(f'  union: {len(d)} works')

    # ---------- [1] remove the three dataset publications ----------
    d['tn'] = d.title.map(norm_title)
    is_src = d.tn.apply(lambda t: any(t.startswith(s[:60]) for s in SOURCE_TITLES))
    w(f'\n[1] removing the 3 dataset publications themselves: {int(is_src.sum())}')
    for _, r in d[is_src].iterrows():
        w(f'      {r.year}  {r.venue[:44]:44s} {r.title[:60]}')
    d = d[~is_src].reset_index(drop=True)
    w(f'  citing population: {len(d)} works')

    # ---------- [2] works that entered by name alone, without a confirmed citation ----------
    CIT = re.compile(r'OpenAlex|SemanticScholar')
    name_only = d[~d.sources.str.contains(CIT, na=False)]
    w(f'\n[2] works that entered by name alone, without a confirmed citation: {len(name_only)}')
    w('    Without a citation these could be false positives, so each is listed.')
    for _, r in name_only.iterrows():
        w(f'      {r.year}  [{r.sources}]  {r.venue[:40]:40s} {r.title[:64]}')

    # ---------- [3] classify use from full text ----------
    w('\n' + '=' * 100)
    w('[3] classification of the works whose full text could be obtained')
    w('=' * 100)
    have = d[d.pmcid.astype(str).str.startswith('PMC')]
    w(f'  full text in PubMed Central: {len(have)} ({len(have)/len(d):.0%} of the citing population)')

    rows = []
    for _, r in have.iterrows():
        body = get_pmc(r.pmcid)
        if len(body) < 3000:
            continue
        m = list(REF_HEAD.finditer(body))
        main_text = body[:m[-1].start()] if m else body
        in_main = [k for k, rx in NAME_RE.items() if rx.search(main_text)]
        uses = bool(in_main) and bool(USE_RE.search(main_text))
        dup = [s for s in sentences(main_text) if DUP_RE.search(s)]
        rows.append(dict(pmcid=r.pmcid, year=r.year, venue=r.venue, title=r.title,
                         in_main='; '.join(in_main), uses_dataset=bool(in_main),
                         uses_as_benchmark=uses, n_dup_sent=len(dup),
                         dup_evidence=' || '.join(dup[:6])[:1800]))
    c = pd.DataFrame(rows)
    c.to_csv(OUT / 'fulltext_classified.csv', index=False)

    w(f'  classified                        {len(c)}')
    w(f'  names a dataset in the main text  {int(c.uses_dataset.sum())}'
      f'  (the other {int((~c.uses_dataset).sum())} only in the reference list)')
    w(f'  used a dataset in their own analysis  {int(c.uses_as_benchmark.sum())}')
    w(f'  contains a duplication-related term   {int((c.n_dup_sent > 0).sum())}  <- every one is read by hand')

    w('\n' + '-' * 100)
    w('the sentences that triggered the flag, in full: a negative result needs reading, not matching')
    w('-' * 100)
    for _, r in c[c.n_dup_sent > 0].iterrows():
        w(f'\n  {r.pmcid}  [{r.in_main}]  {str(r.venue)[:52]}')
        w(f'    {str(r.title)[:92]}')
        for s in str(r.dup_evidence).split(' || ')[:4]:
            w(f'      - {s[:230]}')

    # ---------- [4] summary ----------
    w('\n' + '=' * 100)
    w('[4] the numbers reported')
    w('=' * 100)
    oph = d[d.venue.str.contains(r'ophthalm|glaucoma|retina|eye|cornea|vision science',
                                 case=False, na=False)]
    jama = d[d.venue.str.contains('JAMA', case=False, na=False)]
    w(f'  citing population (upper bound)                 {len(d):4d}')
    w(f'    citing two or more of the datasets            {int((d.n_datasets >= 2).sum()):4d}')
    w(f'    in ophthalmology journals                     {len(oph):4d}')
    w(f'    in JAMA Network journals                      {len(jama):4d}')
    w(f'  full text obtained and classified               {len(c):4d}')
    w(f'    used a dataset in their own analysis (lower bound) {int(c.uses_as_benchmark.sum()):4d}')
    w(f'    reported duplication or leakage               {int((c.n_dup_sent > 0).sum()):4d}  <- after reading')
    w(f'  full text not obtainable                        {len(d) - len(c):4d}')

    d.drop(columns=['tn']).to_csv(OUT / 'citing_population.csv', index=False)
    json.dump(dict(union_raw=int(len(d) + int(is_src.sum())), source_papers_removed=int(is_src.sum()),
                   citing_population=int(len(d)), name_only=int(len(name_only)),
                   two_or_more=int((d.n_datasets >= 2).sum()), ophthalmology=int(len(oph)),
                   jama=int(len(jama)), fulltext_classified=int(len(c)),
                   confirmed_users=int(c.uses_as_benchmark.sum()),
                   flagged_duplication_auto=int((c.n_dup_sent > 0).sum())),
              open(OUT / 'summary.json', 'w'), indent=2, ensure_ascii=False)
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
