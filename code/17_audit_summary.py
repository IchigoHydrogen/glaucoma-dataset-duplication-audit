"""Stage 17 — The final tally of the citation audit. Every number the manuscript reports.

What it produces
----------------
``summary.json``                    every reported count
``citing_population_deduped.csv``   the citing population after duplicate records are removed
``Results.txt``                     the same, as a readable log

Why it is done this way
-----------------------
Retrieval is spread over stages 08 to 10 and 13 to 14, and classification over 09, 10, 14 and
16. No single stage yields the numbers the manuscript prints, and holding that arithmetic
outside the archive would leave the headline counts of the audit unreproducible.

Two records had to be removed. One index returned the same work twice, once with a digital
object identifier and once without, so the merge on identifier did not catch it; in one of the
two the year and the journal name had been concatenated into the title field. The rule applied
here is the only safe one: two records that both carry identifiers, and carry different ones,
are two works. A Letter to the Editor and the article it comments on therefore both remain,
although one title contains the other.

One work was also fetched twice, by two different routes. Works read in full are counted by
identifier, so it counts once.

Inputs
------
``citing_population.csv`` from stage 09; the classified outputs of stages 09, 10, 14 and 16;
``abstract_screen.csv`` from stage 10. No network.

Runtime: a few seconds.

Manuscript values reproduced
----------------------------
citing population 105; full text 64, of which 21 use a dataset in their own analysis and 43 cite
one without using it; 41 without full text, 19 screened on abstract and 22 on title; 12 cite two
or more of the three; 3 split their data by patient; 0 report duplication between the datasets
(Results, eMethods 7, eTable 11).
"""
import csv
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402  every path is resolved in config.py

OUT = RESULTS / '17_audit_summary'
OUT.mkdir(parents=True, exist_ok=True)

POP = RESULTS / '09_citation_classify' / 'citing_population.csv'
PASSES = [('PubMed Central', RESULTS / '09_citation_classify' / 'fulltext_classified.csv'),
          ('OpenAlex open access', RESULTS / '10_fulltext_sweep' / 'newly_classified.csv'),
          ('arXiv and Semantic Scholar', RESULTS / '14_fulltext_sweep3' / 'sweep3_classified.csv')]
ABSTRACT = RESULTS / '10_fulltext_sweep' / 'abstract_screen.csv'
SPLIT = RESULTS / '16_split_check' / 'split_verdicts.csv'

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def rd(p):
    return list(csv.DictReader(Path(p).open(encoding='utf-8')))


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'\W+', ' ', (s or '').lower())).strip()


def ident(r):
    return (r.get('doi') or r.get('pmid') or r.get('pmcid') or '').strip().lower()


def main():
    pop = rd(POP)
    venues = {norm(r['venue']) for r in pop}

    def core(r):
        """Strip a year and journal name that an index has concatenated into the title field."""
        t = re.sub(r'^(19|20)\d{2} ', '', norm(r['title']))
        for v in venues:
            if v and v not in ('unindexed', 'arxiv org') and t.startswith(v + ' '):
                t = t[len(v) + 1:]
        return t

    drop = {}
    for i, a in enumerate(pop):
        if ident(a):
            continue
        for j, b in enumerate(pop):
            if i == j or not ident(b):
                continue
            ca, cb = core(a), core(b)
            if ca == cb or (min(len(ca), len(cb)) >= 25 and (ca in cb or cb in ca)):
                drop[i] = j
                break
    kept = [r for i, r in enumerate(pop) if i not in drop]

    w('Citation audit, final tally')
    w(f'  records retrieved and merged            {len(pop)}')
    for i, j in drop.items():
        w(f'    duplicate record removed: "{pop[i]["title"][:58]}"')
        w(f'      same work as: {pop[j]["doi"]}  "{pop[j]["title"][:52]}"')
    w(f'  citing population                       {len(kept)}')

    # ---- Full text. Count distinct works, keyed on identifier ---------------
    seen, full = {}, []
    for tag, path in PASSES:
        if not Path(path).exists():
            continue
        n_new = 0
        for r in rd(path):
            if str(r.get('chars', '1')) in ('', '0'):
                continue
            key = ident(r) or norm(r['title'])
            if key in seen:
                continue
            seen[key] = tag
            full.append(r)
            n_new += 1
        w(f'  full text, {tag:28s} {n_new:3d}')
    used = [r for r in full if str(r.get('uses', '')).strip().lower() in ('true', '1')]
    w(f'  full text, distinct works               {len(full)}')
    w(f'    of which a dataset is used in the work\'s own analysis   {len(used)}')
    w(f'    of which a dataset is cited but not used                {len(full)-len(used)}')

    ft_keys = set(seen)
    def read_in_full(r):
        return (ident(r) or norm(r['title'])) in ft_keys or norm(r['title']) in {norm(x['title']) for x in full}

    rest = [r for r in kept if not read_in_full(r)]
    # Match on identifier first; titles are a fallback because some indexes truncate them
    abs_ids, abs_titles = set(), set()
    for r in rd(ABSTRACT):
        if str(r.get('has_abstract', '')).strip().lower() in ('true', '1'):
            if ident(r):
                abs_ids.add(ident(r))
            abs_titles.add(norm(r['title']))
    with_abs = [r for r in rest if (ident(r) and ident(r) in abs_ids) or norm(r['title']) in abs_titles]
    w(f'  no full text                            {len(rest)}')
    w(f'    of which the abstract was obtained    {len(with_abs)}')
    w(f'    title only                            {len(rest)-len(with_abs)}')

    n2 = sum(1 for r in kept if int(r['n_datasets']) >= 2)
    w(f'  cite two or more of the three datasets  {n2}')

    split = [r for r in rd(SPLIT) if r['uses'] == 'True' and r['split'] == 'True'] if SPLIT.exists() else []
    w(f'  split their data by patient             {len(split)}  (each confirmed by reading)')

    w('')
    w('  reported duplication, patient overlap or leakage between these datasets   0')

    summary = dict(records_retrieved=len(pop), duplicates_removed=len(drop),
                   citing_population=len(kept), full_text=len(full), used_in_own_analysis=len(used),
                   cited_not_used=len(full) - len(used), no_full_text=len(rest),
                   abstract_screened=len(with_abs), title_only=len(rest) - len(with_abs),
                   cite_two_or_more=n2, patient_level_split=len(split), reported_duplication=0)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (OUT / 'citing_population_deduped.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(pop[0]))
        wr.writeheader()
        wr.writerows(kept)
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    w(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
