"""Stage 13 — Ask whether any full text remains reachable by a route not yet tried.

What it produces
----------------
``availability.csv``  one row per work still without full text, with the route that reaches it

Why it is done this way
-----------------------
Stages 09 and 10 looked in PubMed Central and then in the open-access locations recorded by
OpenAlex. Two routes were never tried: arXiv itself, which serves a PDF for any identifier, and
the open-access link held by Semantic Scholar. Neither is exotic, and roughly a third of the
works still unread were preprints, so the question was worth asking before the audit was
reported as complete.

This stage only asks whether a PDF exists and is reachable. It downloads nothing and classifies
nothing; stage 14 does that. Separating the two keeps a slow, network-bound probe out of the
stage whose output the manuscript depends on.

No language model is asked whether an unread paper mentions duplication. That would be a
recollection rather than a reading, and a manuscript about unverified computational results
cannot rest on one.

Inputs
------
``citing_population.csv`` from stage 09; ``fulltext_classified.csv`` and ``newly_classified.csv``
to know what has already been read; network access to arXiv, Unpaywall and Semantic Scholar

Runtime: about 8 minutes, most of it waiting on rate limits. No GPU.

Manuscript values reproduced
----------------------------
16 works reachable by these two routes, which stage 14 then reads.
"""
import csv
import sys
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402  every path is resolved in config.py

OUT = RESULTS / '13_fulltext_probe3'
OUT.mkdir(parents=True, exist_ok=True)
MAIL = 's2422005@stu.musashino-u.ac.jp'
UA = {'User-Agent': f'paperB-audit/1.0 (mailto:{MAIL})'}


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'\W+', ' ', (s or '').lower())).strip()


def read(p):
    return list(csv.DictReader(p.open(encoding='utf-8')))


pop = read(RESULTS / '09_citation_classify' / 'citing_population.csv')
done = {norm(r.get('title', '')) for r in
        read(RESULTS / '09_citation_classify' / 'fulltext_classified.csv')}
done |= {norm(r.get('title', '')) for r in
         read(RESULTS / '10_fulltext_sweep' / 'newly_classified.csv')}

# Drop a record without an identifier that names a work already present with one.
# Two records that both carry identifiers, and carry different ones, are two works.
def ident(r):
    return (r['doi'] or r['pmid'] or r['pmcid'] or '').strip().lower()


def core(r):
    t = norm(r['title'])
    t = re.sub(r'^(19|20)\d{2} ', '', t)
    for o in pop:
        v = norm(o['venue'])
        if v and v not in ('unindexed', 'arxiv org') and t.startswith(v + ' '):
            t = t[len(v) + 1:]
    return t


dup = set()
for i, a in enumerate(pop):
    if ident(a):
        continue
    ca = core(a)
    for j, b in enumerate(pop):
        if i != j and ident(b):
            cb = core(b)
            if ca == cb or (min(len(ca), len(cb)) >= 25 and (ca in cb or cb in ca)):
                dup.add(i)
                break

pending = [r for i, r in enumerate(pop) if i not in dup and norm(r['title']) not in done]
print(f'{len(pop)} records retrieved, {len(dup)} duplicates removed, {len(pop)-len(dup)} works')
print(f'{len(done)} already read in full; probing the remaining {len(pending)}\n')


def get_json(url, timeout=30):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as f:
            return json.load(f)
    except Exception:
        return None


def head_ok(url, timeout=25):
    """Ask only whether a PDF is served at the URL. The body is not downloaded."""
    try:
        req = urllib.request.Request(url, headers=UA, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as f:
            ctype = f.headers.get('Content-Type', '')
            head = f.read(2048)
        return (b'%PDF' in head[:1024]) or ('pdf' in ctype.lower()), ctype
    except Exception as e:
        return False, str(e)[:40]


def arxiv_id(r):
    for field in (r.get('doi', ''), r.get('venue', '')):
        m = re.search(r'(?:arxiv[.:/]|abs/)\s*(\d{4}\.\d{4,5})', field or '', re.I)
        if m:
            return m.group(1)
    return None


rows = []
for k, r in enumerate(pending, 1):
    doi = (r['doi'] or '').strip()
    found, route, url = False, '', ''

    aid = arxiv_id(r)
    if aid:
        u = f'https://arxiv.org/pdf/{aid}'
        ok, _ = head_ok(u)
        if ok:
            found, route, url = True, 'arXiv direct', u
        time.sleep(0.6)

    if not found and doi.startswith('10.'):
        j = get_json(f'https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={MAIL}')
        loc = (j or {}).get('best_oa_location') or {}
        u = loc.get('url_for_pdf') or loc.get('url')
        if u:
            ok, _ = head_ok(u)
            if ok:
                found, route, url = True, 'Unpaywall', u
        time.sleep(0.6)

    if not found and doi.startswith('10.'):
        j = get_json('https://api.semanticscholar.org/graph/v1/paper/DOI:'
                     + urllib.parse.quote(doi) + '?fields=openAccessPdf')
        u = ((j or {}).get('openAccessPdf') or {}).get('url')
        if u:
            ok, _ = head_ok(u)
            if ok:
                found, route, url = True, 'SemanticScholar', u
        time.sleep(3.5)

    rows.append(dict(title=r['title'][:110], year=r['year'], venue=r['venue'],
                     doi=doi, available=int(found), route=route, url=url))
    print(f'  {k:>3}/{len(pending)}  {"reachable    " if found else "not reachable"} {route:16s} {r["title"][:58]}',
          flush=True)

with (OUT / 'availability.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

n = sum(r['available'] for r in rows)
print(f'\nfull text newly reachable: {n} of {len(pending)}')
from collections import Counter
print('  by route:', dict(Counter(r['route'] for r in rows if r['available'])))
print(f'  full text would rise from {len(done)} to {len(done)+n}')
print(f'\nWrote {OUT/"availability.csv"}')
