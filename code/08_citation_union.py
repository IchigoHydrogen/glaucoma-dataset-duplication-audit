"""Stage 08 — Retrieve every work that cites or names the three datasets, from 5 indexes.

What it produces
----------------
``citing_works_union.csv``  the merged citing population, with the index or indexes that found each

Why it is done this way
-----------------------
An earlier version of this audit used a full-text search of PubMed Central alone. That index
carries no full text from JAMA Network, Elsevier or IEEE journals, nor from preprint servers,
which are the venues where this literature appears; two of the three studies named in the
manuscript's own Results were invisible to it. An audit that cannot find the examples in its own
paper cannot be offered as evidence of scale. The superseded version is kept in
``superseded/citation_audit_pmc_only.py``.

Five indexes are therefore combined. Two are citation-based, retrieving every work that cites one
of the three dataset publications; this is what the editor's comment 9 asks for and it does not
depend on any publisher's indexing policy. Three are name-based, retrieving works that name a
dataset, which additionally catches use without citation. Records are merged on normalised DOI
and, where absent, on normalised title.

A caution on reproducibility: these are live bibliographic services. Re-running this stage later
will return a larger population as new work accumulates. The retrieved snapshot is shipped with
the archive so that the classification in stages 09 and 10 can be checked against the population
that was actually analysed.

Inputs
------
network access to OpenAlex, Semantic Scholar, Europe PMC, PubMed Central and arXiv

Runtime: about 10 minutes, rate-limited. No GPU.

Manuscript values reproduced
----------------------------
110 works in the union, 107 after removing the three dataset publications themselves; the
per-index contribution table in eMethods 7.
"""
import json
import sys
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'cache'
CACHE.mkdir(exist_ok=True)
MAIL = 's2422005@stu.musashino-u.ac.jp'
UA = {'User-Agent': f'paperB-dataset-audit/1.0 (mailto:{MAIL})'}

SOURCE_WORKS = {
    'Harvard-GDP': ['W4390874570', 'W4386228586'],
    'Harvard-GF': ['W4392903260', 'W4380994586'],
    'Harvard-FairVision': ['W4387389835'],
}
S2_IDS = {
    'Harvard-GDP': 'DOI:10.1109/ICCV51070.2023.01877',
    'Harvard-GF': 'DOI:10.1109/TMI.2024.3377552',
    'Harvard-FairVision': 'ARXIV:2310.02492',
}
NAME_QUERIES = ['"Harvard-GDP"', '"Harvard GDP"', '"Harvard Glaucoma Detection and Progression"',
                '"Harvard-GF"', '"Harvard GF"', '"Harvard Glaucoma Fairness"',
                '"Harvard-FairVision"', '"Harvard FairVision"', '"FairVision"']

NAME_RE = {
    'Harvard-GDP': re.compile(r'Harvard[-\s]?GDP\b|Harvard\s+Glaucoma\s+Detection\s+and\s+Progression', re.I),
    'Harvard-GF': re.compile(r'Harvard[-\s]?GF\b|Harvard\s+Glaucoma\s+Fairness', re.I),
    'Harvard-FairVision': re.compile(r'Harvard[-\s]?FairVision|\bFairVision\b', re.I),
}
USE_RE = re.compile(r'external(ly)?\s+valid|independent\s+(test|validation|cohort|dataset)|'
                    r'pre[-\s]?train|benchmark|fine[-\s]?tun|train(ed|ing)?\s+on|'
                    r'evaluat(ed|ion)\s+on|we\s+use[d]?\s+the', re.I)
DUP_RE = re.compile(r'duplicat|overlap|data\s+leak|leakage|same\s+patients?|shared\s+patients?|'
                    r'patient\s+overlap|identical\s+(image|scan|volume)|contaminat|'
                    r'non[-\s]?independen|not\s+independent', re.I)
REF_HEAD = re.compile(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b')
OPHTH = re.compile(r'ophthalm|glaucoma|retina|eye|cornea|IOVS|vision science|visual science', re.I)

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def fetch(url, headers=None, tries=5, raw=False):
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers=headers or UA)
            with urllib.request.urlopen(r, timeout=90) as x:
                b = x.read()
            return b.decode('utf-8', 'replace') if raw else json.loads(b)
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    w(f'    [could not fetch] {url[:96]} ... {last}')
    return None


def cached(key, fn):
    f = CACHE / (re.sub(r'[^A-Za-z0-9_.-]', '_', key)[:150] + '.json')
    if f.exists():
        return json.loads(f.read_text())
    v = fn()
    if v is not None:
        f.write_text(json.dumps(v, ensure_ascii=False))
    return v


def norm_doi(d):
    if not d:
        return ''
    return re.sub(r'^https?://(dx\.)?doi\.org/', '', str(d)).strip().lower()


def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())[:110]


# ------------------------------------------------------------ retrieval, index by index
def from_openalex():
    recs = {}
    for name, ids in SOURCE_WORKS.items():
        for wid in ids:
            cursor = '*'
            while cursor:
                u = (f'https://api.openalex.org/works?mailto={MAIL}&filter=cites:{wid}'
                     f'&per-page=200&cursor={urllib.parse.quote(cursor)}')
                d = cached(f'oa_{wid}_{cursor}', lambda u=u: fetch(u))
                if not d:
                    break
                for r in d['results']:
                    loc = (r.get('primary_location') or {}).get('source') or {}
                    key = r['id']
                    e = recs.setdefault(key, dict(
                        title=r.get('title') or '', year=r.get('publication_year'),
                        venue=loc.get('display_name') or '', doi=norm_doi(r.get('doi')),
                        pmid=((r.get('ids') or {}).get('pmid') or '').split('/')[-1],
                        datasets=set(), sources=set()))
                    e['datasets'].add(name)
                    e['sources'].add('OpenAlex')
                cursor = d['meta'].get('next_cursor')
                time.sleep(0.15)
    return list(recs.values())


def from_semanticscholar():
    recs = []
    for name, pid in S2_IDS.items():
        off = 0
        while True:
            u = (f'https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(pid, safe="")}'
                 f'/citations?fields=title,year,venue,externalIds&limit=100&offset={off}')
            d = cached(f's2_{pid}_{off}', lambda u=u: fetch(u))
            time.sleep(3.5)          # unauthenticated access is rate-limited; always wait
            if not d or 'data' not in d:
                break
            for c in d['data']:
                p = c.get('citingPaper') or {}
                ex = p.get('externalIds') or {}
                recs.append(dict(title=p.get('title') or '', year=p.get('year'),
                                 venue=p.get('venue') or '', doi=norm_doi(ex.get('DOI')),
                                 pmid=str(ex.get('PubMed') or ''),
                                 datasets={name}, sources={'SemanticScholar'}))
            if len(d['data']) < 100:
                break
            off += 100
    return recs


def from_europepmc():
    recs = []
    for q in NAME_QUERIES:
        cur = '*'
        while cur:
            u = ('https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
                {'query': q, 'format': 'json', 'pageSize': 100, 'resultType': 'core', 'cursorMark': cur}))
            d = cached(f'epmc_{q}_{cur}', lambda u=u: fetch(u))
            if not d:
                break
            for r in d.get('resultList', {}).get('result', []):
                nm = {k for k, rx in NAME_RE.items()
                      if rx.search((r.get('title') or '') + ' ' + (r.get('abstractText') or '') + ' ' + q)}
                recs.append(dict(title=r.get('title') or '', year=int(r.get('pubYear') or 0) or None,
                                 venue=r.get('journalTitle') or r.get('bookOrReportDetails', {}).get('publisher') or '',
                                 doi=norm_doi(r.get('doi')), pmid=str(r.get('pmid') or ''),
                                 datasets=nm or {'(name match)'}, sources={'EuropePMC'}))
            nxt = d.get('nextCursorMark')
            cur = nxt if nxt and nxt != cur else None
            time.sleep(0.3)
    return recs


def from_ncbi_pmc():
    recs = []
    q = ' OR '.join(NAME_QUERIES)
    u = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' + urllib.parse.urlencode(
        {'db': 'pmc', 'term': q, 'retmax': 400, 'retmode': 'json'}))
    d = cached('ncbi_pmc_ids', lambda: fetch(u))
    ids = d['esearchresult'].get('idlist', []) if d else []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        su = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?' + urllib.parse.urlencode(
            {'db': 'pmc', 'id': ','.join(chunk), 'retmode': 'json'}))
        s = cached(f'ncbi_sum_{i}', lambda su=su: fetch(su))
        time.sleep(0.4)
        if not s:
            continue
        for k, v in s.get('result', {}).items():
            if k == 'uids':
                continue
            doi = ''
            for a in v.get('articleids', []):
                if a.get('idtype') == 'doi':
                    doi = norm_doi(a.get('value'))
            recs.append(dict(title=v.get('title') or '', year=int((v.get('pubdate') or '0')[:4] or 0) or None,
                             venue=v.get('fulljournalname') or v.get('source') or '',
                             doi=doi, pmid='', pmcid=f'PMC{k}',
                             datasets={'(name match)'}, sources={'NCBI-PMC'}))
    return recs


def from_arxiv():
    recs = []
    q = ' OR '.join(f'all:{n}' for n in NAME_QUERIES)
    u = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(
        {'search_query': q, 'max_results': 200})
    t = cached('arxiv', lambda: {'xml': fetch(u, raw=True)})
    t = (t or {}).get('xml') or ''
    for e in re.findall(r'<entry>(.*?)</entry>', t, flags=re.S):
        ti = re.search(r'<title>(.*?)</title>', e, flags=re.S)
        yr = re.search(r'<published>(\d{4})', e)
        doi = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', e, flags=re.S)
        aid = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', e)
        summ = re.search(r'<summary>(.*?)</summary>', e, flags=re.S)
        blob = (ti.group(1) if ti else '') + ' ' + (summ.group(1) if summ else '')
        recs.append(dict(title=re.sub(r'\s+', ' ', ti.group(1)).strip() if ti else '',
                         year=int(yr.group(1)) if yr else None,
                         venue=f'arXiv:{aid.group(1)}' if aid else 'arXiv',
                         doi=norm_doi(doi.group(1)) if doi else '', pmid='',
                         datasets={k for k, rx in NAME_RE.items() if rx.search(blob)} or {'(name match)'},
                         sources={'arXiv'}))
    return recs


# ------------------------------------------------------------ merge
def merge(all_recs):
    by_doi, by_title, out = {}, {}, []
    for r in all_recs:
        r = dict(r)
        r['datasets'] = set(r.get('datasets') or [])
        r['sources'] = set(r.get('sources') or [])
        k = ('doi', r['doi']) if r['doi'] else ('t', norm_title(r['title']))
        idx = by_doi if k[0] == 'doi' else by_title
        if not k[1]:
            continue
        if k[1] in idx:
            e = idx[k[1]]
            e['datasets'] |= r['datasets']
            e['sources'] |= r['sources']
            for f in ('venue', 'year', 'pmid', 'pmcid'):
                if not e.get(f) and r.get(f):
                    e[f] = r[f]
        else:
            idx[k[1]] = r
            out.append(r)
    # Reconcile records that entered by DOI with those that entered by title, once.
    seen_t = {}
    final = []
    for r in out:
        t = norm_title(r['title'])
        if t and t in seen_t:
            e = seen_t[t]
            e['datasets'] |= r['datasets']
            e['sources'] |= r['sources']
            continue
        seen_t[t] = r
        final.append(r)
    return final


def main():
    w('=' * 100)
    w(f'{PREFIX}: downstream use, audited over the union of 5 bibliographic indexes')
    w('=' * 100)

    got = {}
    for label, fn in [('OpenAlex (citations of the 3 publications)', from_openalex),
                      ('Semantic Scholar (citations)', from_semanticscholar),
                      ('Europe PMC (name, metadata and open-access full text)', from_europepmc),
                      ('PubMed Central (name in full text)', from_ncbi_pmc),
                      ('arXiv (name in title or abstract)', from_arxiv)]:
        w(f'\n  retrieving: {label}')
        try:
            r = fn()
        except Exception as e:
            w(f'    failed: {e}')
            r = []
        got[label] = r
        w(f'    {len(r)} records')

    merged = merge([r for v in got.values() for r in v])
    w(f'\n  union after merging on DOI and title: {len(merged)} works')

    d = pd.DataFrame([dict(
        title=r['title'][:220], year=r.get('year'), venue=r.get('venue') or '(unindexed)',
        doi=r.get('doi', ''), pmid=r.get('pmid', ''), pmcid=r.get('pmcid', ''),
        datasets='; '.join(sorted(x for x in r['datasets'] if x != '(name match)')),
        n_datasets=len([x for x in r['datasets'] if x != '(name match)']),
        sources='; '.join(sorted(r['sources'])), n_sources=len(r['sources']),
    ) for r in merged]).sort_values(['n_sources', 'year'], ascending=[False, False])
    d.to_csv(OUT / 'citing_works_union.csv', index=False)

    w('\n' + '=' * 100)
    w('contribution of each index, and what a single-index audit would have missed')
    w('=' * 100)
    for label in got:
        tag = label.split('（')[0]
        key = {'OpenAlex': 'OpenAlex', 'SemanticScholar': 'SemanticScholar',
               'EuropePMC': 'EuropePMC', 'NCBI-PMC': 'NCBI-PMC', 'arXiv': 'arXiv'}[tag]
        n = int(d.sources.str.contains(key).sum())
        only = int((d.sources == key).sum())
        w(f'  {tag:16s} contributed {n:4d}   unique to it {only:4d}   '
          f'would have been missed alone {len(d) - n:4d}')

    w('\n' + '=' * 100)
    w('venues (top 30)')
    w('=' * 100)
    for v, c in Counter(d.venue).most_common(30):
        mark = '   ★ JAMA' if re.search(r'JAMA', v, re.I) else ('   <- ophthalmology journal' if OPHTH.search(v) else '')
        w(f'  {c:4d}  {v[:76]}{mark}')

    w('\n' + '=' * 100)
    w('works published in JAMA Ophthalmology or another JAMA Network journal')
    w('=' * 100)
    j = d[d.venue.str.contains('JAMA', case=False, na=False)]
    w(f'  {len(j)} works')
    for _, r in j.iterrows():
        w(f'    {r.year}  {r.venue}  |  {r.title[:78]}  |  doi:{r.doi}')

    w('\n' + '=' * 100)
    w('works published in ophthalmology journals, which reach clinical readers directly')
    w('=' * 100)
    o = d[d.venue.str.contains(OPHTH, na=False)]
    w(f'  {len(o)} works')
    for _, r in o.sort_values('year', ascending=False).iterrows():
        w(f'  {r.year}  {r.venue[:44]:44s} {r.title[:72]}')

    w('\n' + '=' * 100)
    w('works citing two or more of the datasets, where the overlap bites hardest')
    w('=' * 100)
    t2 = d[d.n_datasets >= 2]
    w(f'  {len(t2)} works')
    for _, r in t2.sort_values('year', ascending=False).iterrows():
        w(f'  {r.year}  [{r.datasets}]  {r.venue[:40]:40s} {r.title[:62]}')

    json.dump(dict(
        per_index={k: len(v) for k, v in got.items()},
        union=int(len(d)),
        index_only={k: int((d.sources == k).sum()) for k in
                    ['OpenAlex', 'SemanticScholar', 'EuropePMC', 'NCBI-PMC', 'arXiv']},
        in_jama=int(len(j)), in_ophthalmology=int(len(o)),
        two_or_more_datasets=int(len(t2)),
        with_pmcid=int((d.pmcid.fillna('') != '').sum()),
    ), open(OUT / 'summary.json', 'w'), indent=2, ensure_ascii=False)
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
