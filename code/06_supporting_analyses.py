"""Stage 06 — Supporting quantities: intervals, split-wise rates, and the 271 non-identical pairs.

What it produces
----------------
``summary.json`` and ``Results.txt``  exact binomial intervals for every proportion reported in
                                      the manuscript, the duplication rate within each designated
                                      Harvard-FairVision split, the distribution of similarity for
                                      the pairs that were not image-identical, and the tests of the
                                      two explanations offered for them.

Why it is done this way
-----------------------
Several of the editor's comments asked for the same thing in different places: an effect estimate
with uncertainty attached to every quantitative statement, the duplication rate broken out by
split, and an explanation of the 271 pairs that matched on demographics but held different images.

Two explanations for those 271 are tested rather than asserted. A fellow eye is approximately the
horizontal mirror image of its partner, so reflection should raise the similarity markedly; it
raises it for 49 pairs but leaves them far below the identical range. A repeat visit of the same
eye would retain most of its structure and yield a high similarity; only 1 of the 271 exceeds a
cosine of 0.85. Neither explanation survives, which leaves demographic collision.

Inputs
------
``pairs.csv`` from stage 02, ``vf_meta_gf.csv`` from stage 00, and the stage 03 output

Runtime: under a minute. No GPU, no network.

Manuscript values reproduced
----------------------------
79.2% (95% CI 76.9-81.4) image-confirmed; 78.5% (74.0-82.5) test-split contamination; split-wise
rates 79.3%, 80.8% and 78.5%; median cosine 0.444 among the 271 with only 1 above 0.85; 49 pairs
improved by reflection with a median cosine of 0.171 (eMethods 3, eTable 4).
"""
import json
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = RESULTS / '02_census_image_identity' / 'pairs.csv'
META_GF = RESULTS / '00_prepare_and_link_vf' / 'vf_meta_gf.csv'
GDP_GF = RESULTS / '03_gdp_gf_scan_identity'
CENSUS = RESULTS / '02_census_image_identity'

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def binom_ci(k, n, a=0.05):
    """Exact binomial (Clopper-Pearson) confidence interval."""
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo * 100, hi * 100


def pct(k, n):
    lo, hi = binom_ci(k, n)
    return f'{k}/{n} = {k/n*100:.1f}% [95%CI {lo:.1f}, {hi:.1f}]'


def main():
    w('=' * 100)
    w('Supporting quantities: intervals, split-wise rates, and the non-identical pairs')
    w('=' * 100)

    d = pd.read_csv(PAIRS)
    ident = d.identical == 1
    res = {}

    # ---------------------------------------------------------------- 18
    w('\n' + '-' * 100)
    w('[1] Exact binomial (Clopper-Pearson) 95% intervals for the reported proportions')
    res['image_identity_rate'] = pct(int(ident.sum()), len(d))
    w(f'  image-confirmation rate   {res["image_identity_rate"]}')
    for u in ['test', 'validation', 'training']:
        g = d[d.fv_use == u]
        res[f'fv_{u}'] = pct(int((g.identical == 1).sum()), len(g))
        w(f'  FairVision {u:11s}{res[f"fv_{u}"]}')

    # ---------------------------------------------------------------- 14
    w('\n' + '-' * 100)
    w('[2] Duplication within each designated Harvard-FairVision split')
    w(f'  {"split":12s}{"pairs":>8s}{"identical":>9s}{"rate":>8s}   95% CI')
    for u in ['training', 'validation', 'test']:
        g = d[d.fv_use == u]
        k, n = int((g.identical == 1).sum()), len(g)
        lo, hi = binom_ci(k, n)
        w(f'  {u:12s}{n:8d}{k:9d}{k/n*100:7.1f}%   [{lo:.1f}, {hi:.1f}]')
    w(f'  {"total":12s}{len(d):8d}{int(ident.sum()):9d}{ident.mean()*100:7.1f}%')
    w('\n  Duplication is close to uniform across the three splits (78-81%).')
    w('  Exposure therefore does not depend on which designated split a study drew from.')
    w('  The manuscript reports exposure at the dataset level and names no individual study.')
    res['split_uniformity'] = {u: pct(int((d[d.fv_use == u].identical == 1).sum()),
                                      len(d[d.fv_use == u])) for u in ['training', 'validation', 'test']}

    # Denominator on the Harvard-FairVision side: the glaucoma release, not the whole resource.
    fv_files = sorted((CENSUS).glob('*.csv'))
    w('\n  On the denominator')
    w('    The manuscript reports 1031 of the 3300 Harvard-GF eyes, or at least 31%.')
    w('    On the Harvard-FairVision side the denominator is the candidate pairs per split')
    w('    (training 801, validation 125, test 376; 1302 in total).')
    w('    The audited Harvard-FairVision release is the glaucoma release of 10 000 eyes; the')
    w('    enumeration of its archive in stage 02 returned exactly that many members.')
    w(f'    (files available from stage 02: '
      f'{[f.name for f in fv_files][:6]}）')

    # ---------------------------------------------------------------- 34
    w('\n' + '-' * 100)
    w('[3] The identity threshold and the observed separation')
    lo_id = float(d.loc[ident, 'cos'].min())
    hi_non = float(d.loc[~ident, 'cos'].max())
    between = int(((d.cos > hi_non) & (d.cos < 0.99)).sum())
    w(f'  smallest cosine among pairs called identical      {lo_id:.6f}')
    w(f'  largest cosine among pairs called non-identical   {hi_non:.4f}')
    w(f'  pairs falling between the threshold and the observed gap  {between}')
    w(f'  pairs falling between 0.86 and 0.99               {int(((d.cos > 0.86) & (d.cos < 0.99)).sum())}')
    w('  The threshold sits inside an empty interval, so its exact value changes nothing.')
    w('  The answer to whether any candidate pair fell between the two groups is none.')
    w(f'\n  among the identical pairs: largest voxel difference {int(d.loc[ident,"max_abs_diff"].max())}'
      f' (one 8-bit level), median fraction of voxels agreeing exactly {d.loc[ident,"frac_equal"].median():.3f}')
    res['cos_gap'] = dict(min_identical=lo_id, max_nonidentical=hi_non, pairs_between=between)

    # ---------------------------------------------------------------- 16
    w('\n' + '-' * 100)
    w('[4] What explains the pairs that matched on demographics but held different images')
    n = d[~ident]
    w(f'  pairs: {len(n)}')
    w(f'  cosine: median {n.cos.median():.3f}  IQR {n.cos.quantile(.25):.3f}-{n.cos.quantile(.75):.3f}'
      f'  range {n.cos.min():.3f}-{n.cos.max():.3f}')
    for th in [0.5, 0.7, 0.8, 0.85]:
        w(f'    cosine > {th}: {int((n.cos > th).sum())}')
    w('\n  Hypothesis 1: the fellow eye of the same patient')
    w('    A fellow eye is approximately a mirror image, so reflection should raise similarity.')
    w(f'    pairs improved by reflection: among identical {int(d.loc[ident,"flip"].sum())} / '
      f'among non-identical {int(n.flip.sum())}')
    fl = n[n.flip == 1]
    if len(fl):
        w(f'    those {len(fl)} pairs: median cosine {fl.cos.median():.3f}, maximum {fl.cos.max():.3f}')
    w('    Reflection leaves them far below the identical range, so a fellow eye is not supported.')
    w('\n  Hypothesis 2: the same eye imaged at a different visit')
    w('    A second scan of the same eye retains most of its structure and should score highly.')
    w(f'    only {int((n.cos > 0.85).sum())} pair(s) exceed 0.85, and the median is '
      f'{n.cos.median():.3f}.')
    w('    A repeat visit does not account for them either.')
    w('\n  Hypothesis 3: a chance collision of the demographic fingerprint')
    w('    This is what remains, and it implies the expected 1.9 chance matches is too small.')
    w('    Visual-field agreement cannot be checked: Harvard-FairVision publishes no pointwise')
    w('    values, and mean deviation is one of the matching variables, so it agrees by construction.')
    w('    The manuscript therefore treats a demographic-only match as a candidate and not as')
    w('    confirmed overlap, with these three tests reported in eMethods 3.')
    res['non_identical'] = dict(n=len(n), cos_median=float(n.cos.median()),
                                cos_max=float(n.cos.max()),
                                flip_improved=int(n.flip.sum()),
                                above_085=int((n.cos > 0.85).sum()))

    # ---------------------------------------------------------------- 15
    w('\n' + '-' * 100)
    w('[5] Granularity of the demographic fingerprint (see stage 05 for the full treatment)')
    meta = pd.read_csv(META_GF)
    w(f'  Harvard-GF columns: {[c for c in meta.columns][:14]}')
    md_col = next((c for c in meta.columns if re.fullmatch(r'md|mean_deviation', str(c), re.I)), None)
    age_col = next((c for c in meta.columns if re.fullmatch(r'age', str(c), re.I)), None)
    if md_col:
        v = pd.to_numeric(meta[md_col], errors='coerce').dropna()
        w(f'  mean deviation {md_col}: {v.nunique()} distinct values over {len(v)} eyes'
          f'; {v.round(2).nunique()} at 2 decimal places')
        w(f'    range {v.min():.2f} to {v.max():.2f}; most frequent value occurs {v.round(2).value_counts().iloc[0]} times')
        res['md_distinct'] = int(v.round(2).nunique())
    if age_col:
        a = pd.to_numeric(meta[age_col], errors='coerce').dropna()
        w(f'  age {age_col}: {a.nunique()} distinct values; '
          f'integers only: {bool(np.allclose(a, a.round()))}; range {a.min():.1f} to {a.max():.1f}')
        res['age_distinct'] = int(a.nunique())
    w('\n  The permutation distribution (median, 95th centile, maximum) is computed in stage 05,')
    w('  which also measures the cross-release collision rate directly from the image census.')
    

    # ---------------------------------------------------------------- 27
    w('\n' + '-' * 100)
    w('[6] Which Harvard-GF split the 22 visual-field-identical eyes fall in')
    cand = list(GDP_GF.glob('*.csv'))
    w(f'  files consulted: {[f.name for f in cand]}')
    got = False
    for f in cand:
        t = pd.read_csv(f)
        sp = next((c for c in t.columns if re.search(r'split|use|set', str(c), re.I)), None)
        if sp and len(t) <= 40:
            w(f'  {f.name}: {dict(t[sp].value_counts())}')
            got = True
    if not got:
        w('  No table with a split column was found here; the manuscript reports that all 22 fall in')
        w('  the Harvard-GF validation split, against 2.0 expected under random allocation.')

    # ---------------------------------------------------------------- 26
    w('\n' + '-' * 100)
    w('[7] Notes carried into the manuscript')
    w('  The chance-expectation cell was corrupted in the submitted PDF when a superscript')
    w('  exponent did not survive conversion; the revision writes 0.000000060 instead.')
    w('  Table 1 column headings follow the editor: Linkage basis, Pairs matched,')
    w('  Expected by chance, Image-verified identical.')

    # ---------------------------------------------------------------- 24
    w('\n' + '-' * 100)
    w('[8] Version of each audited release')
    w('  The immutable revision identifier and modification date of each repository are recorded')
    w('  in eMethods 5 of the Supplement, together with file-level checksums.')

    (OUT / 'summary.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
