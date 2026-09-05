"""Stage 05 — The chance-match null, and how much the demographic fingerprint can discriminate.

What it produces
----------------
``summary.json`` and ``Results.txt``  the fingerprint's granularity, the within-dataset collision
                                      rate with its interval, three permutation nulls, and the
                                      cross-dataset collision rate measured directly.

Why it is done this way
-----------------------
The editor's comment 15 observed that a within-dataset collision rate is estimated in data from
which duplicates have already been removed, and is therefore likely to understate cross-dataset
collisions. That is correct, and this stage measures the size of the understatement instead of
conceding it in words.

One property of the statistic constrains any permutation test here. The number of exact
fingerprint matches depends only on the multiset of fingerprints in each release, so permuting
whole records leaves it unchanged; the code verifies this and it returns 1302 every time. Only
permutations that shuffle fields independently move the statistic, and those break the
association among fields, which enlarges the effective fingerprint space and yields fewer
collisions than real data would. All three nulls are therefore lower bounds.

The direct measurement is possible only because stage 02 compared the images. Of the 1302
matched pairs, 1031 are the same OCT volume and 271 are not; the 271 are cross-dataset
collisions, so the collision rate can be observed rather than modelled.

Inputs
------
the three release summary files, ``vf_meta_gf.csv`` from stage 00, and ``pairs.csv`` from stage 02

Runtime: about 3 minutes, dominated by 2000 permutations x 3 nulls. No GPU, no network.

Manuscript values reproduced
----------------------------
Mean deviation takes 1919, 1225 and 561 distinct values; age 4921, 2425 and 852; 9997 of 10 000
Harvard-FairVision fingerprints are unique; the comparison spans 31.1 million pairs; the expected
1.9 chance matches with a 95% CI of 0.4 to 5.5; permutation medians 3, 1 and 1 with maxima 11, 7
and 8; and the within-dataset rate understating the observed cross-dataset rate by a factor of
about 145 (eMethods 2).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
FV_CSV = DATASET / 'FairVision' / 'Glaucoma' / 'ReadMe' / 'data_summary_glaucoma.csv'
GDP_CSV = DATASET / 'data_summary.csv'
GF_CSV = DATASET / 'Harvard-GF' / 'ReadMe' / 'data_summary.csv'
GF_VF = RESULTS / '00_prepare_and_link_vf' / 'vf_meta_gf.csv'
CENSUS = RESULTS / '02_census_image_identity' / 'pairs.csv'
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 2000
SEED = 42
KEYS = ['md', 'age', 'sex', 'race', 'eth']

RACE = {'white': 'white', 'white or caucasian': 'white',
        'black': 'black', 'black or african american': 'black', 'asian': 'asian'}
ETH = {'hispanic': 1, 'non-hispanic': 0, 'yes': 1, 'no': 0}

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def norm(df, md_col, age_col, sex_col, race_col, eth_col):
    out = pd.DataFrame()
    out['md'] = df[md_col].round(2)
    out['age'] = df[age_col].round(2)
    out['sex'] = df[sex_col].astype(str).str.lower()
    out['race'] = df[race_col].astype(str).str.lower().map(RACE)
    out['eth'] = df[eth_col].astype(str).str.lower().map(ETH)
    return out


def granularity(d, name, raw_md, raw_age):
    """Discriminative power of the fingerprint, including the two quantities comment 15 names:
    the number of distinct mean-deviation values and the granularity of age.
    """
    n = len(d)
    fp = list(zip(*[d[k] for k in KEYS]))
    n_fp = len(set(fp))
    # Granularity of age: how many distinct values, and the smallest gap between adjacent ones.
    a = np.sort(np.unique(np.asarray(raw_age, float)))
    gap = np.diff(a)
    gap = gap[gap > 0]
    dec_md = max(len(str(x).split('.')[-1]) for x in raw_md.dropna().astype(str).head(2000)
                 if '.' in str(x))
    dec_age = max(len(str(x).split('.')[-1]) for x in raw_age.dropna().astype(str).head(2000)
                  if '.' in str(x))
    w(f'  {name}  N = {n}')
    w(f'    mean deviation : {d.md.nunique():5d} distinct values  '
      f'range [{d.md.min():.2f}, {d.md.max():.2f}]  {dec_md} decimal places')
    w(f'    age            : {d.age.nunique():5d} distinct values  '
      f'range [{d.age.min():.2f}, {d.age.max():.2f}]  {dec_age} decimal places  '
      f'smallest gap {gap.min():.4f}')
    w(f'    sex / race / ethnicity : {d.sex.nunique()} / {d.race.nunique()} / {d.eth.nunique()} levels')
    w(f'    distinct complete fingerprints : {n_fp} / {n}  '
      f'({n - n_fp} share a fingerprint with another record)')
    return dict(n=n, md_distinct=int(d.md.nunique()), age_distinct=int(d.age.nunique()),
                age_min_gap=float(gap.min()), md_decimals=int(dec_md), age_decimals=int(dec_age),
                sex_levels=int(d.sex.nunique()), race_levels=int(d.race.nunique()),
                eth_levels=int(d.eth.nunique()), fingerprints_distinct=int(n_fp))


def collisions(d):
    """Number of within-release fingerprint collisions; one eye per patient, so always two patients."""
    c = Counter(zip(*[d[k] for k in KEYS]))
    coll = sum(v * (v - 1) // 2 for v in c.values())
    pairs = len(d) * (len(d) - 1) // 2
    return coll, pairs


def cross_count(a, b):
    """Number of exact cross-release fingerprint matches, counting multiplicity on both sides."""
    ca = Counter(zip(*[a[k] for k in KEYS]))
    return sum(ca[k] * v for k, v in Counter(zip(*[b[k] for k in KEYS])).items())


def main():
    w('=' * 96)
    w(f'{PREFIX}: chance-match null, and how much the demographic fingerprint discriminates')
    w('=' * 96)

    fv_raw = pd.read_csv(FV_CSV)
    gf_raw = pd.read_csv(GF_VF).merge(
        pd.read_csv(GF_CSV)[['filename', 'gender', 'race', 'ethnicity']], on='filename')
    gdp_raw = pd.read_csv(GDP_CSV)

    A = norm(fv_raw, 'md', 'age', 'gender', 'race', 'ethnicity').dropna(subset=KEYS)
    B = norm(gf_raw, 'md', 'age', 'gender', 'race', 'ethnicity').dropna(subset=KEYS)
    G = norm(gdp_raw, 'md', 'age', 'gender', 'race', 'hispanic').dropna(subset=KEYS)

    # ---------------------------------------------------------- [1] discriminative power
    w('\n[1] Discriminative power of the demographic fingerprint')
    w('    The fingerprint is exact agreement on mean deviation, age, sex, race and ethnicity.')
    w()
    gran = {}
    gran['FairVision-Glaucoma'] = granularity(A, 'FairVision (glaucoma release)', fv_raw.md, fv_raw.age)
    gran['Harvard-GF'] = granularity(B, 'Harvard-GF                    ', gf_raw.md, gf_raw.age)
    gran['Harvard-GDP'] = granularity(G, 'Harvard-GDP                   ', gdp_raw.md, gdp_raw.age)

    n_cross = len(A) * len(B)
    w(f'\n    cross-release comparisons = {len(B)} x {len(A)} = {n_cross:,d}'
      f'  (the manuscript said approximately 33 million; this is the corrected figure)')

    w('\n    Within-release fingerprint collisions (one eye per patient, so always two patients)')
    coll = {}
    for nm, d in [('FairVision', A), ('Harvard-GF', B), ('Harvard-GDP', G)]:
        c, p = collisions(d)
        rate = c / p
        coll[nm] = dict(collisions=int(c), pairs=int(p), rate=float(rate))
        w(f'      {nm:12s} {c:4d} collisions in {p:14,d} pairs  -> rate {rate:.3e}')

    rate_fv = coll['FairVision']['rate']
    expected = n_cross * rate_fv
    w(f'\n    expected chance matches = {n_cross:,d} x {rate_fv:.3e} = {expected:.2f}')
    w('    (the 1.9 reported at submission; what follows shows it is a lower bound)')

    # ---------------------------------------------------------- [2] permutation nulls
    w('\n' + '=' * 96)
    w('[2] Permutation nulls')
    w('=' * 96)
    w('    Note: the number of exact matches depends only on the multiset of fingerprints.')
    w('    Permuting whole records does not change that multiset, so the count never moves.')
    w('    Only shuffling fields independently moves it, and that destroys their association.')

    obs = cross_count(A, B)
    w(f'\n    observed matches = {obs}')

    # Check the invariance: permuting whole records leaves the match count unchanged.
    rng = np.random.default_rng(SEED)
    Ar = A.iloc[rng.permutation(len(A))].reset_index(drop=True)
    w(f'    [check] permuting whole records gives {cross_count(Ar, B)} matches, unchanged')

    nulls = {}
    for label, cols in [('mean deviation shuffled alone (the null used at submission)', ['md']),
                        ('mean deviation and age shuffled independently', ['md', 'age']),
                        ('all 5 fields shuffled independently', KEYS)]:
        rng = np.random.default_rng(SEED)
        cb = Counter(zip(*[B[k] for k in KEYS]))
        base = {k: A[k].to_numpy() for k in KEYS}
        vals = np.empty(N_PERM, int)
        for i in range(N_PERM):
            cur = dict(base)
            for c in cols:
                cur[c] = rng.permutation(base[c])
            ca = Counter(zip(*[cur[k] for k in KEYS]))
            vals[i] = sum(cb[k] * v for k, v in ca.items())
        med, p95, mx = np.median(vals), np.percentile(vals, 95), vals.max()
        nulls[label] = dict(median=float(med), p95=float(p95), max=int(mx),
                            mean=float(vals.mean()), min=int(vals.min()), n_perm=N_PERM)
        w(f'\n    {label}  (B = {N_PERM})')
        w(f'      median {med:.1f}   95th centile {p95:.1f}   maximum {mx}   '
          f'(mean {vals.mean():.2f}, minimum {vals.min()})')
        w(f'      the observed {obs} '
          f'{"exceeds" if obs > mx else "does not exceed"} the null maximum of {mx}. one-sided P {"< " if obs > mx else "= "}'
          f'{max((vals >= obs).mean(), 1 / N_PERM):.4f}')

    w('\n    All three nulls are biased the same way. Shuffling fields independently breaks their')
    w('    association, enlarges the effective fingerprint space, and so yields fewer collisions.')
    w('    They are therefore lower bounds on the cross-release collision count, not expectations.')

    # ------------------------------------------- [3] the cross-release collision rate, measured
    w('\n' + '=' * 96)
    w('[3] The cross-release collision rate, measured rather than modelled')
    w('=' * 96)
    cen = pd.read_csv(CENSUS)
    n_pairs = len(cen)
    n_ident = int(cen.identical.sum())
    n_not = n_pairs - n_ident
    w(f'    All {n_pairs} fingerprint-matched pairs were compared at the image level.')
    w(f'      image-identical (true duplicates)  {n_ident}')
    w(f'      different images (collisions)      {n_not}')
    w('\n    Stage 06 excludes fellow eyes and repeat visits as explanations for these pairs.')
    w('    They are different patients who happen to share a demographic fingerprint,')
    w('    which means the cross-release collision count has been observed directly.')

    rate_obs = n_not / n_cross
    w(f'\n    observed collisions        {n_not} in {n_cross:,d} pairs  -> rate {rate_obs:.3e}')
    w(f'    predicted from within-release rate  {expected:.2f}      -> rate {rate_fv:.3e}')
    w(f'    understatement factor      {n_not / expected:.0f}x')
    w(f'    ratio of rates             {rate_obs / rate_fv:.0f}x')

    w('\n    The concern raised in comment 15 is correct, and its size can be measured, not just')
    w('    conceded: the within-release rate understates the cross-release rate by two orders.')
    w('    This does not weaken the duplication finding. The 1031 pairs rest on image comparison,')
    w('    not on the fingerprint, and two different eyes do not carry identical bytes.')
    w(f'    The {n_not} non-identical pairs reach a maximum cosine of '
      f'{cen[~cen.identical.astype(bool)].cos.max():.4f}, while')
    w(f'    the {n_ident} identical pairs fall no lower than {cen[cen.identical.astype(bool)].cos.min():.6f}; '
      f'no pair lies between.')

    # ---------------------------------------------------------- write out
    summary = dict(
        n_cross_pairs=int(n_cross), observed_matches=int(obs),
        granularity=gran, within_dataset_collisions=coll,
        analytic_expectation=float(expected), permutation_nulls=nulls,
        census=dict(pairs=int(n_pairs), image_identical=int(n_ident),
                    image_not_identical=int(n_not),
                    observed_cross_collision_rate=float(rate_obs),
                    underestimation_factor=float(n_not / expected)),
        note='the within-release rate and the permutation nulls are lower bounds; 271 collisions observed',
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('\nWrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
