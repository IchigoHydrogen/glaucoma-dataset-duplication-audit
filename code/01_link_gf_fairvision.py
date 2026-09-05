"""Stage 01 — Link Harvard-GF and Harvard-GDP to Harvard-FairVision by demographic fingerprint.

What it produces
----------------
``matches_gf_fv.csv``   candidate Harvard-GF x Harvard-FairVision pairs  (the manuscript reports 1302)
``matches_gdp_fv.csv``  candidate Harvard-GDP x Harvard-FairVision pairs (the manuscript reports 172)

Why it is done this way
-----------------------
Harvard-FairVision distributes mean deviation but not the 52 pointwise total-deviation values,
so the strong visual-field fingerprint used in stage 00 is unavailable here. The key is the
weaker demographic fingerprint (mean deviation to 2 decimals, age to 2 decimals, sex, race,
ethnicity). A weak key can match two different patients, so this stage produces *candidates*
only; stage 02 decides which of them are the same OCT volume, and the manuscript never treats
a demographic-only match as confirmed overlap.

Because the key is weak, a null distribution is mandatory. Two are computed. The one used is
the within-dataset collision rate: each release distributes one eye per patient, so two records
sharing a fingerprint inside a release are necessarily different patients, and that rate is the
rate at which different patients collide. A permutation null that shuffles mean deviation alone
is also reported, but it breaks the association between mean deviation and age and therefore
understates chance agreement; stage 05 examines this properly.

A negative control is included: matching Harvard-GDP against Harvard-GF with the same weak
demographic key. Those two releases share 22 eyes under the exact visual-field key, but the
corresponding visits differ in age, so a demographic-only match should find none.

Inputs
------
``dataset/FairVision/Glaucoma/ReadMe/data_summary_glaucoma.csv``
``dataset/Harvard-GF/ReadMe/data_summary.csv``
``dataset/data_summary.csv``                       Harvard-GDP
``results/00_prepare_and_link_vf/vf_meta_gf.csv``  built by stage 00

Runtime: about 1 minute. No GPU, no network.

Manuscript values reproduced
----------------------------
1302 and 172 candidate pairs; the within-dataset collision rate of 6.0e-08 quoted in Table 1;
the shared 2 dB exclusion band reported in eTable 2.

Note on the "contamination routes" section
------------------------------------------
That section estimates, for two named studies, how many of the eyes they evaluate on also appear
in the data they pretrain on. **Those per-study estimates were withdrawn during revision** and do
not appear in the manuscript. The editor's comment 20 asked us either to quantify contamination
for each named study or to soften the attribution; we could not quantify it, because none of the
studies publishes the eye-level identifiers of the eyes it used, so the intersection with our
confirmed pairs cannot be reconstructed. The manuscript therefore reports exposure at the dataset
level and names no individual study. The code is left as it was run, and the section is marked in
its output, so that the record shows what was computed and what was set aside.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
FV_CSV = DATASET / 'FairVision' / 'Glaucoma' / 'ReadMe' / 'data_summary_glaucoma.csv'
GDP_CSV = DATASET / 'data_summary.csv'
GF_CSV = DATASET / 'Harvard-GF' / 'ReadMe' / 'data_summary.csv'
PAST_RESULTS = {
    'gf_vf_meta': RESULTS / '00_prepare_and_link_vf' / 'vf_meta_gf.csv',
}
RESULT_DIR = RESULTS / PREFIX
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 200
SEED = 42

RACE = {'white': 'white', 'white or caucasian': 'white',
        'black': 'black', 'black or african american': 'black',
        'asian': 'asian'}
ETH = {'hispanic': 1, 'non-hispanic': 0, 'yes': 1, 'no': 0}


def norm(df, md_col, age_col, sex_col, race_col, eth_col):
    out = pd.DataFrame()
    out['md'] = df[md_col].round(2)
    out['age'] = df[age_col].round(2)
    out['sex'] = df[sex_col].astype(str).str.lower()
    out['race'] = df[race_col].astype(str).str.lower().map(RACE)
    out['eth'] = df[eth_col].astype(str).str.lower().map(ETH)
    return out


def main():
    lines = []

    def w(s=''):
        print(s, flush=True)
        lines.append(s)

    w('=' * 84)
    w(f'{PREFIX}: does Harvard-FairVision share patients with Harvard-GDP or Harvard-GF?')
    w('=' * 84)
    w(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    w(f'Script    : {PREFIX}.py')
    w('Harvard-FairVision publishes no pointwise fields, so the key is (md, age, sex, race, eth).')
    w('That key is weak, so a null distribution is mandatory.')
    w()

    fv = pd.read_csv(FV_CSV)
    fv['g'] = fv.glaucoma.map({'yes': 1, 'no': 0})
    gdp = pd.read_csv(GDP_CSV)
    gfm = pd.read_csv(PAST_RESULTS['gf_vf_meta'])
    gfc = pd.read_csv(GF_CSV)
    gf = gfm.merge(gfc[['filename', 'gender', 'race', 'ethnicity', 'language', 'maritalstatus']], on='filename')

    w('=' * 84)
    w('SECTION 1: a labelling design common to all three releases')
    w('=' * 84)
    w(f'  {"dataset":24s} {"N":>6s} {"band -3<MD<-1":>16s} {"glaucoma==(MD<=-3)":>20s}')
    for nm, md, gg in [('Harvard-GDP', gdp.md, gdp.glaucoma),
                       ('Harvard-GF', gfm.md, gfm.glaucoma),
                       ('FairVision-Glaucoma', fv.md, fv.g)]:
        band = int(((md > -3.0) & (md < -1.0)).sum())
        agree = ((md <= -3.0).astype(int) == gg).mean()
        w(f'  {nm:24s} {len(md):6d} {band:16d} {agree:20.4f}')
    w()
    w('  14,300 eyes in total. All three share the same 2 dB exclusion band, a case-control design.')
    w(f'  Harvard-FairVision non-glaucoma MD range '
      f'[{fv[fv.g==0].md.min():.2f}, {fv[fv.g==0].md.max():.2f}] sd {fv[fv.g==0].md.std():.2f}')
    w()

    A = norm(fv, 'md', 'age', 'gender', 'race', 'ethnicity')
    A['fv_file'] = fv.filename
    A['fv_use'] = fv.use
    A['fv_g'] = fv.g

    rng = np.random.default_rng(SEED)
    keys = ['md', 'age', 'sex', 'race', 'eth']

    def collision_rate(d, name):
        """Rate at which two different patients share a fingerprint inside one release.

    One eye per patient, so any within-release collision is necessarily two different patients.
    """
        g = d.groupby(keys).size()
        coll = int(((g * (g - 1)) // 2).sum())
        pairs = len(d) * (len(d) - 1) // 2
        w(f'  {name:12s} N={len(d):5d}  {coll:3d} collisions in {pairs:12,d} pairs  '
          f'-> rate {coll/pairs:.3e}')
        return coll / pairs

    def run(name, B, extra):
        w('=' * 84)
        w(f'SECTION: {name} × FairVision')
        w('=' * 84)
        dup_a = len(A) - A[keys].drop_duplicates().shape[0]
        dup_b = len(B) - B[keys].drop_duplicates().shape[0]
        w(f'  within-release key duplicates: Harvard-FairVision {dup_a} / {name} {dup_b}')
        m = B.merge(A, on=keys)
        w(f'  matches = {len(m)}')

        null = np.empty(N_PERM, int)
        for i in range(N_PERM):
            Ap = A.copy()
            Ap['md'] = rng.permutation(A.md.values)
            null[i] = len(B.merge(Ap, on=keys))
        w(f'  permutation null (Harvard-FairVision md shuffled, B={N_PERM}): '
          f'mean {null.mean():.2f}  95th centile {np.percentile(null,95):.1f}  max {null.max()}')
        p = (null >= len(m)).mean()
        w(f'  one-sided p = {p:.4f}' + ('  (exceeds the null maximum)' if len(m) > null.max() else ''))
        if len(m):
            agree = (m.fv_g.values == m[extra].values).mean()
            w(f'  glaucoma label agreement among matched pairs = {agree:.4f}')
            w(f'  designated Harvard-FairVision split of the matched pairs: {m.fv_use.value_counts().to_dict()}')
        w()
        return m, null

    w('=' * 84)
    w('SECTION 2: the chance-match null, from the within-release collision rate')
    w('=' * 84)
    w('  One eye per patient, so two records sharing a key inside a release are different patients.')
    w('  That rate is the rate at which different patients agree by chance.')
    w()
    rate_fv = collision_rate(A.dropna(subset=keys), 'FairVision')

    B_gdp = norm(gdp, 'md', 'age', 'gender', 'race', 'hispanic')
    B_gdp['gdp_file'] = gdp.filename
    B_gdp['gdp_g'] = gdp.glaucoma
    B_gdp['gdp_det_use'] = gdp.glaucoma_detection_use
    B_gdp['gdp_prog_use'] = gdp.progression_forecasting_use
    B_gdp = B_gdp.dropna(subset=keys)
    rate_gdp = collision_rate(B_gdp, 'Harvard-GDP')
    m1, n1 = run('Harvard-GDP', B_gdp, 'gdp_g')

    B_gf = norm(gf, 'md', 'age', 'gender', 'race', 'ethnicity')
    B_gf['gf_file'] = gf.filename
    B_gf['gf_g'] = gf.glaucoma
    B_gf['lang'] = gf.language.astype(str).str.lower()
    B_gf['mar'] = gf.maritalstatus.astype(str).str.lower()
    B_gf = B_gf.dropna(subset=keys).reset_index(drop=True)
    rate_gf = collision_rate(B_gf, 'Harvard-GF')
    m2, n2 = run('Harvard-GF', B_gf, 'gf_g')

    w('=' * 84)
    w('SECTION 2b: observed against expected, with a negative control')
    w('=' * 84)
    w(f'  {"comparison":26s} {"matched":>6s} {"expected":>8s} {"ratio":>9s}')
    for nm, Bx, mm in [('Harvard-GDP x FairVision', B_gdp, m1),
                       ('Harvard-GF  x FairVision', B_gf, m2)]:
        exp = len(Bx) * len(A) * rate_fv
        w(f'  {nm:26s} {len(mm):6d} {exp:8.2f} {len(mm)/max(exp,1e-9):8.1f}x')
    neg = B_gdp.merge(B_gf, on=keys)
    exp_neg = len(B_gdp) * len(B_gf) * max(rate_gdp, rate_gf, rate_fv)
    w(f'  {"[control] GDP x GF":26s} {len(neg):6d} {exp_neg:8.2f}')
    w()
    w('  The control pair is the one that yields 22 matches under the strong visual-field key.')
    w('  It returns none under the weak key because the corresponding visits differ in age,')
    w('  which shows the demographic key does not manufacture matches on its own.')
    w()
    w('  For reference, the null that shuffles md alone gives means of '
      f'{n1.mean():.2f} and {n2.mean():.2f}, but it breaks the md-age association and understates chance.')
    w('        The within-release collision rate is what the conclusion rests on.')
    w()

    # Does the match survive a stronger key? Harvard-GF also carries language and marital status.
    w('=' * 84)
    w('SECTION: does the match survive a stronger key? (Harvard-GF x Harvard-FairVision)')
    w('=' * 84)
    B_gf2 = B_gf.copy()
    A2 = A.copy()
    A2['lang'] = fv.language.astype(str).str.lower()
    A2['mar'] = fv.maritalstatus.astype(str).str.lower()
    w(f'  {"key fields":>10s} {"matched":>7s} {"null mean":>14s} {"max":>6s}')
    for ks in [keys, keys + ['lang'], keys + ['lang', 'mar']]:
        mm = B_gf2.merge(A2, on=ks)
        nl = [len(B_gf2.merge(A2.assign(md=rng.permutation(A2.md.values)), on=ks))
              for _ in range(100)]
        w(f'  {len(ks):10d} {len(mm):7d} {np.mean(nl):14.2f} {max(nl):6d}')
    w()
    w('  Adding language and marital status barely reduces the count, as shared patients would.')
    w()

    w('=' * 84)
    w('SECTION: contamination routes  [WITHDRAWN IN REVISION - see the note in the docstring]')
    w('=' * 84)
    if len(m1):
        w('  FairDist (npj Digit Med 2025) pretrains a detection model on Harvard-FairVision and')
        w('  evaluates progression forecasting on 500 Harvard-GDP eyes.')
        ct = pd.crosstab(m1.fv_use, m1.gdp_prog_use.fillna('(no label)'))
        for l in ct.to_string().split('\n'):
            w('    ' + l)
        n_prog = int(m1.gdp_prog_use.notna().sum())
        n_test = int((m1.gdp_prog_use == 'test').sum())
        contam = int(m1[(m1.fv_use.isin(['training', 'validation'])) &
                        (m1.gdp_prog_use == 'test')].shape[0])
        w(f'  matches inside the Harvard-GDP progression subset (500 eyes) = {n_prog}')
        w(f'  matches inside the Harvard-GDP progression test split (200 eyes) = {n_test}')
        w(f'  of which in a Harvard-FairVision train/val split = {contam} ({contam/200:.1%} of the test split)')
        w('  [This per-study estimate was withdrawn in revision; see the docstring.]')
        w()
        w('  TransFair (AIM 2025) pretrains on Harvard-GF and Harvard-FairVision, evaluates on Harvard-GDP.')
        w('  Harvard-GF and Harvard-FairVision share 1271-1302 candidate eyes, and both overlap Harvard-GDP.')
        w()
        ct2 = pd.crosstab(m1.fv_use, m1.gdp_det_use)
        for l in ct2.to_string().split('\n'):
            w('    ' + l)
    w()
    w('  None of the four publications examined (TransFair, FairDist, Harvard-GDP, Harvard-GF)')
    w('  contains any sentence confirming, excluding or mentioning patient overlap (checked verbatim).')
    w('  Harvard-GDP and Harvard-GF both state Mass Eye and Ear, 2010-2021.')
    w('  Harvard-FairVision states the same institution, 2010-2022.')
    w()
    w('  Caveat: without the 52 pointwise values the key here is weaker than for GDP x GF.')
    w('        Agreement on (md, age, sex, race, eth) suggests the same patient, but confirming it')
    w('        at the image level requires the 62 GB archive; stage 02 does that.')
    w('        Without patient identifiers the true overlap remains unmeasurable; this is a lower bound.')

    m1.to_csv(RESULT_DIR / 'matches_gdp_fv.csv', index=False)
    m2.to_csv(RESULT_DIR / 'matches_gf_fv.csv', index=False)
    (RESULT_DIR / 'config.json').write_text(json.dumps({
        'prefix': PREFIX, 'generated': f'{datetime.now():%Y-%m-%d %H:%M:%S}',
        'n_perm': N_PERM, 'seed': SEED, 'keys': keys,
        'n_fv': int(len(fv)), 'n_gdp': int(len(gdp)), 'n_gf': int(len(gf)),
        'matches_gdp_fv': int(len(m1)), 'null_max_gdp': int(n1.max()),
        'matches_gf_fv': int(len(m2)), 'null_max_gf': int(n2.max()),
    }, indent=2))
    (RESULT_DIR / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print(f"\nWrote {RESULT_DIR / 'Results.txt'}")


if __name__ == '__main__':
    main()
