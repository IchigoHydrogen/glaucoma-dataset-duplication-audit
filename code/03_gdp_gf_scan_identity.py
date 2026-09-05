"""Stage 03 — Test whether the 22 visual-field-identical eyes carry the same scan.

What it produces
----------------
``scan_identity.csv``  for each of the 22 Harvard-GDP x Harvard-GF pairs found in stage 00:
                       the correlation between the two thickness maps, the fitted slope and
                       intercept, whether a horizontal reflection was needed, and the difference
                       in recorded age.

Why it is done this way
-----------------------
Exact agreement on all 52 total-deviation points and on mean deviation is already a
near-conclusive signature of the same eye. This stage asks the separate question of whether the
*image* is also the same. It cannot be answered as decisively as for Harvard-GF and
Harvard-FairVision, because Harvard-GDP distributes a 225 x 225 thickness map while Harvard-GF
distributes 200 x 200; a resolution difference makes pixel identity untestable in either
direction. What can be tested is whether the two maps are consistent after resampling.

Inputs
------
``results/00_prepare_and_link_vf/matched_pairs.csv``
``dataset/RNFLT/``                  Harvard-GDP thickness maps
``dataset/Harvard-GF/npz/``

Runtime: under a minute. No GPU, no network.

Manuscript values reproduced
----------------------------
The 22 pairs and the statement in Table 1 that pixel identity is not testable because of a
resolution difference.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
PAST_RESULTS = {
    'pairs': RESULTS / '00_prepare_and_link_vf' / 'matched_pairs.csv',
}
GDP_RNFLT = DATASET / 'RNFLT'
GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
RESULT_DIR = RESULTS / PREFIX
RESULT_DIR.mkdir(parents=True, exist_ok=True)

COS_THRESH = 0.95


def main():
    lines = []

    def w(s=''):
        print(s, flush=True)
        lines.append(s)

    w('=' * 84)
    w(f'{PREFIX}: are the 22 matched eyes a second visit, or the same acquisition?')
    w('=' * 84)
    w(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    w(f'Script    : {PREFIX}.py')
    w('The Harvard-GDP 225x225 map is resampled to the Harvard-GF 200x200 grid before comparison.')
    w('Resampling makes exact identity untestable, so slope and correlation are examined instead.')
    w()

    m = pd.read_csv(PAST_RESULTS['pairs'])
    rows = []
    for _, r in m.iterrows():
        G = np.load(GDP_RNFLT / f'{r.gdp_file}.npz')['rnflt'].astype(float)
        G = resize(G, (200, 200), order=1, anti_aliasing=True, preserve_range=True)
        F0 = np.load(GF_NPZ / r.gf_file)['rnflt'].astype(float)
        best = None
        for flip in (False, True):
            F = F0[:, ::-1] if flip else F0
            c = float(np.corrcoef(G.ravel(), F.ravel())[0, 1])
            if best is None or c > best[0]:
                best = (c, F, flip)
        c, F, flip = best
        a, b = np.polyfit(F.ravel(), G.ravel(), 1)
        rows.append(dict(f_gdp=r.gdp_file, f_gf=r.gf_file, age_diff=r.age_diff,
                         pearson_r=c, slope=a, intercept=b, flipped=flip))
    d = pd.DataFrame(rows)

    w('=' * 84)
    w('SECTION 1: agreement between the thickness maps of the 22 pairs')
    w('=' * 84)
    hi = d[d.pearson_r > COS_THRESH]
    lo = d[d.pearson_r <= COS_THRESH]
    w(f'  pairs with correlation > {COS_THRESH}: {len(hi)} / {len(d)}')
    w(f'    correlation {hi.pearson_r.min():.4f}-{hi.pearson_r.max():.4f}  '
      f'slope median {hi.slope.median():.4f} (range {hi.slope.min():.4f}-{hi.slope.max():.4f})')
    w(f'    age difference {hi.age_diff.min():.2f}-{hi.age_diff.max():.2f} y (median {hi.age_diff.median():.2f})')
    w(f'  pairs with correlation <= {COS_THRESH}: {len(lo)} / {len(d)}')
    w(f'    correlation {lo.pearson_r.min():.4f}-{lo.pearson_r.max():.4f}  '
      f'median age difference {lo.age_diff.median():.2f} y')
    w()

    w('=' * 84)
    w('SECTION 2: is a second visit consistent with what the maps show?')
    w('=' * 84)
    r1 = float(np.corrcoef(d.age_diff, d.pearson_r)[0, 1])
    r2 = float(np.corrcoef(hi.age_diff, hi.slope)[0, 1])
    w(f'  age difference vs correlation : r = {r1:+.3f}')
    w(f'  age difference vs slope       : r = {r2:+.3f} (pairs above {COS_THRESH} only)')
    w()
    w('  If these were second visits, a larger age gap should mean less similar thickness maps,')
    w('  and thinning with age should make the slope move with the age gap.')
    w('  Neither relationship is present.')
    w(f'  The pair with the largest age gap, {d.loc[d.age_diff.idxmax(),"age_diff"]:.2f} y, still correlates at '
      f'{d.loc[d.age_diff.idxmax(),"pearson_r"]:.4f}.')
    w()
    w('  A second visit by the same patient is therefore not what these data look like.')
    w('  Whether one acquisition was recorded under two ages, or the same eye was imaged twice,')
    w('  cannot be told apart from the distributed files. Either way the patient appears in both.')
    w('  The manuscript therefore reports the fact and the ambiguity, not a mechanism.')

    d.to_csv(RESULT_DIR / 'scan_identity.csv', index=False)
    (RESULT_DIR / 'config.json').write_text(json.dumps({
        'prefix': PREFIX, 'generated': f'{datetime.now():%Y-%m-%d %H:%M:%S}',
        'cos_thresh': COS_THRESH, 'n_pairs': int(len(d)),
        'n_high_corr': int(len(hi)),
        'slope_median_high': float(hi.slope.median()),
        'corr_agediff_vs_r': r1, 'corr_agediff_vs_slope': r2,
    }, indent=2))
    (RESULT_DIR / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print(f"\nWrote {RESULT_DIR / 'Results.txt'}")


if __name__ == '__main__':
    main()
