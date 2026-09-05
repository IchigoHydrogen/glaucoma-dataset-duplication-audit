"""Stage 00 — Build the Harvard-GF visual-field table, and link Harvard-GDP to Harvard-GF.

What this produces
------------------
1. ``vf_meta_gf.csv`` — one row per Harvard-GF eye with mean deviation, the 52 total-deviation
   values, age and the glaucoma label, read from the distributed ``.npz`` files. Later stages
   need this table; it is rebuilt here rather than shipped with the archive, because the
   datasets are released under CC BY-NC-ND and a reformatted extract of their contents would
   be an adaptation.
2. ``matched_pairs.csv`` — the Harvard-GDP and Harvard-GF eyes whose visual fields agree
   exactly on all 52 total-deviation points and on mean deviation. This is the "same eye"
   signature described in eMethods 2 of the manuscript. **The manuscript reports 22 such pairs.**

Why an exact match is used
--------------------------
A 52-point integer vector plus mean deviation is a very high-dimensional key. Within either
dataset no two eyes share it (0 collisions in 5 443 350 Harvard-GF pairs and 499 500
Harvard-GDP pairs), so an exact agreement across the two releases is near-conclusive evidence
of the same eye. The trade-off is stated in the manuscript: exact matching favours specificity
over sensitivity, so the count is a lower bound.

Inputs (see README for the expected layout)
-------------------------------------------
``dataset/Harvard-GF/npz/*.npz``              3300 files
``dataset/Harvard-GF/ReadMe/data_summary.csv``
``dataset/data_summary.csv``                  Harvard-GDP, carries md and td1..td52

Runtime: about 2 minutes, no GPU, no network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  paths live in config.py

OUT = RESULTS / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)

GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
GF_CSV = DATASET / 'Harvard-GF' / 'ReadMe' / 'data_summary.csv'
GDP_CSV = DATASET / 'data_summary.csv'


def build_gf_table():
    """Read every Harvard-GF .npz once and keep only the visual-field fields.

    Cached: reading 3300 archives takes about 2 minutes, so a previous run is reused.
    """
    cached = OUT / 'vf_meta_gf.csv'
    if cached.exists():
        print(f'  reusing {cached.name} from a previous run')
        return pd.read_csv(cached)
    rows = []
    files = sorted(GF_NPZ.glob('*.npz'))
    if not files:
        raise SystemExit(f'No .npz files under {GF_NPZ}. See the README for the expected layout.')
    for k, f in enumerate(files, 1):
        d = np.load(f)
        row = {'filename': f.name, 'glaucoma': int(d['glaucoma']),
               'md': float(d['md']), 'age': float(d['age'])}
        row.update({f'td{i + 1}': float(v) for i, v in enumerate(np.asarray(d['tds']).ravel())})
        rows.append(row)
        if k % 500 == 0:
            print(f'  ...{k}/{len(files)}', flush=True)
    return pd.DataFrame(rows)


def vf_key(md, tds):
    """The fingerprint: mean deviation to 2 decimals plus the 52 integer total-deviation values."""
    return (round(float(md), 2),) + tuple(int(round(float(x))) for x in tds)


def main():
    print('Stage 00 — Harvard-GF visual-field table and Harvard-GDP x Harvard-GF linkage')

    gf = build_gf_table()
    gf.to_csv(OUT / 'vf_meta_gf.csv', index=False)
    print(f'  vf_meta_gf.csv written: {len(gf)} eyes')

    gdp = pd.read_csv(GDP_CSV)
    # Harvard-GDP names its columns td1..td54 and omits the two blind-spot locations
    # (td25 and td34), leaving 52. Harvard-GF stores the same 52 values as a plain vector.
    # The two are therefore matched by position, not by column name.
    gdp_td = sorted((c for c in gdp.columns if c.startswith('td') and c[2:].isdigit()),
                    key=lambda c: int(c[2:]))
    gf_td = [f'td{i}' for i in range(1, 53)]
    if len(gdp_td) != 52 or any(c not in gf.columns for c in gf_td):
        raise SystemExit(f'expected 52 total-deviation columns on each side, '
                         f'found {len(gdp_td)} in Harvard-GDP')

    gf_index = {}
    for _, r in gf.iterrows():
        gf_index.setdefault(vf_key(r.md, [r[c] for c in gf_td]), []).append(r.filename)

    pairs = []
    for _, r in gdp.iterrows():
        for gf_file in gf_index.get(vf_key(r.md, [r[c] for c in gdp_td]), []):
            g = gf[gf.filename == gf_file].iloc[0]
            pairs.append(dict(gdp_file=r.filename, gf_file=gf_file, md=round(float(r.md), 2),
                              gdp_age=r.age, gf_age=g.age, age_diff=round(float(g.age) - float(r.age), 2),
                              gdp_glaucoma=int(r.glaucoma), gf_glaucoma=int(g.glaucoma)))
    p = pd.DataFrame(pairs)
    # Only the file identifiers and the quantities we derive from them are written to disk.
    # The mean deviation, the two ages and the two glaucoma labels are values belonging to
    # CC BY-NC-ND datasets, so they are printed for whoever runs this and not redistributed.
    if len(p):
        p['label_agree'] = (p.gdp_glaucoma == p.gf_glaucoma).astype(int)
    p.reindex(columns=['gdp_file', 'gf_file', 'age_diff', 'label_agree']).to_csv(
        OUT / 'matched_pairs.csv', index=False)

    print(f'  matched_pairs.csv written: {len(p)} pairs (the manuscript reports 22)')
    if len(p):
        print(f'  glaucoma label agrees in {int(p.label_agree.sum())} of {len(p)}')
        print(f'  age difference between the two releases: '
              f'median {p.age_diff.median():.2f} y, range {p.age_diff.min():.2f} to {p.age_diff.max():.2f}')

    # Within-dataset collision counts, quoted in Table 1 of the manuscript
    def collisions(df, cols):
        keys = [vf_key(r.md, [r[c] for c in cols]) for _, r in df.iterrows()]
        n = len(keys)
        return n * (n - 1) // 2, n - len(set(keys))
    gf_pairs, gf_dupe = collisions(gf, gf_td)
    gdp_pairs, gdp_dupe = collisions(gdp, gdp_td)
    print(f'  within-dataset visual-field collisions: '
          f'Harvard-GF {gf_dupe} of {gf_pairs:,} pairs, Harvard-GDP {gdp_dupe} of {gdp_pairs:,} pairs')
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
