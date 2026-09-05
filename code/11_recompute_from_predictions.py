"""Stage 11 — Recompute every reported leakage statistic from the stored predictions.

This is the one stage that needs nothing but the archive itself: no GPU, no network, and
none of the three datasets. It reads a single file,
``results/07_leakage_5seeds/leakage_eval_set.csv``, which holds, for each of the 295 eyes
common to Harvard-GF and the Harvard-FairVision test split, the glaucoma label, the racial
group, and the model's score under both evaluation regimes for each of the 5 training runs.

It then reproduces, from those numbers alone, every value the manuscript reports for the
demonstration: the two areas under the curve and their difference, the paired bootstrap
intervals, the per-subgroup inflation, the across-race range, the rank correlation between
out-of-sample discrimination and the size of the inflation, and the clustered permutation
test used for its P value.

Anyone can therefore check the arithmetic behind the headline numbers in about ten seconds,
without reproducing the training. Re-running the training itself is stage 07.

Expected output (manuscript values in brackets)
----------------------------------------------
out-of-sample AUC          0.78  [0.78, 95% CI 0.72-0.83]
leaked AUC                 0.90  [0.90, 95% CI 0.83-0.94]
inflation                 +0.12  [+0.12, 95% CI 0.06-0.18], positive in all 5 runs
across-race range         0.12 out of sample, 0.08 leaked  [0.12 and 0.08]
change in the range       -0.04 [95% CI -0.19 to 0.06], interval includes zero
Spearman rho              -0.57 with a two-sided permutation P of .02
most-disadvantaged group  changes in 2 of 5 runs

Runtime: a few seconds.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS  # noqa: E402

EVAL = RESULTS / '07_leakage_5seeds' / 'leakage_eval_set.csv'
SEEDS = [42, 43, 44, 45, 46]
RACES = ['asian', 'black', 'white']
N_BOOT = 2000
BOOT_SEED = 0


def auc(y, p):
    """Area under the ROC curve as the Mann-Whitney statistic, ties counted as half."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within ties
    allp = np.concatenate([pos, neg])
    for v in np.unique(allp):
        m = allp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def across_race_range(y, p, race):
    """Largest minus smallest per-group area under the curve. Reported as a range, not a gap."""
    vals = [auc(y[race == r], p[race == r]) for r in RACES]
    return (np.nan if any(np.isnan(v) for v in vals) else max(vals) - min(vals)), vals


def main():
    if not EVAL.exists():
        raise SystemExit(f'{EVAL} not found. It ships with the archive; see the README.')
    d = pd.read_csv(EVAL)
    y = d.glaucoma.to_numpy().astype(int)
    race = d.race.to_numpy()
    oof = {s: d[f'oof_seed{s}'].to_numpy() for s in SEEDS}
    inf = {s: d[f'infold_seed{s}'].to_numpy() for s in SEEDS}

    print(f'Stage 11 — recomputed from {EVAL.name}')
    print(f'  {len(d)} eyes: {int(y.sum())} glaucoma, {int((1 - y).sum())} non-glaucoma; '
          + ', '.join(f'{r} {int((race == r).sum())}' for r in RACES))

    # ---------------------------------------------------------------- per run
    print('\nPer training run')
    print(f'  {"seed":>6}{"out-of-sample":>16}{"leaked":>10}{"inflation":>12}'
          f'{"range (oos)":>14}{"range (leaked)":>16}')
    rows = []
    for s in SEEDS:
        a_o, a_l = auc(y, oof[s]), auc(y, inf[s])
        g_o, per_o = across_race_range(y, oof[s], race)
        g_l, per_l = across_race_range(y, inf[s], race)
        rows.append(dict(seed=s, oos=a_o, leaked=a_l, d=a_l - a_o, g_oos=g_o, g_leak=g_l,
                         per_o=per_o, per_l=per_l))
        print(f'  {s:>6}{a_o:>16.3f}{a_l:>10.3f}{a_l - a_o:>+12.3f}{g_o:>14.3f}{g_l:>16.3f}')
    R = pd.DataFrame(rows)

    # ---------------------------------------------------------------- pooled point estimates
    print('\nPooled across the 5 runs (point estimates are the statistics on the data)')
    print(f'  out-of-sample AUC        {R.oos.mean():.3f}')
    print(f'  leaked AUC               {R.leaked.mean():.3f}')
    print(f'  inflation                {R.d.mean():+.3f}   positive in '
          f'{int((R.d > 0).sum())} of {len(R)} runs')
    print(f'  across-race range        {R.g_oos.mean():.3f} out of sample, '
          f'{R.g_leak.mean():.3f} leaked')
    print(f'  change in the range      {(R.g_leak - R.g_oos).mean():+.3f}')

    # ---------------------------------------------------------------- paired bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    keep = {k: [] for k in ['oos', 'leaked', 'd', 'g_oos', 'g_leak', 'dg']}
    tries = 0
    while len(keep['oos']) < N_BOOT and tries < N_BOOT * 20:
        tries += 1
        i = rng.integers(0, len(d), len(d))
        yy, rr = y[i], race[i]
        if len(set(yy)) < 2 or any(len(set(yy[rr == r])) < 2 for r in RACES):
            continue                      # a resample that cannot define every quantity
        s = SEEDS[rng.integers(0, len(SEEDS))]   # training variability enters here
        po, pl = oof[s][i], inf[s][i]
        ao, al = auc(yy, po), auc(yy, pl)
        go, _ = across_race_range(yy, po, rr)
        gl, _ = across_race_range(yy, pl, rr)
        keep['oos'].append(ao); keep['leaked'].append(al); keep['d'].append(al - ao)
        keep['g_oos'].append(go); keep['g_leak'].append(gl); keep['dg'].append(gl - go)
    q = {k: np.percentile(v, [2.5, 97.5]) for k, v in keep.items()}
    print(f'\nPaired bootstrap, {len(keep["oos"])} resamples of the 295 eyes, '
          f'a training run drawn at random within each')
    for lab, pt, k in [('out-of-sample AUC', R.oos.mean(), 'oos'),
                       ('leaked AUC', R.leaked.mean(), 'leaked'),
                       ('inflation', R.d.mean(), 'd'),
                       ('range, out of sample', R.g_oos.mean(), 'g_oos'),
                       ('range, leaked', R.g_leak.mean(), 'g_leak'),
                       ('change in the range', (R.g_leak - R.g_oos).mean(), 'dg')]:
        print(f'  {lab:<24}{pt:+.3f}  95% CI {q[k][0]:+.3f} to {q[k][1]:+.3f}')

    # ---------------------------------------------------------------- subgroup structure
    print('\nBy racial subgroup, mean across the 5 runs')
    clean = np.array([r['per_o'] for r in rows])       # 5 x 3
    leaked = np.array([r['per_l'] for r in rows])
    lift = leaked - clean
    print(f'  {"group":<8}{"out-of-sample":>16}{"leaked":>10}{"inflation":>12}{"range across runs":>22}')
    for j, r in enumerate(RACES):
        print(f'  {r:<8}{clean[:, j].mean():>16.3f}{leaked[:, j].mean():>10.3f}'
              f'{lift[:, j].mean():>+12.3f}{f"{lift[:, j].min():+.3f} to {lift[:, j].max():+.3f}":>22}')
    print(f'  inflation positive in {int((lift > 0).sum())} of {lift.size} subgroup-run '
          f'combinations, smallest {lift.min():+.3f}')

    # rank correlation, with a permutation test that respects the clustering
    from scipy import stats
    rho = stats.spearmanr(clean.ravel(), lift.ravel())[0]
    null = np.array([stats.spearmanr(clean.ravel(),
                                     np.array([lift[s, list(c[s])] for s in range(len(SEEDS))]).ravel())[0]
                     for c in itertools.product(itertools.permutations(range(3)), repeat=len(SEEDS))])
    p_two = (np.abs(null) >= abs(rho)).mean()
    print(f'\n  Spearman rho between out-of-sample AUC and inflation: {rho:.3f}')
    print(f'  The 15 points are 3 subgroups repeated over 5 runs and are not independent, so the')
    print(f'  P value comes from permuting inflation among the 3 subgroups within each run')
    print(f'  ({len(null)} permutations enumerated exhaustively): two-sided P = {p_two:.3f}')
    print(f'  For comparison, the naive Spearman P, which the manuscript does not report, is '
          f'{stats.spearmanr(clean.ravel(), lift.ravel())[1]:.3f}')

    sw = sum(1 for s in range(len(SEEDS))
             if RACES[int(np.argmin(clean[s]))] != RACES[int(np.argmin(leaked[s]))])
    print(f'\n  The subgroup with the lowest area under the curve differs between the two '
          f'evaluations in {sw} of {len(SEEDS)} runs')


if __name__ == '__main__':
    main()
