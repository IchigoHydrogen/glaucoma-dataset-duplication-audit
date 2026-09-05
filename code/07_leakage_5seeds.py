"""Stage 07 — What the overlap does to a downstream evaluation, with uncertainty.

What it produces
----------------
``per_seed.csv``           one row per training run: the two areas under the curve, their
                           difference, the across-race range under each, and the per-race values
``bootstrap_ci.csv``       paired bootstrap intervals, per run
``leakage_eval_set.csv``   the 295 evaluated eyes with their labels, racial group and the model's
                           score under both regimes for all 5 runs. **Stage 11 recomputes every
                           reported statistic from this one file, with no GPU and no datasets.**
``oof_seed*.npy`` / ``infold_seed*.npy``  the raw prediction vectors over all 3300 Harvard-GF eyes

Why it is done this way
-----------------------
The submitted version of this manuscript reported a single training run and no intervals. The
editor asked whether the effect was robust or a small-sample artifact. Answering that required
repeating the whole procedure, and in doing so we found that the original single run did not
reproduce on different hardware: the same code with the same seed gave an out-of-sample area
under the curve of 0.78 rather than the 0.73 first reported. The original script is kept in
``superseded/leakage_single_run.py`` so that this can be checked.

The comparison is paired by construction. The out-of-sample estimate is an out-of-fold
prediction: Harvard-GF is split into 5 stratified folds and every eye is scored by a model that
never saw it. The leaked estimate comes from a model trained on all 3300 eyes and applied to the
same eyes in-fold. Both are then read off the same 295 shared eyes, so the two numbers differ
only in whether the evaluated eye was in training. Nothing else changes: same architecture, same
hyperparameters, same seed.

The training conditions are unchanged from the original implementation; only the seed was made a
parameter and the bootstrap added.

Inputs
------
``vf_meta_gf.csv`` from stage 00, ``pairs.csv`` from stage 02, ``dataset/Harvard-GF/npz/``

Runtime: several hours on one GPU (30 models: 5 folds plus a full-data fit, for each of 5 seeds).

Manuscript values reproduced
----------------------------
Out-of-sample 0.78 (95% CI 0.72-0.83), leaked 0.90 (0.83-0.94), inflation +0.12 (0.06-0.18)
positive in all 5 runs; across-race range 0.12 and 0.08 with a change of -0.04 (-0.19 to 0.06);
inflation in all 15 subgroup-run combinations (eTable 10).
"""
import csv
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
META = RESULTS / '00_prepare_and_link_vf' / 'vf_meta_gf.csv'
GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
PAIRS = RESULTS / '02_census_image_identity' / 'pairs.csv'
CACHE = DATASET / 'Harvard-GF' / 'derived' / 'rnflt_all_3300.npy'
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)

# ---- training conditions, unchanged from the original implementation ----
N_FOLDS = 5
EPOCHS = 30
PATIENCE = 6
BATCH = 32
LR = 3e-4
WD = 1e-2
# ---- added for the revision ----
SEEDS = [42, 43, 44, 45, 46]      # seed 42 corresponds to the run reported at submission
N_BOOT = 2000
RACES = ['asian', 'black', 'white']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

lines = []


def w(s=''):
    print(s, flush=True)
    lines.append(s)


def build_cache(meta):
    if CACHE.exists():
        return np.load(CACHE, mmap_mode='r')
    arr = np.empty((len(meta), 200, 200), np.float16)
    for i, f in enumerate(meta.filename):
        arr[i] = np.load(GF_NPZ / f)['rnflt'].astype(np.float32)
    np.save(CACHE, arr)
    return arr


def make_model():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m.to(DEVICE)


def train_predict(V, y, tr, pred_idx, mu, sd, seed, tag):
    """Identical to the original implementation; only the seed was made a parameter."""
    torch.manual_seed(seed)
    model = make_model()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = np.random.default_rng(seed)

    perm = gen.permutation(len(tr))
    nval = max(BATCH, int(len(tr) * 0.1))
    va_i, tr_i = tr[perm[:nval]], tr[perm[nval:]]

    def batch(idx):
        x = (V[idx].astype(np.float32) - mu) / sd
        return torch.from_numpy(x).to(DEVICE)[:, None].repeat(1, 3, 1, 1)

    best, best_state, bad, ep = -1, None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        p = gen.permutation(len(tr_i))
        for i in range(0, len(tr_i) - 1, BATCH):
            sel = tr_i[p[i:i + BATCH]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
                out = model(batch(sel)).squeeze(1)
                loss = crit(out, torch.from_numpy(y[sel].astype(np.float32)).to(DEVICE))
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            pv = torch.cat([torch.sigmoid(model(batch(va_i[i:i + BATCH])).squeeze(1)).float()
                            for i in range(0, len(va_i), BATCH)]).cpu().numpy()
        auc = roc_auc_score(y[va_i], pv)
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
        pr = torch.cat([torch.sigmoid(model(batch(pred_idx[i:i + BATCH])).squeeze(1)).float()
                        for i in range(0, len(pred_idx), BATCH)]).cpu().numpy()
    w(f'      {tag}: internal val AUC {best:.4f}, stopped epoch {ep+1}')
    del model
    torch.cuda.empty_cache()
    return pr


def auc_safe(yy, pp):
    return roc_auc_score(yy, pp) if len(set(yy)) == 2 else float('nan')


def race_gap(yy, pp, rr):
    per = {r: auc_safe(yy[rr == r], pp[rr == r]) for r in RACES}
    v = [x for x in per.values() if x == x]
    return (max(v) - min(v) if len(v) >= 2 else float('nan')), per


def paired_bootstrap(yo, rc, p_clean, p_leak, n_boot=N_BOOT, seed=0):
    """Resample the 295 eyes. Both regimes are evaluated on the same eyes, so each resample
    is used for both and the difference receives an interval too (a paired bootstrap).
    """
    rng = np.random.default_rng(seed)
    n = len(yo)
    keep = {k: [] for k in ['auc_clean', 'auc_leak', 'd_auc', 'gap_clean', 'gap_leak', 'd_gap']}
    tries = 0
    while len(keep['auc_clean']) < n_boot and tries < n_boot * 20:
        tries += 1
        i = rng.integers(0, n, n)
        yy, rr = yo[i], rc[i]
        if len(set(yy)) < 2:
            continue
        # Skip a resample in which any racial group lacks both classes: the range is undefined there.
        if any(len(set(yy[rr == r])) < 2 for r in RACES):
            continue
        ac, al = auc_safe(yy, p_clean[i]), auc_safe(yy, p_leak[i])
        gc, _ = race_gap(yy, p_clean[i], rr)
        gl, _ = race_gap(yy, p_leak[i], rr)
        keep['auc_clean'].append(ac); keep['auc_leak'].append(al); keep['d_auc'].append(al - ac)
        keep['gap_clean'].append(gc); keep['gap_leak'].append(gl); keep['d_gap'].append(gl - gc)
    return {k: np.array(v) for k, v in keep.items()}, tries


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    t0 = time.time()
    w('=' * 96)
    w(f'{PREFIX}: repeated runs and confidence intervals for the leakage demonstration')
    w(f'Generated {datetime.now():%Y-%m-%d %H:%M:%S}   Device {DEVICE}')
    w('=' * 96)

    w('\n[architecture and training regimen]  (unchanged from the original implementation)')
    w('  input       Harvard-GF thickness map, 200x200, z-normalised over all eyes and')
    w('              replicated to 3 channels to match the ImageNet weights')
    w('  model       EfficientNet-B0 (IMAGENET1K_V1), classifier replaced by a single linear output')
    w('  loss        BCEWithLogitsLoss')
    w(f'  optimiser   AdamW  lr={LR}  weight_decay={WD}  batch={BATCH}')
    w(f'  schedule    CosineAnnealingLR (T_max={EPOCHS})')
    w(f'  early stop  10% of the training data held out internally; stop if the area under the '
      f'curve does not improve for {PATIENCE} epochs (at most {EPOCHS}); best epoch kept')
    w('  precision   bfloat16 autocast')
    w('\n[how the out-of-sample estimate is obtained]')
    w('  Harvard-GF is split into 5 stratified folds; every eye is scored by a model trained on')
    w('  the other 4 folds, so no eye is scored by a model that saw it (out-of-fold prediction).')
    w('  This makes the comparison paired: the same 295 eyes under both regimes.')
    w('\n[how the leaked estimate is obtained]')
    w('  A model trained on all 3300 Harvard-GF eyes scores the same 295 eyes in-fold.')

    meta = pd.read_csv(META).reset_index(drop=True)
    y = meta.glaucoma.to_numpy().astype(int)
    V = build_cache(meta)

    prows = list(csv.DictReader(open(PAIRS)))
    ov = {r['gf_file']: r['race'] for r in prows if r['fv_use'] == 'test' and r['identical'] == '1'}
    meta['ov'] = meta.filename.isin(ov)
    meta['race'] = meta.filename.map(ov)
    ov_idx = meta.index[meta.ov].to_numpy()
    yo = y[ov_idx]
    rc = meta.loc[meta.ov, 'race'].to_numpy()

    w(f'\ntraining set: {len(V)} Harvard-GF eyes ({int(y.sum())} glaucoma / {int((1-y).sum())} non-glaucoma)')
    w(f'  evaluation set: {len(ov_idx)} eyes (those of the 376 in the Harvard-FairVision test split '
      f'that are image-identical to a Harvard-GF eye)')
    w(f'  of which {int(yo.sum())} glaucoma / {int((1-yo).sum())} non-glaucoma, '
      f'by race {dict(pd.Series(rc).value_counts())}')

    _vf = np.asarray(V).astype(np.float32)
    mu, sd = np.float32(_vf.mean()), np.float32(_vf.std())
    del _vf
    all_idx = np.arange(len(V))

    per_seed, preds = [], {}
    for s in SEEDS:
        w(f'\n--- seed {s} ---')
        oof = np.full(len(V), np.nan)
        for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                                     random_state=s).split(all_idx, y)):
            oof[te] = train_predict(V, y, tr, te, mu, sd, s, f'clean fold{k+1}')
        infold = train_predict(V, y, all_idx, all_idx, mu, sd, s, 'leaked full-train')
        preds[s] = (oof[ov_idx], infold[ov_idx])
        np.save(OUT / f'oof_seed{s}.npy', oof)
        np.save(OUT / f'infold_seed{s}.npy', infold)

        ac, al = auc_safe(yo, oof[ov_idx]), auc_safe(yo, infold[ov_idx])
        gc, pc = race_gap(yo, oof[ov_idx], rc)
        gl, pl = race_gap(yo, infold[ov_idx], rc)
        w(f'    clean  AUC {ac:.4f}  ' + ' '.join(f'{r} {pc[r]:.3f}' for r in RACES) + f'  gap {gc:.4f}')
        w(f'    leaked AUC {al:.4f}  ' + ' '.join(f'{r} {pl[r]:.3f}' for r in RACES) + f'  gap {gl:.4f}')
        per_seed.append(dict(seed=s, auc_clean=ac, auc_leaked=al, d_auc=al - ac,
                             gap_clean=gc, gap_leaked=gl, d_gap=gl - gc,
                             **{f'clean_{r}': pc[r] for r in RACES},
                             **{f'leaked_{r}': pl[r] for r in RACES}))

    ps = pd.DataFrame(per_seed)
    ps.to_csv(OUT / 'per_seed.csv', index=False)

    w('\n' + '=' * 96)
    w(f'variation across the {len(SEEDS)} independent training runs')
    w('=' * 96)
    w(f'{"quantity":26s}{"mean":>9s}{"SD":>8s}{"min":>9s}{"max":>9s}')
    for c, lab in [('auc_clean', 'clean AUC'), ('auc_leaked', 'leaked AUC'),
                   ('d_auc', 'inflation'), ('gap_clean', 'across-race range, out of sample'),
                   ('gap_leaked', 'across-race range, leaked'), ('d_gap', 'change in the range')]:
        v = ps[c]
        w(f'  {lab:24s}{v.mean():9.4f}{v.std(ddof=1):8.4f}{v.min():9.4f}{v.max():9.4f}')

    w('\n' + '=' * 96)
    w(f'paired bootstrap 95% intervals ({len(ov_idx)} eyes resampled, {N_BOOT} replicates)')
    w('=' * 96)
    boot_rows = []
    for s in SEEDS:
        pc_, pl_ = preds[s]
        b, tries = paired_bootstrap(yo, rc, pc_, pl_, seed=s)
        r = {'seed': s, 'n_boot_used': len(b['auc_clean']), 'n_tries': tries}
        for k in ['auc_clean', 'auc_leak', 'd_auc', 'gap_clean', 'gap_leak', 'd_gap']:
            lo, hi = ci(b[k])
            r[f'{k}_lo'], r[f'{k}_hi'] = lo, hi
        boot_rows.append(r)
        w(f'\n  seed {s} (usable resamples {len(b["auc_clean"])}/{tries})')
        w(f'    clean  AUC {np.mean(b["auc_clean"]):.3f} [{r["auc_clean_lo"]:.3f}, {r["auc_clean_hi"]:.3f}]'
          f'   across-race range {np.mean(b["gap_clean"]):.3f} [{r["gap_clean_lo"]:.3f}, {r["gap_clean_hi"]:.3f}]')
        w(f'    leaked AUC {np.mean(b["auc_leak"]):.3f} [{r["auc_leak_lo"]:.3f}, {r["auc_leak_hi"]:.3f}]'
          f'   across-race range {np.mean(b["gap_leak"]):.3f} [{r["gap_leak_lo"]:.3f}, {r["gap_leak_hi"]:.3f}]')
        w(f'    difference: AUC {np.mean(b["d_auc"]):+.3f} [{r["d_auc_lo"]:+.3f}, {r["d_auc_hi"]:+.3f}]'
          f'   range {np.mean(b["d_gap"]):+.3f} [{r["d_gap_lo"]:+.3f}, {r["d_gap_hi"]:+.3f}]')
    pd.DataFrame(boot_rows).to_csv(OUT / 'bootstrap_ci.csv', index=False)

    # Pooled interval: a training run is drawn at random inside each resample, so the interval
    w('\n' + '-' * 96)
    w('pooled interval: a training run is drawn at random inside each resample, so the interval')
    rng = np.random.default_rng(0)
    pooled = {k: [] for k in ['auc_clean', 'auc_leak', 'd_auc', 'gap_clean', 'gap_leak', 'd_gap']}
    n = len(yo)
    tries = 0
    while len(pooled['auc_clean']) < N_BOOT and tries < N_BOOT * 20:
        tries += 1
        s = SEEDS[rng.integers(0, len(SEEDS))]
        pc_, pl_ = preds[s]
        i = rng.integers(0, n, n)
        yy, rr = yo[i], rc[i]
        if len(set(yy)) < 2 or any(len(set(yy[rr == r])) < 2 for r in RACES):
            continue
        ac, al = auc_safe(yy, pc_[i]), auc_safe(yy, pl_[i])
        gc, _ = race_gap(yy, pc_[i], rr)
        gl, _ = race_gap(yy, pl_[i], rr)
        pooled['auc_clean'].append(ac); pooled['auc_leak'].append(al); pooled['d_auc'].append(al - ac)
        pooled['gap_clean'].append(gc); pooled['gap_leak'].append(gl); pooled['d_gap'].append(gl - gc)
    pooled = {k: np.array(v) for k, v in pooled.items()}
    summary = {}
    for k, lab in [('auc_clean', 'out-of-sample AUC'), ('auc_leak', 'leaked AUC'),
                   ('d_auc', 'inflation'), ('gap_clean', 'across-race range, out of sample'),
                   ('gap_leak', 'across-race range, leaked'), ('d_gap', 'change in the range')]:
        lo, hi = ci(pooled[k])
        summary[k] = dict(mean=float(np.mean(pooled[k])), lo=lo, hi=hi)
        w(f'  {lab:34s} {np.mean(pooled[k]):+.3f} [95%CI {lo:+.3f}, {hi:+.3f}]')

    w('\n  Note: the across-race range is a maximum minus a minimum over 3 groups, not a smooth')
    w('  functional; point estimates are the statistics on the data, intervals are percentiles.')
    w('  Note: each per-race value rests on 90 to 104 eyes, so those intervals are wide.')

    (OUT / 'summary.json').write_text(json.dumps({
        'seeds': SEEDS, 'n_boot': N_BOOT, 'n_overlap': int(len(ov_idx)),
        'architecture': 'efficientnet_b0 (IMAGENET1K_V1), classifier -> Linear(1)',
        'input': 'Harvard-GF RNFLT 200x200, z-normalised, replicated to 3 channels',
        'optimiser': f'AdamW lr={LR} wd={WD} batch={BATCH}',
        'schedule': f'CosineAnnealingLR T_max={EPOCHS}',
        'early_stopping': f'10% internal validation, patience {PATIENCE}, max {EPOCHS} epochs',
        'clean_definition': '5-fold stratified out-of-fold prediction within Harvard-GF',
        'leaked_definition': 'train on all 3300 Harvard-GF eyes, predict the same eyes in-fold',
        'per_seed': per_seed, 'pooled_ci': summary,
    }, indent=2, ensure_ascii=False))
    w(f'\nelapsed {time.time()-t0:.0f}s')
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('Wrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
