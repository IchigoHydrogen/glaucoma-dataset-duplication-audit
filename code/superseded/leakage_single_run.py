"""SUPERSEDED — the single-run leakage demonstration reported in the submitted manuscript.

This script is kept for one reason. The submitted version reported an out-of-sample area under
the curve of 0.73 from a single training run. When the demonstration was repeated for the
revision, the same code with the same random seed produced 0.78 on different hardware; the
difference arises from nondeterministic algorithm selection in the convolution backend together
with library-version differences, and the originally reported inflation of 0.192 lies outside the
interval now reported.

The manuscript discloses this rather than substituting the new number silently. Keeping the
original code here lets a reader check that disclosure instead of taking it on trust.

**Do not use this script for the reported analysis.** The analysis of record is
``07_leakage_5seeds.py``, which fixes and reports all seeds, repeats the whole procedure 5 times,
and attaches paired bootstrap intervals.
"""
import json, time, csv
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # config.py lives in code/
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

PREFIX = Path(__file__).stem
META = RESULTS / '00_prepare_and_link_vf' / 'vf_meta_gf.csv'
GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
PAIRS = RESULTS / '02_census_image_identity' / 'pairs.csv'
CACHE = DATASET / 'Harvard-GF' / 'derived' / 'rnflt_all_3300.npy'
OUT = RESULTS / PREFIX
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED = 42
EPOCHS = 30
PATIENCE = 6
BATCH = 32
LR = 3e-4
WD = 1e-2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


def build_cache(meta):
    if CACHE.exists():
        return np.load(CACHE, mmap_mode='r')
    print('building RNFLT cache for 3300 eyes...', flush=True)
    arr = np.empty((len(meta), 200, 200), np.float16)
    for i, f in enumerate(meta.filename):
        arr[i] = np.load(GF_NPZ / f)['rnflt'].astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(meta)}', flush=True)
    np.save(CACHE, arr)
    return arr


def make_model():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m.to(DEVICE)


def train_predict(V, y, tr, pred_idx, mu, sd, log, tag):
    torch.manual_seed(SEED)
    model = make_model()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    yt = torch.from_numpy(y[tr].astype(np.float32)).to(DEVICE)
    gen = np.random.default_rng(SEED)

    # simple 10% internal val for early stopping
    perm = gen.permutation(len(tr))
    nval = max(BATCH, int(len(tr) * 0.1))
    va_i, tr_i = tr[perm[:nval]], tr[perm[nval:]]
    yva = torch.from_numpy(y[va_i].astype(np.float32)).to(DEVICE)

    def batch(idx):
        x = ((V[idx].astype(np.float32) - mu) / sd)
        t = torch.from_numpy(x).to(DEVICE)[:, None].repeat(1, 3, 1, 1)
        return t

    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        p = gen.permutation(len(tr_i))
        for i in range(0, len(tr_i) - 1, BATCH):
            sel = tr_i[p[i:i + BATCH]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
                out = model(batch(sel)).squeeze(1)
                loss = crit(out, torch.from_numpy(y[sel].astype(np.float32)).to(DEVICE))
            loss.backward(); opt.step()
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
    log(f'    {tag}: val AUC {best:.4f}, stopped ep {ep+1}, peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    del model; torch.cuda.empty_cache()
    return pr


def auc_safe(yy, pp):
    return roc_auc_score(yy, pp) if len(set(yy)) == 2 else float('nan')


def main():
    lines = []
    def w(s=''):
        print(s, flush=True); lines.append(s)
    t0 = time.time()
    w('=' * 78)
    w(f'{PREFIX}: SUPERSEDED single run. Does the overlap move the area under the curve?')
    w(f'Generated {datetime.now():%Y-%m-%d %H:%M:%S}  Device {DEVICE}')
    w('=' * 78)

    meta = pd.read_csv(META).reset_index(drop=True)
    y = meta.glaucoma.to_numpy().astype(int)
    V = build_cache(meta)
    w(f'N = {len(V)} eyes (glaucoma {y.sum()}, normal {(1-y).sum()})  RNFLT {V.shape[1:]}')

    # 295 overlap (FV test, image-identical) with race
    prows = list(csv.DictReader(open(PAIRS)))
    ov = {r['gf_file']: r['race'] for r in prows if r['fv_use'] == 'test' and r['identical'] == '1'}
    meta['ov'] = meta.filename.isin(ov)
    meta['race'] = meta.filename.map(ov)
    ov_idx = meta.index[meta.ov].to_numpy()
    w(f'FV test overlap (leaked into GF training): {len(ov_idx)} / 376 = {len(ov_idx)/376*100:.0f}% of FV test split')
    w(f'  overlap glaucoma balance: {int(y[ov_idx].sum())} glaucoma / {int((1-y[ov_idx]).sum())} normal')
    w(f'  overlap race: {meta.loc[meta.ov,"race"].value_counts().to_dict()}')
    w('')

    # Compute mean and sd of the float16 cache in float32: a float16 sum overflows to inf or nan.
    _vf = np.asarray(V).astype(np.float32)
    mu, sd = np.float32(_vf.mean()), np.float32(_vf.std())
    del _vf
    w(f'thickness-map normalisation: mean {float(mu):.2f} sd {float(sd):.2f}')
    all_idx = np.arange(len(V))

    # CLEAN: 5-fold OOF
    w('run 1: 5-fold out-of-fold prediction (the eye is not in training)')
    oof = np.full(len(V), np.nan)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(all_idx, y))
    for k, (tr, te) in enumerate(folds):
        oof[te] = train_predict(V, y, tr, te, mu, sd, w, f'fold{k+1}')
    np.save(OUT / 'oof.npy', oof)
    w(f'  out-of-fold AUC over all eyes = {auc_safe(y, oof):.4f}')

    # LEAKED: train on ALL GF, predict overlap (in-fold=memorized)
    w('\nrun 2: trained on all of Harvard-GF, predicted in-fold (the eye was in training)')
    infold = train_predict(V, y, all_idx, all_idx, mu, sd, w, 'full-train')
    np.save(OUT / 'infold.npy', infold)

    # ANALYSIS on 295 overlap
    w('\n' + '=' * 78)
    w('result: area under the curve and across-race range on the shared eyes')
    w('=' * 78)
    yo = y[ov_idx]
    rc = meta.loc[meta.ov, 'race'].to_numpy()
    rows = []
    def report(name, pred):
        a = auc_safe(yo, pred[ov_idx])
        per = {}
        for race in ['asian', 'black', 'white']:
            mask = rc == race
            per[race] = auc_safe(yo[mask], pred[ov_idx][mask])
        gap = np.nanmax(list(per.values())) - np.nanmin(list(per.values()))
        w(f'  {name:10s} AUC {a:.4f} | asian {per["asian"]:.3f} black {per["black"]:.3f} white {per["white"]:.3f} | by racegap {gap:.4f}')
        rows.append(dict(setting=name, auc=a, **{f'auc_{k}': v for k, v in per.items()}, race_gap=gap))
        return a, gap
    a_clean, g_clean = report('clean(OOF)', oof)
    a_leak, g_leak = report('leaked', infold)
    w('')
    w(f'  inflation (leaked - out-of-sample) = {a_leak - a_clean:+.4f}')
    w(f'  across-race range: out of sample {g_clean:.4f} -> leaked {g_leak:.4f} (change {g_leak - g_clean:+.4f})')
    w(f'  {len(ov_idx)/376*100:.0f}% of the external test split are eyes that were in training')
    w('')
    if a_leak - a_clean > 0.03:
        w('  NOTE: this single run did not reproduce on other hardware; see the docstring and stage 07.')
    else:
        w('  NOTE: inflation is small under this model and split; report the limitation plainly.')

    pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)
    (OUT / 'config.json').write_text(json.dumps({
        'prefix': PREFIX, 'n': int(len(V)), 'n_overlap': int(len(ov_idx)),
        'auc_clean': a_clean, 'auc_leaked': a_leak, 'auc_inflation': a_leak - a_clean,
        'race_gap_clean': g_clean, 'race_gap_leaked': g_leak,
        'epochs': EPOCHS, 'folds': N_FOLDS, 'model': 'efficientnet_b0', 'seed': SEED,
    }, indent=2, ensure_ascii=False))
    w(f'\nelapsed {time.time()-t0:.0f}s')
    (OUT / 'Results.txt').write_text('\n'.join(lines) + '\n')
    print('Wrote', OUT / 'Results.txt')


if __name__ == '__main__':
    main()
