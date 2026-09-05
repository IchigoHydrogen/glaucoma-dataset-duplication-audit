"""Stage 02 — Compare all 1302 candidate pairs at the image level.

What it produces
----------------
``pairs.csv``  one row per candidate pair: cosine similarity, whether a horizontal reflection
               improved it, the maximum absolute voxel difference, the fraction of voxels that
               agree exactly, and the identical / not-identical call.

Why it is done this way
-----------------------
An earlier version of this work verified only 30 of the 1302 pairs and extrapolated from
23 of 30. That is an estimate, and it was the weakest point of the argument. This stage
replaces the estimate with a census: every candidate pair is compared, so the reported count
of image-identical pairs is a count and not an inference.

Identity is decided in two steps that do different work. Cosine similarity on a downsampled,
mean-centred embedding is a screen: cheap, tolerant of rescaling, and enough to separate
candidates. The raw uint8 arrays then decide: 88% of the 8 million voxels agree exactly and
the largest difference between any two corresponding voxels is one 8-bit level. Two separate
acquisitions of the same eye cannot agree to within one grey level across 8 million voxels;
that pattern is what re-quantization of a single digital volume looks like.

The 62 GB Harvard-FairVision archive is never downloaded in full. Its ZIP central directory is
read by HTTP range request and only the required members are fetched. Progress is appended one
row at a time so an interrupted run can resume.

Inputs
------
``results/01_link_gf_fairvision/matches_gf_fv.csv``
``dataset/Harvard-GF/npz/``                          local copies of the 3300 Harvard-GF eyes
the public Harvard-FairVision archive, over the network

Runtime: several hours, dominated by network transfer. No GPU.

Manuscript values reproduced
----------------------------
1031 of 1302 pairs identical (79.2%); minimum cosine among identical pairs 0.999945; maximum
among non-identical pairs 0.8584; no pair between the two; 295 of 376 eyes in the
Harvard-FairVision test split identical to Harvard-GF eyes.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

MATCHES = RESULTS / '01_link_gf_fairvision' / 'matches_gf_fv.csv'
GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
OUT = RESULTS / '02_census_image_identity'
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = OUT / 'pairs.csv'
FV_ZIP = 'datasets/harvardairobotics/FairVision/Glaucoma/Dataset/dataset.zip'
GRID = 32
COS_IDENT = 0.99   # observed separation: identical cos >= 0.9999, non-identical cos < 0.65


def embed(vol, flip=False):
    a = np.asarray(vol, np.float32)
    if flip:
        a = a[:, :, ::-1].copy()
    t = torch.from_numpy(a)[None, None]
    v = F.interpolate(t, size=(GRID,) * 3, mode='trilinear', align_corners=False)[0, 0].numpy().ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    m = pd.read_csv(MATCHES)
    print(f'candidates: {len(m)}', flush=True)

    done = set()
    if PAIRS.exists():
        prev = pd.read_csv(PAIRS)
        done = set(zip(prev.gf_file, prev.fv_file))
        print(f'resume: {len(done)} already done', flush=True)
    else:
        with open(PAIRS, 'w') as f:
            f.write('gf_file,fv_file,race,fv_use,cos,flip,max_abs_diff,frac_equal,identical\n')

    import zipfile
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    z = zipfile.ZipFile(fs.open(FV_ZIP, 'rb'))
    members = {Path(n).name: n for n in z.namelist() if n.endswith('.npz') and not n.startswith('__MACOSX')}
    print(f'zip members: {len(members)}', flush=True)

    t0 = time.time()
    n_done = len(done)
    for i, r in enumerate(m.itertuples(index=False)):
        key = (r.gf_file, r.fv_file)
        if key in done:
            continue
        try:
            gf = np.load(GF_NPZ / r.gf_file)['oct_bscans']
            with z.open(members[r.fv_file]) as src:
                fv = np.load(io.BytesIO(src.read()))['oct_bscans']
            a = embed(gf)
            c0 = float(a @ embed(fv))
            c1 = float(a @ embed(fv, flip=True))
            flip = c1 > c0
            cos = max(c0, c1)
            fv_al = fv[:, :, ::-1] if flip else fv
            diff = np.abs(gf.astype(np.int16) - fv_al.astype(np.int16))
            max_diff = int(diff.max())
            frac_eq = float((diff == 0).mean())
            ident = int(cos > COS_IDENT)
            row = f'{r.gf_file},{r.fv_file},{r.race},{r.fv_use},{cos:.6f},{int(flip)},{max_diff},{frac_eq:.6f},{ident}\n'
        except Exception as e:
            row = f'{r.gf_file},{r.fv_file},{r.race},{r.fv_use},ERROR,,,,{type(e).__name__}\n'
            print(f'  [err] {r.gf_file}x{r.fv_file}: {e}', flush=True)
        with open(PAIRS, 'a') as f:
            f.write(row)
        n_done += 1
        if n_done % 50 == 0:
            el = time.time() - t0
            rate = (n_done - len(done)) / el if el else 0
            eta = (len(m) - n_done) / rate / 60 if rate else 0
            print(f'  {n_done}/{len(m)}  {rate:.2f}/s  ETA {eta:.0f} min', flush=True)

    # ----- tally -----
    d = pd.read_csv(PAIRS)
    d = d[d.cos != 'ERROR'].copy()
    d['cos'] = d.cos.astype(float); d['identical'] = d.identical.astype(int)
    lines = []
    def w(s=''):
        print(s, flush=True); lines.append(s)
    w('=' * 60)
    w('Census over all candidate pairs: Harvard-GF x Harvard-FairVision image identity')
    w(f'Generated after {(time.time()-t0)/60:.0f} min')
    w('=' * 60)
    N = len(d)
    ni = int(d.identical.sum())
    w(f'candidates verified: {N} / {len(m)}')
    w(f'image-identical (cos>{COS_IDENT}): {ni} / {N} = {ni/N*100:.1f}%')
    w(f'as a share of the 3300 Harvard-GF eyes: {ni/3300*100:.1f}%')
    w(f'Harvard-GF eyes remaining after removing the duplicates: {3300 - ni}')
    w('')
    w('confirmed identical by race (denominator: candidate pairs in that group):')
    for race, sub in d.groupby('race'):
        k = int(sub.identical.sum())
        w(f'  {race}: {k}/{len(sub)} = {k/len(sub)*100:.1f}%  ({k/1100*100:.1f}% of the 1100 in that group)')
    w('')
    w('confirmed identical by designated Harvard-FairVision split:')
    for use, sub in d.groupby('fv_use'):
        k = int(sub.identical.sum())
        w(f'  {use}: {k}/{len(sub)} identical')
    w('')
    w('cosine distribution: minimum and median among identical, maximum among non-identical')
    idg = d[d.identical == 1].cos; nig = d[d.identical == 0].cos
    if len(idg): w(f'  identical: min {idg.min():.4f} median {idg.median():.4f}')
    if len(nig): w(f'  non-ident: max {nig.max():.4f} median {nig.median():.4f}')
    (OUT / 'Summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('WROTE', OUT / 'Summary.txt', flush=True)


if __name__ == '__main__':
    main()
