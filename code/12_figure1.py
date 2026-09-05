"""Stage 12 — Figure 1: the same OCT scan distributed under two identifiers.

What it produces
----------------
The three panels of Figure 1: a representative B-scan from a Harvard-GF file, its
Harvard-FairVision counterpart distributed under a different identifier, and the absolute
voxelwise difference between them.

Why it is done this way
-----------------------
The pair shown is chosen from the confirmed-identical set of stage 02. Both orientations are
tried and the one with the higher cosine similarity is used, because a small number of pairs are
stored mirrored. The difference panel is what carries the argument: it is empty except for
single-level rounding.

Inputs
------
``dataset/Harvard-GF/npz/`` and the public Harvard-FairVision archive over the network

Runtime: a few minutes, dominated by fetching one archive member. No GPU.

Manuscript values reproduced
----------------------------
Figure 1 and its legend: cosine similarity >= 0.9999, 88% of voxels identical, differences
confined to one 8-bit level.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, RESULTS  # noqa: E402  every path is resolved in config.py

GF_NPZ = DATASET / 'Harvard-GF' / 'npz'
OUT = RESULTS / '12_figure1'
OUT.mkdir(parents=True, exist_ok=True)
FV_ZIP = 'datasets/harvardairobotics/FairVision/Glaucoma/Dataset/dataset.zip'
GF_FILE, FV_FILE = 'data_1574.npz', 'data_00102.npz'


def embed(vol, flip=False):
    v = vol.astype(np.float32)
    if flip:
        v = v[:, :, ::-1]
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    gf = np.load(GF_NPZ / GF_FILE)['oct_bscans']
    print(f'GF {GF_FILE}: {gf.shape} {gf.dtype}')

    fv_cache = OUT / FV_FILE
    if fv_cache.exists():
        fv = np.load(fv_cache)['oct_bscans']
        print(f'FV {FV_FILE}: {fv.shape} {fv.dtype} (from cache)')
    else:
        try:
            import zipfile
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            z = zipfile.ZipFile(fs.open(FV_ZIP, 'rb'))
            members = {Path(n).name: n for n in z.namelist() if n.endswith('.npz') and not n.startswith('__MACOSX')}
            if FV_FILE not in members:
                print(f'FairVision member {FV_FILE} not found in zip.'); return
            with z.open(members[FV_FILE]) as src:
                (OUT / FV_FILE).write_bytes(src.read())
            fv = np.load(OUT / FV_FILE)['oct_bscans']
            print(f'FV {FV_FILE}: {fv.shape} {fv.dtype}')
        except Exception as e:
            print(f'[fetch failed] {type(e).__name__}: {e}')
            print('Network unavailable. Re-run when the archive can be reached; the figure is built then.')
            return

    # Try both orientations and keep the one with the higher cosine similarity.
    a = embed(gf)
    c_noflip = float((a * embed(fv)).sum())
    c_flip = float((a * embed(fv, flip=True)).sum())
    flip = c_flip > c_noflip
    fv_al = fv[:, :, ::-1] if flip else fv
    cos = max(c_noflip, c_flip)
    diff = np.abs(gf.astype(np.int16) - fv_al.astype(np.int16))
    pct_equal = float((diff == 0).mean() * 100)
    print(f'cosine={cos:.6f}  flip={flip}  max|diff|={diff.max()}/255  voxels equal={pct_equal:.1f}%')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z_mid = gf.shape[0] // 2
    gfb, fvb, dfb = gf[z_mid], fv_al[z_mid], diff[z_mid]

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
    ax[0].imshow(gfb, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Harvard-GF\n(one identifier)', fontsize=11)
    ax[1].imshow(fvb, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Harvard-FairVision\n(a different identifier)', fontsize=11)
    dmax = max(1, int(diff.max()))
    im = ax[2].imshow(dfb, cmap='magma', vmin=0, vmax=dmax)
    ax[2].set_title('Absolute difference', fontsize=11)
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label=f'|difference| (8-bit level; max {dmax})')
    cb.set_ticks(list(range(dmax + 1)))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(
        f'The same 3-dimensional OCT scan distributed under two identifiers '
        f'(cosine ≥ {cos:.4f}; {pct_equal:.0f}% of voxels identical; difference ≤ {diff.max()}/255)',
        fontsize=10.5, y=1.02)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig1_image_identity.{ext}', dpi=300, bbox_inches='tight')
    print(f'Wrote {OUT}/fig1_image_identity.png (+pdf)')


if __name__ == '__main__':
    main()
