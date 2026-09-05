# Verification archive — image-level duplication among three public glaucoma datasets

This archive accompanies the article *"Image-Level Duplication Threatens Validation and Fairness
Assessment in Public Glaucoma Datasets."* It contains the analysis code and the derived results,
so that the reported numbers can be checked rather than taken on trust.

It contains **no source images and no patient-level data**. The three audited datasets are
third-party resources under a licence that does not permit redistribution of adapted material;
see *Data* below.

---

## Start here: the ten-second check

One stage needs nothing but this archive — no GPU, no network, and none of the datasets:

```bash
pip install -r requirements.txt
python code/11_recompute_from_predictions.py
```

It reads a single 38 KB file holding, for each of the 295 eyes common to Harvard-GF and the
Harvard-FairVision test split, the glaucoma label, the racial group, and the model's score under
both evaluation regimes in each of the 5 training runs. From those numbers alone it reproduces
every statistic the manuscript reports for the leakage demonstration, including the paired
bootstrap intervals and the clustered permutation test.

---

## Where each reported number comes from

| Reported in the manuscript | Value | Produced by | Output |
|---|---|---|---|
| Eyes matching on the exact visual-field key | 22 | `00_prepare_and_link_vf.py` | `matched_pairs.csv` |
| Within-dataset visual-field collisions | 0 of 5 443 350 and 0 of 499 500 | `00_prepare_and_link_vf.py` | run log |
| Candidate pairs, Harvard-GF × Harvard-FairVision | 1302 | `01_link_gf_fairvision.py` | `matches_gf_fv.csv` |
| Candidate pairs, Harvard-GDP × Harvard-FairVision | 172 | `01_link_gf_fairvision.py` | `matches_gdp_fv.csv` |
| **Image-identical pairs** | **1031 of 1302 (79.2%)** | `02_census_image_identity.py` | `pairs.csv` |
| Separation: identical ≥0.999945, non-identical ≤0.8584, none between | — | `02`, checked in `06` | `pairs.csv` |
| Contamination of the Harvard-FairVision test split | 295 of 376 (78.5%) | `02`, intervals in `06` | `summary.json` |
| Thickness maps comparable for 13 of the 22 eyes | cosine ≥0.95, median 0.987 | `04_gdp_gf_rnflt_match.py` | `gdp_gf_13_rnflt_cosine.csv` |
| Chance expectation, and its interval | 1.9 (95% CI 0.4–5.5) | `05_chance_and_permutation.py` | `summary.json` |
| Permutation nulls (median / 95th / max) | 3/7/11, 1/4/7, 1/4/8 | `05_chance_and_permutation.py` | `summary.json` |
| Within-release rate understates the cross-release rate | ~145× | `05_chance_and_permutation.py` | run log |
| Fingerprint granularity (distinct MD and age values) | 1919/1225/561, 4921/2425/852 | `05_chance_and_permutation.py` | `summary.json` |
| Split-wise duplication with intervals | 79.3 / 80.8 / 78.5 % | `06_supporting_analyses.py` | `summary.json` |
| The 271 non-identical pairs: three hypotheses | median cosine 0.444; 1 above 0.85; 49 improved by reflection | `06_supporting_analyses.py` | run log |
| **Out-of-sample vs leaked area under the curve** | **0.78 vs 0.90, inflation +0.12** | `07_leakage_5seeds.py` | `per_seed.csv` |
| Paired bootstrap intervals | 0.72–0.83, 0.83–0.94, 0.06–0.18 | `07`, recomputed in `11` | `bootstrap_ci.csv` |
| Across-race range and its change | 0.12 and 0.08; change −0.04 (−0.19 to 0.06) | `07`, recomputed in `11` | `per_seed.csv` |
| Inflation in all 15 subgroup-run combinations | smallest +0.061 | `11_recompute_from_predictions.py` | stdout |
| Rank correlation and its permutation P | ρ = −0.57, two-sided P = .02 | `11_recompute_from_predictions.py` | stdout |
| Most-disadvantaged subgroup changes | 2 of 5 runs | `11_recompute_from_predictions.py` | stdout |
| Citing population | 107 works | `08` and `09` | `citing_population.csv` |
| Per-index contribution | see eMethods 7 | `08_citation_union.py` | `summary.json` |
| Full text obtained; confirmed users | 49; 17 | `09` and `10` | `fulltext_classified.csv`, `newly_classified.csv` |
| Works reporting the duplication | 0 | `09` and `10`, then read by hand | run logs |
| Remainder screened on title and abstract | 29 and 29, none flagged | `10_fulltext_sweep.py` | `abstract_screen.csv` |

Figure 1 is produced by `12_figure1.py`.

---

## What can and cannot be reproduced

Three tiers, stated plainly because the manuscript is about verifiability.

**Deterministic from this archive alone.** Stage 11, and any re-analysis of the shipped CSV and
JSON files. Same input, same output, on any machine.

**Deterministic given the datasets.** Stages 00, 01, 03, 04, 05, 06 and 12. Download the three
releases, place them as shown below, and the numbers follow. Stage 02 also belongs here but
transfers several gigabytes over the network by HTTP range request.

**Not deterministic, and the manuscript says so.**
Stage 07 trains 30 models. The submitted version of this work reported a single run and that run
did not reproduce on different hardware: the same code with the same seed gave an out-of-sample
area under the curve of 0.78 rather than the 0.73 first reported, because the convolution backend
selects algorithms nondeterministically. This is why the analysis of record uses 5 fixed seeds and
reports intervals, and why `superseded/leakage_single_run.py` is kept — so that the discrepancy
can be checked rather than believed.
Stages 08 to 10 query live bibliographic services. Re-running them later will return a larger
citing population as new work accumulates. The snapshot analysed for the manuscript is shipped
under `results/`, so the classification can be checked against the population that was actually
audited.

---

## Layout and how to run

```
code/            the analysis, in the order the argument is built
  config.py      the only place paths are resolved
  00 … 12        stages; each docstring says what it produces and which reported value it gives
  superseded/    three earlier versions, kept because the manuscript reports that they were
                 inadequate and a reader should be able to check that
results/         the derived outputs, one directory per stage
dataset/         you create this; see below
```

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python code/00_prepare_and_link_vf.py     # rebuilds the Harvard-GF visual-field table, ~2 min
python code/01_link_gf_fairvision.py      # ~1 min
...                                        # stages run in order; each is independent given its inputs
```

Paths are resolved once, in `code/config.py`. By default the archive expects `dataset/` and
`results/` beside `code/`. Override with `PAPERB_DATASET` and `PAPERB_RESULTS` if your copies
live elsewhere.

Expected dataset layout:

```
dataset/
  data_summary.csv                                   Harvard-GDP summary (md, td1..td54)
  RNFLT/                                             Harvard-GDP thickness maps
  Harvard-GF/npz/                                    3300 archives
  Harvard-GF/ReadMe/data_summary.csv
  FairVision/Glaucoma/ReadMe/data_summary_glaucoma.csv
```

---

## Data

The three datasets are distributed by the Harvard Ophthalmology AI Laboratory under
CC BY-NC-ND 4.0. Download them from their official repositories; the exact revisions audited,
with their modification dates and checksums, are recorded in eMethods 5 of the article's
Supplement.

**Nothing derived from their contents is redistributed here.** In particular
`vf_meta_gf.csv`, which holds mean deviation and the 52 total-deviation values for each
Harvard-GF eye, is **rebuilt by stage 00 from your own copy** rather than shipped, because a
reformatted extract would be adapted material under that licence.

What is shipped is our own analysis output: computed similarities, match decisions, model scores,
and bibliographic records of third-party publications. Two of those files carry a racial group and
a glaucoma label for the eyes involved, which are needed to check the reported statistics and which
already appear in the article's online supplement.

---

## Run logs

Each stage writes a plain-text log alongside its machine-readable output. The logs are the record
of what was actually run. Where a stage's conclusion was later qualified in the manuscript, the
log says so at that point in the output, and the corresponding docstring explains why. Two such
places exist: the per-study contamination estimates in stage 01, which were withdrawn following
the editor's comment 20, and the racial distribution of the overlap in stage 04, which the
manuscript treats as a sampling artifact rather than as selective duplication.

---

## Citation

Please cite the article. This archive's DOI is listed on its Zenodo record, and the article's DOI
will be added to that record's *Related works* on publication.
