# Paper B 開示監査 (2026-07-23)

査読(JAMA Oph 4名)の最大懸念=「重複は未開示/検出不能」が未証明、への一次確認。
問い: GF/FairVision/GDP の原論文・README・配布アーカイブ・ラボページは、
      データセット間で患者/眼/OCTスキャンを共有・重複・部分集合であることを開示しているか。

## 確認したソースと結果 (WebFetch, 2026-07-23)
| ソース | URL | 重複開示 |
|---|---|---|
| Harvard-GF 論文 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11251413/ | なし (FairVision/GDP言及ゼロ) |
| FairVision 論文 | https://arxiv.org/abs/2310.02492 (html) | なし (GF/GDP言及ゼロ) |
| ラボ Data Resources | https://ophai.hms.harvard.edu/data/ | なし (別資源として列挙のみ) |
| Zenodo 配布 | https://zenodo.org/records/13178701 | なし |
| Harvard-GF GitHub README | https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF | なし |

## ソース記述の要点
- GF: 「大学眼科病院、2010-2021、最終visit・1眼/被験者をランダム選択、各人種1000、信号強度<6除外」。
- FairVision: 「VF検査を要件にしたため緑内障側に偏る」。glaucoma normal/glaucoma定義あり。
- 共通ソース(Mass Eye and Ear)と時期重複(GF2010-2021/FairVision2015-2022)はWebSearchで追えるが、
  スキャン重複の明示はどの文書にも無い。

## 結論
共通施設ソースは明白/追跡可能。しかし「同一の眼・OCTスキャンを別IDで再配布」していることは、
原論文・リポジトリ・Zenodo・ラボページのいずれも開示していない。
=> Paper Bの「未開示・配布物から検出不能」フレームは一次情報で支持される(弱めるでなく固める)。
中立化: 当事者非難でなく「公開文書が何を述べていないか」の事実として記述する。
