# コンペ情報(Kaggleのオーバービューをここに貼る。Claudeはまずこのファイルを読む)

- コンペ: Predicting Airline Satisfaction (Playground Series S6E10)
- URL: https://www.kaggle.com/competitions/playground-series-s6e10
- 種類: 表形式の二値分類(予測するのは satisfaction の確率)
- 評価指標: ROC-AUC(予測確率と実際のラベルの間)
- 提出形式: ヘッダー付きCSV。列は `id,satisfaction`(satisfaction は確率)
- 期間: 2026-10-01 開始 / 2026-10-31 23:59 UTC 締切(参加・チーム統合・最終提出とも同じ)
- データ: 合成データ(実データから生成)。実データとの分布のズレに注意
- 賞: Kaggleグッズ(学習目的が中心)

## 確認すること(EDAで埋める)
- satisfaction の値の種類(陽性ラベルはどれか)→ configの `positive`
  - 確認済み: `True` / `False` の bool 型(`True` が train の 44.4%)。`True` を 1 として確率を予測する。config は `positive: true`
  - 「True が何を意味するか(満足 / 不満足)」は、データからは確認できていない
- 欠損・カテゴリ列・外れ値
  - 欠損: `Arrival Delay in Minutes` のみ(train 292件 / test 130件、どちらも約0.04%)
  - カテゴリ列: 文字列4本(Gender, Customer Type, Type of Travel, Class)。評価列13本は0〜5の整数
  - train/test で値の種類は一致(連続値3列のみユニーク数に差)。詳細は `01_eda.ipynb`
