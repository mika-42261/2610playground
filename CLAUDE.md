# CLAUDE.md

## 最初にやること
- 作業の最初に `competition.md` を読み、指標・target・提出形式を config に反映する
- 新しい施策は `configs/exp_XXX.yaml` を足して `python run.py exp_XXX` で試す(run.py を直接いじらない)

## 構成と役割
- `run.py`: configを読んでCV実行、OOF/test予測を `outputs/exp_XXX/` に保存し、`results.csv` に1行追記
- `features.py`: 特徴量は関数1つ=1施策。`@feature("名前")` で登録し、configの `features:` に名前を書いてON/OFF
- `models.py`: モデルを名前で切り替える
- `metrics.py`: 評価指標を名前で切り替える
- `cv.py`: fold分割は最初に1回だけ作って保存し、全実験で共通利用する(変えない)
- `blend.py`: 保存済みOOFから最適な重みを求め、`submission.csv` を作る

## 実行環境
- コードはClaude Codeで書き、実行はKaggle Notebookで行う(Claude Codeからはデータに触れない)
- Kaggle側は `!git clone` で取り込み、`!python run.py exp_XXX` で実行する
- 実行結果(results.csvの新しい行、EDAの出力)はユーザーがClaude Codeに貼って共有する

## ノートブックの書き方
- 1セル = 1目的。データ読み込み / 特徴量 / CV / 学習 / 提出作成は別セルにする
- 実験部分は `.py`(`# %%` でセル区切り)で書き、必要なら jupytext で変換する
- EDAは `01_eda.ipynb` に分けて書く

## EDAの書き方
- 1セル = 1つの問い
- グラフを描くコードは関数に隠さず、セルに直接書く
- 図の出力は残して保存する
- 各ステップは次の順で書く
  1. markdown: 問い(なぜこのグラフを見るのか)
  2. markdown(任意): 予想(こう見えるはず、を1行)
  3. code: グラフを描くコード
  4. markdown: 見えたこと(下の4点で書く)

## 「見えたこと」のチェック
ユーザーが書いた「見えたこと」を、次の4点で確認する。
1. 事実: 何がどう見えたか(数値つき)
2. 比較: 何と比べてそうなのか
3. 理由の仮説: なぜそうなっていそうか
4. 次の行動: 特徴量やモデルにどう使うか

抜けている点があれば、書き直さず、質問で1つずつ指摘する(例:「比較の相手は何ですか?」)。
ユーザーの言葉は残し、勝手に書き換えない。

## 注意
- OOF target encoding は fold を考慮して作る(リーク防止)
- 実験ごとにconfigとresults.csvの行が対応するようにする
