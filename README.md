# Kaggle Playground S6E10: Predicting Airline Satisfaction

評価指標: ROC-AUC / 詳細は `competition.md`、作業ルールは `CLAUDE.md`

## 実験の回し方
    python run.py exp_001              # CV実行 → results.csv に追記
    python fold_auc.py exp_001         # foldごとの train/valid AUC(過学習の確認)
    python blend.py exp_001 exp_002    # OOFでブレンド → submission.csv
