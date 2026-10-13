"""使い方: python fold_auc.py exp_001
configs/exp_XXX.yaml と同じ設定・同じfold・同じモデルで学習し直して、
foldごとの AUC を「学習に使ったデータ(train)」と「使わなかったデータ(valid)」で出す。
trainがvalidより大きく上回っていたら過学習のサイン。run.py は変更しない。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from cv import get_folds
from features import build
from metrics import score
from models import make_model


def main(exp_id):
    # configを読む
    cfg = yaml.safe_load(open(f"configs/{exp_id}.yaml"))
    train = pd.read_csv(cfg["train_path"])
    test = pd.read_csv(cfg["test_path"])

    # targetを0/1にする(run.pyと同じ)
    y = train[cfg["target"]]
    if not pd.api.types.is_numeric_dtype(y):
        y = y == cfg["positive"]
    y = y.astype(int).values

    # 特徴量を作る(run.pyと同じ)
    drop = [cfg["id_col"]]
    X = build(train.drop(columns=[cfg["target"]]), cfg["features"]).drop(columns=drop)
    Xt = build(test, cfg["features"]).drop(columns=drop)
    for c in X.select_dtypes("object").columns:
        cat = pd.CategoricalDtype(sorted(set(X[c]) | set(Xt[c])))
        X[c], Xt[c] = X[c].astype(cat), Xt[c].astype(cat)

    # 保存済みのfoldを読む(run.pyと同じfold)
    folds = get_folds(y, cfg["n_splits"], cfg["seed"])

    # foldごとに学習して、train側とvalid側のAUCを出す
    rows = []
    valid_pred_all = np.zeros(len(X))
    for k in range(cfg["n_splits"]):
        tr, va = folds != k, folds == k
        m = make_model(cfg["model"], cfg["params"])
        m.fit(X[tr], y[tr])
        train_pred = m.predict_proba(X[tr])[:, 1]
        valid_pred = m.predict_proba(X[va])[:, 1]
        valid_pred_all[va] = valid_pred
        train_auc = score(cfg["metric"], y[tr], train_pred)
        valid_auc = score(cfg["metric"], y[va], valid_pred)
        rows.append({"fold": k, "train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
                     "train_auc": train_auc, "valid_auc": valid_auc, "gap(train-valid)": train_auc - valid_auc})

    # 平均を最後の行に足す
    table = pd.DataFrame(rows)
    mean_row = table.drop(columns=["fold"]).mean()
    mean_row["fold"] = "mean"
    table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    table[["train_rows", "valid_rows"]] = table[["train_rows", "valid_rows"]].astype(int)  # 行数は整数で表示する

    # 保存して表示する
    out = Path(f"outputs/{exp_id}")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "fold_scores.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # run.pyが保存したOOFと同じ予測になっているか確認する(学習し直しが同じ結果か)
    oof_path = out / "oof.npy"
    if oof_path.exists():
        diff = np.abs(np.load(oof_path) - valid_pred_all).max()
        print(f"\n保存済みOOFとの最大の差: {diff:.2e}  (0に近ければ run.py と同じ結果)")


if __name__ == "__main__":
    main(sys.argv[1])
