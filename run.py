"""使い方: python run.py exp_001
configs/exp_001.yaml を読み、CV実行 → OOF/test予測を保存 → results.csv に1行追記。"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from cv import get_folds
from features import build
from metrics import score
from models import make_model


def main(exp_id):
    cfg = yaml.safe_load(open(f"configs/{exp_id}.yaml"))
    train = pd.read_csv(cfg["train_path"])
    test = pd.read_csv(cfg["test_path"])

    y = train[cfg["target"]]
    if not pd.api.types.is_numeric_dtype(y):
        y = y == cfg["positive"]  # 文字列ラベルは陽性=1に変換
    y = y.astype(int).values

    drop = [cfg["id_col"]]
    X = build(train.drop(columns=[cfg["target"]]), cfg["features"]).drop(columns=drop)
    Xt = build(test, cfg["features"]).drop(columns=drop)

    # 文字列列は train/test で同じカテゴリ集合の category 型にする
    for c in X.select_dtypes("object").columns:
        cat = pd.CategoricalDtype(sorted(set(X[c]) | set(Xt[c])))
        X[c], Xt[c] = X[c].astype(cat), Xt[c].astype(cat)

    folds = get_folds(y, cfg["n_splits"], cfg["seed"])
    oof = np.zeros(len(X))
    pred = np.zeros(len(Xt))
    for k in range(cfg["n_splits"]):
        tr, va = folds != k, folds == k
        m = make_model(cfg["model"], cfg["params"])
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / cfg["n_splits"]

    cv = score(cfg["metric"], y, oof)
    out = Path(f"outputs/{exp_id}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof)
    np.save(out / "test.npy", pred)

    row = pd.DataFrame([{
        "exp_id": exp_id,
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "metric": cfg["metric"],
        "cv": round(cv, 5),
        "lb": "",  # 提出後に手入力
        "model": cfg["model"],
        "n_features": X.shape[1],
        "note": cfg.get("note", ""),
        "config": json.dumps(cfg, ensure_ascii=False),
    }])
    row.to_csv("results.csv", mode="a", header=not Path("results.csv").exists(), index=False)
    print(f"{exp_id}: CV {cfg['metric']} {cv:.5f}")


if __name__ == "__main__":
    main(sys.argv[1])
