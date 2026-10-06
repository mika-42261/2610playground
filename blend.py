"""使い方: python blend.py exp_001 exp_002 ...
OOFのスコアが最良になる重みを求め、submission.csv を作る。"""
import sys
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from metrics import score, higher_is_better


def main(exp_ids):
    cfg = yaml.safe_load(open(f"configs/{exp_ids[0]}.yaml"))
    train = pd.read_csv(cfg["train_path"])
    test = pd.read_csv(cfg["test_path"])
    y = train[cfg["target"]]
    if not pd.api.types.is_numeric_dtype(y):
        y = y == cfg["positive"]
    y = y.astype(int).values

    O = np.column_stack([np.load(f"outputs/{e}/oof.npy") for e in exp_ids])
    T = np.column_stack([np.load(f"outputs/{e}/test.npy") for e in exp_ids])
    n = len(exp_ids)

    sign = -1 if higher_is_better(cfg["metric"]) else 1
    res = minimize(
        lambda w: sign * score(cfg["metric"], y, O @ w),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    w = res.x
    print("weights:", dict(zip(exp_ids, w.round(3))), "| OOF", round(sign * res.fun, 5))

    # 提出形式: id,satisfaction(確率)
    sub = pd.DataFrame({cfg["id_col"]: test[cfg["id_col"]], cfg["target"]: T @ w})
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
