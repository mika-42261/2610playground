"""CV分割は最初に1回だけ作って保存し、全実験で使い回す。"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def get_folds(y, n_splits=5, seed=42, path="outputs/folds.npy"):
    p = Path(path)
    if p.exists():
        return np.load(p)
    folds = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va] = k
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, folds)
    return folds
