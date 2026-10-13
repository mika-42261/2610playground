"""モデルは名前で切り替える。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class LGBMWithTargetEncoding:
    """LightGBM + ターゲットエンコーディング(TE)。TE は fit の中で、学習データだけから作る(リーク防止)。

    te_keys の列の組み合わせごとに「Trueの割合」を出して、新しい列 te_<列名>_x_<列名> として足す。
    - 学習データの各行の TE: 学習データをさらに te_inner_folds 個に分けた OOF で作る(自分の正解を使わない)
    - 予測する行(valid や test)の TE: 学習データ全体から作った表を使う(予測する側の正解は使わない)
    - 少ない組み合わせでぶれないよう、全体の割合に近づける(te_smoothing が大きいほど強く近づく)
    te_drop_keys に書いた列は、TE を作ったあと、LightGBM の入力から外す。
    """

    def __init__(self, params):
        params = dict(params)
        self.te_keys = params.pop("te_keys")
        self.te_smoothing = params.pop("te_smoothing", 20)
        self.te_inner_folds = params.pop("te_inner_folds", 5)
        self.te_drop_keys = params.pop("te_drop_keys", [])
        self.te_seed = params.pop("te_seed", 42)
        self.params = params
        self.te_name = "te_" + "_x_".join(self.te_keys)

    def _make_key(self, X):
        # 組み合わせのキー(例: "3|Business")を作る
        key = X[self.te_keys[0]].astype(str)
        for c in self.te_keys[1:]:
            key = key + "|" + X[c].astype(str)
        return key.values

    def _make_stats(self, key, y):
        # キーごとの 行数 と True の数
        return pd.DataFrame({"key": key, "y": y}).groupby("key")["y"].agg(["size", "sum"])

    def _encode(self, key, stats, prior):
        # (Trueの数 + 平滑化 * 全体の割合) / (行数 + 平滑化)。表にないキーは全体の割合になる
        n = pd.Series(key).map(stats["size"]).fillna(0).values
        s = pd.Series(key).map(stats["sum"]).fillna(0).values
        return (s + self.te_smoothing * prior) / (n + self.te_smoothing)

    def _make_input(self, X, te):
        Xm = X.drop(columns=self.te_drop_keys).copy()
        Xm[self.te_name] = te
        return Xm

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        y = np.asarray(y).astype(int)
        key = self._make_key(X)
        # 予測する行に使う表(学習データ全体から)
        self.prior_ = y.mean()
        self.stats_ = self._make_stats(key, y)
        # 学習データの各行の TE(内側の OOF)
        te = np.zeros(len(X))
        inner = KFold(self.te_inner_folds, shuffle=True, random_state=self.te_seed)
        for tr_idx, va_idx in inner.split(X):
            stats = self._make_stats(key[tr_idx], y[tr_idx])
            te[va_idx] = self._encode(key[va_idx], stats, y[tr_idx].mean())
        self.model_ = LGBMClassifier(**self.params).fit(self._make_input(X, te), y)
        return self

    def predict_proba(self, X):
        te = self._encode(self._make_key(X), self.stats_, self.prior_)
        return self.model_.predict_proba(self._make_input(X, te))


def make_model(name, params):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "lgbm_te":
        return LGBMWithTargetEncoding(params)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if name == "logreg":
        # 数値化(エンコード・標準化)が必要。必要になったらここに前処理を足す
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    raise ValueError(f"unknown model: {name}")
