"""評価指標は名前で切り替える。(関数, 大きいほど良いか)"""
from sklearn.metrics import roc_auc_score, mean_squared_error

METRICS = {
    "auc": (roc_auc_score, True),
    "rmse": (lambda y, p: mean_squared_error(y, p) ** 0.5, False),
}


def score(name, y, p):
    return METRICS[name][0](y, p)


def higher_is_better(name):
    return METRICS[name][1]
