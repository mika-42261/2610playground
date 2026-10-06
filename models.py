"""モデルは名前で切り替える。"""


def make_model(name, params):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if name == "logreg":
        # 数値化(エンコード・標準化)が必要。必要になったらここに前処理を足す
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    raise ValueError(f"unknown model: {name}")
