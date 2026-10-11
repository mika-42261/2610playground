"""特徴量は関数1つ=1施策。@feature で登録し、configの features に名前を書いてON/OFFする。"""
REGISTRY = {}


def feature(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def build(df, names):
    out = df.copy()
    for n in names:
        out = REGISTRY[n](out)
    return out


@feature("example")
def example(df):
    # df を受け取って列を足し、dfを返す。ここに施策を増やしていく
    return df


@feature("encode_class")
def encode_class(df):
    # Class(文字列)を、座席のグレード順の数字にする(Eco=1, Eco Plus=2, Business=3。0は使わない)
    class_rules = {"Eco": 1, "Eco Plus": 2, "Business": 3}
    df["Class"] = df["Class"].map(class_rules)
    return df


@feature("select_top2")
def select_top2(df):
    # targetとの相関が強い2列(Online boarding, Class)だけを残す(EDA Q8の結果)
    # id は残す(run.py が後で取り除く)
    keep_cols = ["id", "Online boarding", "Class"]
    return df[keep_cols]
