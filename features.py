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
