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


@feature("encode_binary")
def encode_binary(df):
    # 2種類の文字列の列を、1と2の数字にする(Class と同じく 0 は使わない)
    binary_rules = {
        "Gender": {"Female": 1, "Male": 2},
        "Customer Type": {"disloyal Customer": 1, "Loyal Customer": 2},
        "Type of Travel": {"Personal Travel": 1, "Business travel": 2},
    }
    for c in binary_rules:
        df[c] = df[c].map(binary_rules[c])
    return df


# targetとの相関が強い順の列(EDA Q8の結果)。top3 は先頭から3列、top4 は4列...を残す
CORR_RANK = [
    "Online boarding", "Class", "Type of Travel", "Inflight entertainment", "Seat comfort",
    "Flight Distance", "On-board service", "Cleanliness", "Leg room service", "Inflight wifi service",
    "Baggage handling", "Checkin service", "Customer Type", "Food and drink", "Age",
    "Ease of Online booking", "Arrival Delay in Minutes", "Departure/Arrival time convenient",
    "Departure Delay in Minutes", "Gate location", "Gender",
]


def make_top_n(n):
    # 相関の強い順に n 列だけ残す関数を作る
    def top_n(df):
        keep_cols = ["id"] + CORR_RANK[:n]  # id は残す(run.py が後で取り除く)
        return df[keep_cols]
    return top_n


# top3 〜 top21 を登録する(top2 は select_top2 として登録済み)
for n in range(3, len(CORR_RANK) + 1):
    feature(f"top{n}")(make_top_n(n))
