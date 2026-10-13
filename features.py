"""特徴量は関数1つ=1施策。@feature で登録し、configの features に名前を書いてON/OFFする。"""
import pandas as pd

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


@feature("keep_base2_and_new")
def keep_base2_and_new(df):
    # 土台(exp_002)の2列と、新しく作った列(名前が f_ で始まる列)だけを残す
    # id は残す(run.py が後で取り除く)
    keep_cols = ["id", "Online boarding", "Class"]
    for c in df.columns:
        if c.startswith("f_"):
            keep_cols.append(c)
    return df[keep_cols]


@feature("f_age")
def f_age(df):
    # 案A: 年齢をそのまま新しい列として入れる(木が60歳あたりの区切りを自分で見つける)
    df["f_age"] = df["Age"]
    return df


# 年齢帯の区切り(EDA Q9/Q10 の台形の形から): 20歳未満 / 20-39 / 40-59 / 60-64 / 65歳以上
AGE_EDGES = [0, 20, 40, 60, 65, 200]


def age_band_code(age):
    # 年齢を 1〜5 の番号にする(1=20歳未満, 2=20-39, 3=40-59, 4=60-64, 5=65歳以上)
    return pd.cut(age, AGE_EDGES, labels=[1, 2, 3, 4, 5], right=False).astype(int)


@feature("f_age_band")
def f_age_band(df):
    # 案C: 年齢帯(5区分)の番号を新しい列として入れる
    df["f_age_band"] = age_band_code(df["Age"])
    return df


@feature("f_age_band_travel")
def f_age_band_travel(df):
    # 案D: 年齢帯 × 利用目的(Business travel か)の組み合わせを、1つの番号にする
    # 番号 = 年齢帯の番号 * 2 + (Business travel なら 1、Personal Travel なら 0)  → 2〜11 の10通り
    band = age_band_code(df["Age"])
    is_business_travel = (df["Type of Travel"] == "Business travel").astype(int)
    df["f_age_band_travel"] = band * 2 + is_business_travel
    return df
