# -*- coding: utf-8 -*-
"""
Spaceship Titanic：增强特征 + ExtraTrees，
K 折 OOF 阈值搜索 + 测试集概率平均。
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

RANDOM_SEED = 42
N_SPLITS = 12
np.random.seed(RANDOM_SEED)

EXPENSE_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _passenger_group(s: pd.Series) -> pd.Series:
    return s.str.split("_", expand=True)[0]


def _deck_side_key(cabin: object) -> str:
    if pd.isna(cabin) or cabin == "":
        return "Unknown|Unknown"
    parts = str(cabin).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[2]}"
    return "Unknown|Unknown"


def build_leakage_safe_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_age_median = train_df["Age"].median()
    full_pid = pd.concat([train_df["PassengerId"], test_df["PassengerId"]], ignore_index=True)
    group_sizes = _passenger_group(full_pid).value_counts()

    def _surname(names: pd.Series) -> pd.Series:
        return names.fillna("").str.strip().str.split().str[-1].replace("", np.nan)

    all_names = pd.concat([train_df["Name"], test_df["Name"]], ignore_index=True)
    surname_counts = _surname(all_names).value_counts()

    all_cabins = pd.concat([train_df["Cabin"], test_df["Cabin"]], ignore_index=True)
    deck_side_counts = all_cabins.map(_deck_side_key).value_counts()

    g_train = _passenger_group(train_df["PassengerId"])
    age_imp = train_df["Age"].fillna(train_age_median)
    cryo_f = train_df["CryoSleep"].eq(True).astype(float)
    tsub = pd.DataFrame({"_g": g_train, "_age_imp": age_imp, "_cryo": cryo_f})
    grp_mean_age = tsub.groupby("_g")["_age_imp"].mean()
    grp_std_age = tsub.groupby("_g")["_age_imp"].std()
    grp_cryo_rate = tsub.groupby("_g")["_cryo"].mean()

    global_mean_age = float(age_imp.mean())
    _std = age_imp.std()
    global_std_age = float(_std) if pd.notna(_std) else 0.0
    global_cryo_rate = float(cryo_f.mean())

    tr_tot = train_df[EXPENSE_COLS].fillna(0).sum(axis=1)
    te_tot = test_df[EXPENSE_COLS].fillna(0).sum(axis=1)
    g_all = pd.concat(
        [_passenger_group(train_df["PassengerId"]), _passenger_group(test_df["PassengerId"])],
        ignore_index=True,
    )
    tot_all = pd.concat([tr_tot, te_tot], ignore_index=True)
    spend_grp = pd.DataFrame({"_g": g_all, "_s": tot_all})
    grp_spend_sum = spend_grp.groupby("_g")["_s"].sum()
    grp_spend_mean = spend_grp.groupby("_g")["_s"].mean()
    grp_spend_std = spend_grp.groupby("_g")["_s"].std()
    grp_spend_max = spend_grp.groupby("_g")["_s"].max()

    return {
        "train_age_median": train_age_median,
        "group_sizes": group_sizes,
        "surname_counts": surname_counts,
        "deck_side_counts": deck_side_counts,
        "grp_mean_age": grp_mean_age,
        "grp_std_age": grp_std_age,
        "grp_cryo_rate": grp_cryo_rate,
        "global_mean_age": global_mean_age,
        "global_std_age": global_std_age,
        "global_cryo_rate": global_cryo_rate,
        "grp_spend_sum": grp_spend_sum,
        "grp_spend_mean": grp_spend_mean,
        "grp_spend_std": grp_spend_std,
        "grp_spend_max": grp_spend_max,
    }


def preprocess_data(df: pd.DataFrame, stats: dict, *, has_target: bool) -> pd.DataFrame:
    data = df.copy()

    data[["Group", "GroupNum"]] = data["PassengerId"].str.split("_", expand=True)
    data["Group"] = data["Group"].astype(int)
    data["GroupNum"] = pd.to_numeric(data["GroupNum"], errors="coerce").fillna(1).astype(int)
    g = _passenger_group(data["PassengerId"])
    data["GroupSize"] = g.map(stats["group_sizes"]).fillna(1).astype(int)
    data["AloneInGroup"] = (data["GroupSize"] == 1).astype(int)
    data["IsLastInGroup"] = (data["GroupNum"] == data["GroupSize"]).astype(int)

    if "Cabin" in data.columns:
        raw_cabin = data["Cabin"].copy()
        data["Cabin"] = data["Cabin"].fillna("Unknown/0/Unknown")
        data[["Deck", "CabinNum", "Side"]] = data["Cabin"].str.split("/", expand=True)
        data["CabinNum"] = pd.to_numeric(data["CabinNum"], errors="coerce").fillna(0)
        data["DeckSide"] = data["Deck"].astype(str) + "|" + data["Side"].astype(str)
        ds_key = raw_cabin.map(_deck_side_key)
        data["DeckSideCount"] = ds_key.map(stats["deck_side_counts"]).fillna(0).astype(int)

    data["GroupMeanAgeTrain"] = g.map(stats["grp_mean_age"]).fillna(stats["global_mean_age"]).astype(float)
    std_m = g.map(stats["grp_std_age"])
    data["GroupStdAgeTrain"] = std_m.fillna(0.0).astype(float)
    data["GroupCryoRateTrain"] = g.map(stats["grp_cryo_rate"]).fillna(stats["global_cryo_rate"]).astype(float)

    data["GroupSpendSum"] = g.map(stats["grp_spend_sum"]).fillna(0.0).astype(float)
    data["GroupSpendMean"] = g.map(stats["grp_spend_mean"]).fillna(0.0).astype(float)
    gs_std = g.map(stats["grp_spend_std"])
    data["GroupSpendStd"] = gs_std.fillna(0.0).astype(float)
    data["GroupSpendMax"] = g.map(stats["grp_spend_max"]).fillna(0.0).astype(float)

    missing_cols = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]
    for col in missing_cols:
        if col in data.columns:
            data[f"{col}_Missing"] = data[col].isnull()

    data[EXPENSE_COLS] = data[EXPENSE_COLS].fillna(0)
    data["TotalSpent"] = data[EXPENSE_COLS].sum(axis=1)
    data["HasSpent"] = data["TotalSpent"] > 0
    data["NoSpending"] = (data["TotalSpent"] == 0).astype(int)
    data["NumAmenitiesUsed"] = (data[EXPENSE_COLS] > 0).sum(axis=1)
    data["LuxurySpent"] = data["Spa"] + data["RoomService"]
    data["LogTotalSpent"] = np.log1p(data["TotalSpent"])
    data["LogLuxurySpent"] = np.log1p(data["LuxurySpent"])
    for c in EXPENSE_COLS:
        data[f"Log_{c}"] = np.log1p(data[c])

    med_age = stats["train_age_median"]
    data["Age"] = data["Age"].fillna(med_age)
    data["AgeVsGroupMean"] = (data["Age"] - data["GroupMeanAgeTrain"]).astype(float)
    data["SpendPerAge"] = data["TotalSpent"] / (data["Age"] + 1.0)
    data["SpaShare"] = data["Spa"] / (data["TotalSpent"] + 1.0)
    data["SpendPerGroupMember"] = data["TotalSpent"] / data["GroupSize"].clip(lower=1).astype(float)
    data["GroupSpendSumMinusSelf"] = (data["GroupSpendSum"] - data["TotalSpent"]).clip(lower=0.0)

    if "HomePlanet" in data.columns and "Destination" in data.columns:
        hp = data["HomePlanet"].fillna("NA").astype(str)
        des = data["Destination"].fillna("NA").astype(str)
        data["Home_Dest"] = hp + "|" + des
        if "Deck" in data.columns:
            data["Home_Deck"] = hp + "|" + data["Deck"].astype(str)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-0.001, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    data["IsChild"] = (data["Age"] < 13).astype(int)
    data["IsInfant"] = (data["Age"] < 3).astype(int)

    if "Name" in data.columns:
        sn = data["Name"].fillna("").str.strip().str.split().str[-1].replace("", np.nan)
        data["SurnameFreq"] = sn.map(stats["surname_counts"]).fillna(1).astype(int)

    cryo_true = data["CryoSleep"].eq(True)
    data["CryoButSpent"] = (cryo_true & (data["TotalSpent"] > 0)).astype(int)

    drop_cols = ["PassengerId", "Cabin", "Name"]
    data = data.drop([c for c in drop_cols if c in data.columns], axis=1)

    if has_target and "Transported" in data.columns:
        data = data.drop("Transported", axis=1)

    return data


def best_threshold_from_oof(oof_prob: np.ndarray, y: np.ndarray) -> float:
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.34, 0.661, 0.005):
        pred = (oof_prob >= t).astype(np.int8)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def oof_accuracy_with_best_threshold(oof_prob: np.ndarray, y: np.ndarray):
    thr = best_threshold_from_oof(oof_prob, y)
    acc = accuracy_score(y, (oof_prob >= thr).astype(int))
    return acc, thr


print("正在加载数据...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_passenger_id = test_df["PassengerId"]

print("正在预处理数据...")
stats = build_leakage_safe_stats(train_df, test_df)
X = preprocess_data(train_df, stats, has_target=True)
y = train_df["Transported"].astype(int).to_numpy()
X_test = preprocess_data(test_df, stats, has_target=False)

numeric_features = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
preprocessor_tpl = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

et_estimator = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

print("=" * 60)
print(f"{N_SPLITS} 折：ExtraTrees，收集 OOF / 测试概率...")
cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_et = np.zeros(len(X), dtype=np.float64)
test_et_sum = np.zeros(len(X_test), dtype=np.float64)
fold_acc_et = []

for fold, (tr_idx, val_idx) in enumerate(cv_splitter.split(X, y), start=1):
    pipe_et = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor_tpl)),
            ("classifier", clone(et_estimator)),
        ]
    )

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    pipe_et.fit(X_tr, y_tr)

    p_e = pipe_et.predict_proba(X_val)[:, 1]
    oof_et[val_idx] = p_e

    fold_acc_et.append(accuracy_score(y_val, (p_e >= 0.5).astype(int)))
    test_et_sum += pipe_et.predict_proba(X_test)[:, 1]
    print(f"  折 {fold}/{N_SPLITS}  ET={fold_acc_et[-1]:.4f}")

print(f"平均验证准确率 ET: {float(np.mean(fold_acc_et)):.4f}")

acc_et, thr_et = oof_accuracy_with_best_threshold(oof_et, y)
test_et_mean = test_et_sum / float(N_SPLITS)

print("-" * 60)
print(f"OOF ET={acc_et:.4f}  thr={thr_et:.3f}")
print("=" * 60)

test_pred = (test_et_mean >= thr_et).astype(int)
submission = pd.DataFrame(
    {"PassengerId": test_passenger_id, "Transported": test_pred.astype(bool)}
)
submission.to_csv("submission_ensemble.csv", index=False)
print("完成。submission_ensemble.csv 已写入。")
