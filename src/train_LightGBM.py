# -*- coding: utf-8 -*-
"""
Spaceship Titanic 乘客传送预测 - 仅 LightGBM
特征工程 + K 折 OOF 阈值搜索 + 测试集概率平均；LGBMClassifier 超参数保持不变。
"""
# ===================== 1. 导入依赖库 =====================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# ===================== 2. 全局配置 =====================
RANDOM_SEED = 42
N_SPLITS = 10
np.random.seed(RANDOM_SEED)


# ===================== 3. 数据预处理函数 =====================
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
    """无标签的结构统计 + 仅用训练集聚合的组统计（不含 Transported）。"""
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
    }


def preprocess_data(df: pd.DataFrame, stats: dict, *, has_target: bool) -> pd.DataFrame:
    data = df.copy()
    expense_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

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

    missing_cols = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]
    for col in missing_cols:
        if col in data.columns:
            data[f"{col}_Missing"] = data[col].isnull()

    data[expense_cols] = data[expense_cols].fillna(0)
    data["TotalSpent"] = data[expense_cols].sum(axis=1)
    data["HasSpent"] = data["TotalSpent"] > 0
    data["NoSpending"] = (data["TotalSpent"] == 0).astype(int)
    data["NumAmenitiesUsed"] = (data[expense_cols] > 0).sum(axis=1)
    data["LuxurySpent"] = data["Spa"] + data["RoomService"]
    data["LogTotalSpent"] = np.log1p(data["TotalSpent"])
    data["LogLuxurySpent"] = np.log1p(data["LuxurySpent"])
    for c in expense_cols:
        data[f"Log_{c}"] = np.log1p(data[c])

    med_age = stats["train_age_median"]
    data["Age"] = data["Age"].fillna(med_age)
    data["AgeVsGroupMean"] = (data["Age"] - data["GroupMeanAgeTrain"]).astype(float)
    data["SpendPerAge"] = data["TotalSpent"] / (data["Age"] + 1.0)
    data["SpaShare"] = data["Spa"] / (data["TotalSpent"] + 1.0)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-0.001, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    data["IsChild"] = (data["Age"] < 13).astype(int)

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
    """在 OOF 正类概率上搜索使准确率最大的阈值（单模型、无堆叠）。"""
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.38, 0.621, 0.005):
        pred = (oof_prob >= t).astype(np.int8)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


# ===================== 4. 加载数据 =====================
print("正在加载数据...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_passenger_id = test_df["PassengerId"]

# ===================== 5. 数据预处理 =====================
print("正在预处理数据...")
stats = build_leakage_safe_stats(train_df, test_df)
X = preprocess_data(train_df, stats, has_target=True)
y = train_df["Transported"].astype(int).to_numpy()
X_test = preprocess_data(test_df, stats, has_target=False)

numeric_features = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# ===================== 6. 构建数据处理管道 =====================
print("构建预处理管道...")
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ===================== 7. LightGBM（参数与此前脚本一致）=====================
print("构建LightGBM模型...")
lgb_model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    verbose=-1,
)
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", lgb_model),
    ]
)

# ===================== 8. K 折：OOF 概率 + 验证折准确率 + 测试集概率平均 =====================
print("=" * 60)
print(f"{N_SPLITS} 折训练：OOF 概率、折上准确率、测试集概率平均...")
cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
oof_prob = np.zeros(len(X), dtype=np.float64)
test_prob_sum = np.zeros(len(X_test), dtype=np.float64)
fold_accs = []

for fold, (tr_idx, val_idx) in enumerate(cv_splitter.split(X, y), start=1):
    full_pipeline.fit(X.iloc[tr_idx], y[tr_idx])
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    val_prob = full_pipeline.predict_proba(X_val)[:, 1]
    oof_prob[val_idx] = val_prob
    val_pred = (val_prob >= 0.5).astype(int)
    fold_accs.append(accuracy_score(y_val, val_pred))
    test_prob_sum += full_pipeline.predict_proba(X_test)[:, 1]
    print(f"  折 {fold}/{N_SPLITS} 验证准确率: {fold_accs[-1]:.4f}")

print(f"各折验证准确率: {np.round(fold_accs, 4)}")
print(f"平均验证准确率: {float(np.mean(fold_accs)):.4f} (std {float(np.std(fold_accs)):.4f})")

oof_acc_05 = accuracy_score(y, (oof_prob >= 0.5).astype(int))
thr = best_threshold_from_oof(oof_prob, y)
oof_acc_thr = accuracy_score(y, (oof_prob >= thr).astype(int))
print(f"OOF 准确率 (阈值 0.5): {oof_acc_05:.4f}")
print(f"OOF 准确率 (搜索阈值 {thr:.3f}): {oof_acc_thr:.4f}")
print("=" * 60)

test_prob_mean = test_prob_sum / float(N_SPLITS)
test_pred = (test_prob_mean >= thr).astype(int)

# ===================== 9. 提交 =====================
print("正在生成提交文件...")
submission = pd.DataFrame(
    {
        "PassengerId": test_passenger_id,
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("submission_lgb.csv", index=False)
print("完成。提交文件已保存：submission_lgb.csv")
