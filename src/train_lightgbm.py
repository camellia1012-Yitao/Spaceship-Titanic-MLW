"""
Spaceship Titanic prediction with LightGBM.

This script implements:
- feature engineering from PassengerId, Cabin, spending records, group statistics, and missing indicators
- train/test consistent preprocessing
- 10-fold stratified cross-validation
- out-of-fold threshold search
- test probability averaging
- Kaggle submission generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
N_SPLITS = 10
EXPENSE_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

np.random.seed(RANDOM_SEED)


def passenger_group(passenger_id: pd.Series) -> pd.Series:
    return passenger_id.str.split("_", expand=True)[0]


def deck_side_key(cabin: object) -> str:
    if pd.isna(cabin) or cabin == "":
        return "Unknown|Unknown"
    parts = str(cabin).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[2]}"
    return "Unknown|Unknown"


def build_leakage_safe_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Build unsupervised structural statistics and train-only group statistics."""
    train_age_median = train_df["Age"].median()

    full_pid = pd.concat([train_df["PassengerId"], test_df["PassengerId"]], ignore_index=True)
    group_sizes = passenger_group(full_pid).value_counts()

    def surname(names: pd.Series) -> pd.Series:
        return names.fillna("").str.strip().str.split().str[-1].replace("", np.nan)

    all_names = pd.concat([train_df["Name"], test_df["Name"]], ignore_index=True)
    surname_counts = surname(all_names).value_counts()

    all_cabins = pd.concat([train_df["Cabin"], test_df["Cabin"]], ignore_index=True)
    deck_side_counts = all_cabins.map(deck_side_key).value_counts()

    train_groups = passenger_group(train_df["PassengerId"])
    age_imputed = train_df["Age"].fillna(train_age_median)
    cryo_flag = train_df["CryoSleep"].eq(True).astype(float)

    group_frame = pd.DataFrame(
        {
            "group": train_groups,
            "age_imputed": age_imputed,
            "cryo_flag": cryo_flag,
        }
    )

    group_mean_age = group_frame.groupby("group")["age_imputed"].mean()
    group_std_age = group_frame.groupby("group")["age_imputed"].std()
    group_cryo_rate = group_frame.groupby("group")["cryo_flag"].mean()

    global_std_age = age_imputed.std()
    if pd.isna(global_std_age):
        global_std_age = 0.0

    return {
        "train_age_median": train_age_median,
        "group_sizes": group_sizes,
        "surname_counts": surname_counts,
        "deck_side_counts": deck_side_counts,
        "group_mean_age": group_mean_age,
        "group_std_age": group_std_age,
        "group_cryo_rate": group_cryo_rate,
        "global_mean_age": float(age_imputed.mean()),
        "global_std_age": float(global_std_age),
        "global_cryo_rate": float(cryo_flag.mean()),
    }


def preprocess_data(df: pd.DataFrame, stats: dict, has_target: bool) -> pd.DataFrame:
    data = df.copy()

    data[["Group", "GroupNum"]] = data["PassengerId"].str.split("_", expand=True)
    data["Group"] = data["Group"].astype(int)
    data["GroupNum"] = pd.to_numeric(data["GroupNum"], errors="coerce").fillna(1).astype(int)

    groups = passenger_group(data["PassengerId"])
    data["GroupSize"] = groups.map(stats["group_sizes"]).fillna(1).astype(int)
    data["AloneInGroup"] = (data["GroupSize"] == 1).astype(int)
    data["IsLastInGroup"] = (data["GroupNum"] == data["GroupSize"]).astype(int)

    raw_cabin = data["Cabin"].copy()
    data["Cabin"] = data["Cabin"].fillna("Unknown/0/Unknown")
    data[["Deck", "CabinNum", "Side"]] = data["Cabin"].str.split("/", expand=True)
    data["CabinNum"] = pd.to_numeric(data["CabinNum"], errors="coerce").fillna(0)
    data["DeckSide"] = data["Deck"].astype(str) + "|" + data["Side"].astype(str)
    data["DeckSideCount"] = raw_cabin.map(deck_side_key).map(stats["deck_side_counts"]).fillna(0).astype(int)

    data["GroupMeanAgeTrain"] = groups.map(stats["group_mean_age"]).fillna(stats["global_mean_age"]).astype(float)
    data["GroupStdAgeTrain"] = groups.map(stats["group_std_age"]).fillna(0.0).astype(float)
    data["GroupCryoRateTrain"] = groups.map(stats["group_cryo_rate"]).fillna(stats["global_cryo_rate"]).astype(float)

    for col in ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]:
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

    for col in EXPENSE_COLS:
        data[f"Log_{col}"] = np.log1p(data[col])

    data["Age"] = data["Age"].fillna(stats["train_age_median"])
    data["AgeVsGroupMean"] = (data["Age"] - data["GroupMeanAgeTrain"]).astype(float)
    data["SpendPerAge"] = data["TotalSpent"] / (data["Age"] + 1.0)
    data["SpaShare"] = data["Spa"] / (data["TotalSpent"] + 1.0)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-0.001, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    data["IsChild"] = (data["Age"] < 13).astype(int)

    surname = data["Name"].fillna("").str.strip().str.split().str[-1].replace("", np.nan)
    data["SurnameFreq"] = surname.map(stats["surname_counts"]).fillna(1).astype(int)

    cryo_true = data["CryoSleep"].eq(True)
    data["CryoButSpent"] = (cryo_true & (data["TotalSpent"] > 0)).astype(int)

    drop_cols = ["PassengerId", "Cabin", "Name"]
    data = data.drop([col for col in drop_cols if col in data.columns], axis=1)

    if has_target and "Transported" in data.columns:
        data = data.drop("Transported", axis=1)

    return data


def best_threshold_from_oof(oof_prob: np.ndarray, y: np.ndarray) -> float:
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in np.arange(0.38, 0.621, 0.005):
        prediction = (oof_prob >= threshold).astype(np.int8)
        accuracy = accuracy_score(y, prediction)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_passenger_id = test_df["PassengerId"]

    print("Preprocessing data...")
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
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Building LightGBM model...")
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

    print("=" * 60)
    print(f"Running {N_SPLITS}-fold training with OOF probabilities and test averaging...")

    cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(X), dtype=np.float64)
    test_prob_sum = np.zeros(len(X_test), dtype=np.float64)
    fold_accuracies = []

    for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(X, y), start=1):
        full_pipeline.fit(X.iloc[train_idx], y[train_idx])

        valid_prob = full_pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
        oof_prob[valid_idx] = valid_prob

        valid_pred = (valid_prob >= 0.5).astype(int)
        fold_accuracy = accuracy_score(y[valid_idx], valid_pred)
        fold_accuracies.append(fold_accuracy)

        test_prob_sum += full_pipeline.predict_proba(X_test)[:, 1]
        print(f"Fold {fold}/{N_SPLITS} validation accuracy: {fold_accuracy:.4f}")

    print(f"Fold validation accuracies: {np.round(fold_accuracies, 4)}")
    print(f"Mean validation accuracy: {float(np.mean(fold_accuracies)):.4f} (std {float(np.std(fold_accuracies)):.4f})")

    oof_acc_05 = accuracy_score(y, (oof_prob >= 0.5).astype(int))
    threshold = best_threshold_from_oof(oof_prob, y)
    oof_acc_threshold = accuracy_score(y, (oof_prob >= threshold).astype(int))

    print(f"OOF accuracy at threshold 0.5: {oof_acc_05:.4f}")
    print(f"OOF accuracy at searched threshold {threshold:.3f}: {oof_acc_threshold:.4f}")
    print("=" * 60)

    test_prob_mean = test_prob_sum / float(N_SPLITS)
    test_pred = (test_prob_mean >= threshold).astype(int)

    print("Generating submission file...")
    submission = pd.DataFrame(
        {
            "PassengerId": test_passenger_id,
            "Transported": test_pred.astype(bool),
        }
    )
    output_path = "submissions/lightgbm_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
