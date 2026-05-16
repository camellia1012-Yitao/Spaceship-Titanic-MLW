"""
Train the final CatBoost-based blended pipeline for the Spaceship Titanic project.

This script is the reproducible Python version of the CatBoost notebook. It uses
shared feature engineering utilities from preprocessing.py, trains two CatBoost
feature-set variants with 5-fold stratified cross-validation, selects a blending
weight using out-of-fold predictions, and exports a Kaggle-ready submission file.

Typical usage from the project root:

    python src/train_catboost.py \
        --train data/train.csv \
        --test data/test.csv \
        --output submissions/catboost_final_submission.csv

If data/train.csv and data/test.csv are not found, the script also tries
train.csv and test.csv in the current working directory.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

try:
    from preprocessing import build_train_test_features, prepare_submission
except ImportError:  # Allows running from notebooks or alternative project layouts.
    from src.preprocessing import build_train_test_features, prepare_submission

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

BASE_FEATURES = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupID", "GroupMember", "Deck", "CabinNum", "Side",
    "Surname", "TotalSpend", "NoSpend", "GroupSize", "IsAlone",
    "AgeBand", "LuxurySpend", "BasicSpend",
]

CROSS_FEATURES = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupID", "GroupMember", "Deck", "CabinNum", "Side",
    "Surname", "TotalSpend", "NoSpend", "GroupSize", "IsAlone",
    "AgeBand", "LuxurySpend", "BasicSpend",
    "Deck_Side", "HomePlanet_Deck", "Destination_Deck", "VIP_HomePlanet",
    "Cryo_NoSpend", "Spa_ratio", "VRDeck_ratio", "FoodCourt_ratio", "Luxury_ratio",
    "IsChild", "IsAdult",
]


def resolve_input_path(path: str | Path, fallback_name: str) -> Path:
    """Return an existing input path, trying a local fallback when needed."""
    path = Path(path)
    if path.exists():
        return path

    fallback = Path(fallback_name)
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find {path} or fallback file {fallback}. "
        "Please pass the correct path with --train or --test."
    )


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def add_catboost_notebook_compatibility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aliases and ratio features used by the final CatBoost notebook.

    preprocessing.py creates the shared project features. This function adds the
    few CatBoost-specific feature names used by the original notebook so the
    script keeps the same modeling logic while still relying on the shared
    preprocessing pipeline.
    """
    df = df.copy()

    if "PassengerId" in df.columns:
        group_parts = df["PassengerId"].astype(str).str.split("_", expand=True)
        df["GroupID"] = pd.to_numeric(group_parts[0], errors="coerce")
        df["GroupMember"] = pd.to_numeric(group_parts[1], errors="coerce")

    if "IsSolo" in df.columns:
        df["IsAlone"] = df["IsSolo"].astype(int)
    elif "GroupSize" in df.columns:
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["AgeBand"] = pd.cut(
            df["Age"],
            bins=[-1, 12, 18, 25, 40, 60, 100],
            labels=["Child", "Teen", "Youth", "Adult", "MidAge", "Senior"],
        )
        df["IsChild"] = (df["Age"] < 18).astype(float)
        df["IsAdult"] = (df["Age"] >= 18).astype(float)

    for col in SPENDING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if set(SPENDING_COLS).issubset(df.columns):
        spend_filled = df[SPENDING_COLS].fillna(0)
        df["TotalSpend"] = spend_filled.sum(axis=1)
        df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
        df["LuxurySpend"] = spend_filled["Spa"] + spend_filled["VRDeck"]
        df["BasicSpend"] = (
            spend_filled["RoomService"]
            + spend_filled["FoodCourt"]
            + spend_filled["ShoppingMall"]
        )
        denominator = df["TotalSpend"] + 1
        df["Spa_ratio"] = df["Spa"] / denominator
        df["VRDeck_ratio"] = df["VRDeck"] / denominator
        df["FoodCourt_ratio"] = df["FoodCourt"] / denominator
        df["Luxury_ratio"] = df["LuxurySpend"] / denominator

    if {"VIP", "HomePlanet"}.issubset(df.columns):
        df["VIP_HomePlanet"] = df["VIP"].astype(str) + "_" + df["HomePlanet"].astype(str)

    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df


def prepare_for_catboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[int]]:
    """Prepare categorical and numerical columns for CatBoost."""
    x_train = x_train.copy()
    x_test = x_test.copy()

    categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

    for col in categorical_cols:
        x_train[col] = x_train[col].astype("object").where(x_train[col].notna(), "Missing").astype(str)
        x_test[col] = x_test[col].astype("object").where(x_test[col].notna(), "Missing").astype(str)

    for col in numerical_cols:
        median_value = x_train[col].median()
        if pd.isna(median_value):
            median_value = 0
        x_train[col] = x_train[col].fillna(median_value)
        x_test[col] = x_test[col].fillna(median_value)

    cat_feature_indices = [x_train.columns.get_loc(col) for col in categorical_cols]
    return x_train, x_test, categorical_cols, numerical_cols, cat_feature_indices


def validate_features(df: pd.DataFrame, features: Sequence[str], feature_set_name: str) -> None:
    """Raise a clear error if a feature set references missing columns."""
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {feature_set_name}: {missing}")


def train_catboost_blend(
    x_a: pd.DataFrame,
    x_b: pd.DataFrame,
    x_test_a: pd.DataFrame,
    x_test_b: pd.DataFrame,
    y: pd.Series,
    cat_idx_a: list[int],
    cat_idx_b: list[int],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Train two CatBoost variants and return OOF/test probabilities."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_a = np.zeros(len(x_a))
    oof_b = np.zeros(len(x_b))
    test_pred_a = np.zeros(len(x_test_a))
    test_pred_b = np.zeros(len(x_test_b))
    fold_records = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x_a, y), 1):
        print(f"========== Fold {fold} ==========")

        x_train_a, x_valid_a = x_a.iloc[train_idx], x_a.iloc[valid_idx]
        x_train_b, x_valid_b = x_b.iloc[train_idx], x_b.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model_a = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=42 + fold,
            verbose=0,
        )

        model_b = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.025,
            depth=7,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=100 + fold,
            verbose=0,
        )

        model_a.fit(
            x_train_a,
            y_train,
            cat_features=cat_idx_a,
            eval_set=(x_valid_a, y_valid),
            use_best_model=True,
        )

        model_b.fit(
            x_train_b,
            y_train,
            cat_features=cat_idx_b,
            eval_set=(x_valid_b, y_valid),
            use_best_model=True,
        )

        valid_pred_a = model_a.predict_proba(x_valid_a)[:, 1]
        valid_pred_b = model_b.predict_proba(x_valid_b)[:, 1]

        oof_a[valid_idx] = valid_pred_a
        oof_b[valid_idx] = valid_pred_b
        test_pred_a += model_a.predict_proba(x_test_a)[:, 1] / n_splits
        test_pred_b += model_b.predict_proba(x_test_b)[:, 1] / n_splits

        acc_a = accuracy_score(y_valid, (valid_pred_a >= THRESHOLD).astype(int))
        acc_b = accuracy_score(y_valid, (valid_pred_b >= THRESHOLD).astype(int))
        fold_records.append({"fold": fold, "model_A_accuracy": acc_a, "model_B_accuracy": acc_b})

        print(f"Model A fold accuracy: {acc_a:.5f}")
        print(f"Model B fold accuracy: {acc_b:.5f}")

    return oof_a, oof_b, test_pred_a, test_pred_b, pd.DataFrame(fold_records)


def select_blend_weight(
    y: pd.Series,
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    step: float = 0.02,
) -> tuple[float, float]:
    """Select the best Model A blend weight using OOF accuracy."""
    best_score = -1.0
    best_weight = 0.5

    for weight_a in np.arange(0.0, 1.0 + step / 2, step):
        blend_oof = weight_a * oof_a + (1 - weight_a) * oof_b
        blend_pred = (blend_oof >= THRESHOLD).astype(int)
        score = accuracy_score(y, blend_pred)

        if score > best_score:
            best_score = score
            best_weight = float(weight_a)

    return best_weight, best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the final CatBoost pipeline.")
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--output",
        default="submissions/catboost_final_submission.csv",
        help="Path for the Kaggle submission CSV",
    )
    parser.add_argument(
        "--fold-results",
        default="results/catboost_fold_results.csv",
        help="Path for fold-level validation results",
    )
    parser.add_argument(
        "--oof-probabilities",
        default="results/catboost_oof_probabilities.csv",
        help="Path for OOF probability exports",
    )
    parser.add_argument(
        "--test-probabilities",
        default="results/catboost_test_probabilities.csv",
        help="Path for test probability exports",
    )
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run the full CatBoost training and submission pipeline."""
    args = parse_args()

    train_path = resolve_input_path(args.train, "train.csv")
    test_path = resolve_input_path(args.test, "test.csv")

    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)

    train_fe, test_fe = build_train_test_features(train_df, test_df)
    train_fe = add_catboost_notebook_compatibility_features(train_fe)
    test_fe = add_catboost_notebook_compatibility_features(test_fe)

    y = train_fe["Transported"].astype(int)
    test_passenger_ids = test_fe["PassengerId"].copy()

    validate_features(train_fe, BASE_FEATURES, "BASE_FEATURES")
    validate_features(train_fe, CROSS_FEATURES, "CROSS_FEATURES")

    x_a = train_fe[BASE_FEATURES].copy()
    x_test_a = test_fe[BASE_FEATURES].copy()
    x_b = train_fe[CROSS_FEATURES].copy()
    x_test_b = test_fe[CROSS_FEATURES].copy()

    x_a, x_test_a, cat_cols_a, _, cat_idx_a = prepare_for_catboost(x_a, x_test_a)
    x_b, x_test_b, cat_cols_b, _, cat_idx_b = prepare_for_catboost(x_b, x_test_b)

    print("Model A feature count:", len(BASE_FEATURES))
    print("Model B feature count:", len(CROSS_FEATURES))
    print("Model A categorical columns:", cat_cols_a)
    print("Model B categorical columns:", cat_cols_b)

    oof_a, oof_b, test_pred_a, test_pred_b, fold_results = train_catboost_blend(
        x_a=x_a,
        x_b=x_b,
        x_test_a=x_test_a,
        x_test_b=x_test_b,
        y=y,
        cat_idx_a=cat_idx_a,
        cat_idx_b=cat_idx_b,
        n_splits=args.splits,
        random_state=args.seed,
    )

    best_weight, best_score = select_blend_weight(y, oof_a, oof_b)
    print("Best OOF accuracy:", round(best_score, 5))
    print("Best weight for Model A:", round(best_weight, 2))
    print("Best weight for Model B:", round(1 - best_weight, 2))

    final_oof_prob = best_weight * oof_a + (1 - best_weight) * oof_b
    final_test_prob = best_weight * test_pred_a + (1 - best_weight) * test_pred_b
    final_test_pred = final_test_prob >= THRESHOLD

    output_path = ensure_parent_dir(args.output)
    fold_results_path = ensure_parent_dir(args.fold_results)
    oof_prob_path = ensure_parent_dir(args.oof_probabilities)
    test_prob_path = ensure_parent_dir(args.test_probabilities)

    submission = prepare_submission(test_passenger_ids, final_test_pred, output_path=output_path)
    fold_results.to_csv(fold_results_path, index=False)

    pd.DataFrame({
        "PassengerId": train_df["PassengerId"],
        "Transported": y,
        "Model_A_Probability": oof_a,
        "Model_B_Probability": oof_b,
        "Blended_Probability": final_oof_prob,
    }).to_csv(oof_prob_path, index=False)

    pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Model_A_Probability": test_pred_a,
        "Model_B_Probability": test_pred_b,
        "Blended_Probability": final_test_prob,
    }).to_csv(test_prob_path, index=False)

    print(f"Saved submission file: {output_path}")
    print(f"Saved fold results file: {fold_results_path}")
    print(f"Saved OOF probability file: {oof_prob_path}")
    print(f"Saved test probability file: {test_prob_path}")
    print("Submission preview:")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
