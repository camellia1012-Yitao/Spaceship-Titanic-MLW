"""
Shared preprocessing and feature engineering utilities for the
Spaceship Titanic classification project.

This module is intentionally model-agnostic. It creates reusable tabular
features that can be consumed by CatBoost, XGBoost, LightGBM, and HGBC
pipelines. Model-specific encoding, scaling, and imputation can still be
handled inside each training script or notebook.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "Transported"
ID_COL = "PassengerId"

SPENDING_COLS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

BASE_CATEGORICAL_COLS = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
]


# -----------------------------------------------------------------------------
# Basic validation
# -----------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy so feature functions do not mutate inputs."""
    return df.copy(deep=True)


# -----------------------------------------------------------------------------
# Feature engineering blocks
# -----------------------------------------------------------------------------

def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create group-related features from PassengerId.

    PassengerId format is usually "gggg_pp", where gggg is the group id and
    pp is the passenger number inside the group.
    """
    _require_columns(df, [ID_COL])
    df = copy_dataframe(df)

    passenger_parts = df[ID_COL].astype(str).str.split("_", expand=True)
    df["GroupID"] = passenger_parts[0]
    df["GroupMember"] = pd.to_numeric(passenger_parts[1], errors="coerce").fillna(0).astype(int)
    df["GroupSize"] = df.groupby("GroupID")[ID_COL].transform("count").astype(int)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
    df["IsGroupMemberFirst"] = (df["GroupMember"] == 1).astype(int)

    return df


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin into deck, number, side, and simple derived location features."""
    _require_columns(df, ["Cabin"])
    df = copy_dataframe(df)

    cabin_parts = df["Cabin"].astype("string").str.split("/", expand=True)
    df["Deck"] = cabin_parts[0].fillna("Unknown").astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["Side"] = cabin_parts[2].fillna("Unknown").astype(str)

    df["Deck_Side"] = df["Deck"].astype(str) + "_" + df["Side"].astype(str)
    df["CabinNumMissing"] = df["CabinNum"].isna().astype(int)

    # Coarse cabin location buckets. Missing values are kept as "Unknown".
    cabin_num_filled = df["CabinNum"].fillna(-1)
    df["CabinRegion"] = pd.cut(
        cabin_num_filled,
        bins=[-2, -0.5, 300, 600, 900, 1200, 1500, np.inf],
        labels=["Unknown", "R0", "R1", "R2", "R3", "R4", "R5"],
    ).astype(str)

    return df


def add_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spending aggregates and behavior indicators."""
    _require_columns(df, SPENDING_COLS)
    df = copy_dataframe(df)

    for col in SPENDING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    spending_zero = df[SPENDING_COLS].fillna(0)
    df["TotalSpend"] = spending_zero.sum(axis=1)
    df["BasicSpend"] = spending_zero[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1)
    df["LuxurySpend"] = spending_zero[["Spa", "VRDeck"]].sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["AnySpend"] = (df["TotalSpend"] > 0).astype(int)
    df["SpendingMissingCount"] = df[SPENDING_COLS].isna().sum(axis=1)

    # Ratios are safe because the denominator is protected from zero.
    denom = df["TotalSpend"].replace(0, np.nan)
    df["LuxurySpendRatio"] = (df["LuxurySpend"] / denom).fillna(0)
    df["BasicSpendRatio"] = (df["BasicSpend"] / denom).fillna(0)

    return df


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract surname-based family signals from Name."""
    _require_columns(df, ["Name"])
    df = copy_dataframe(df)

    name = df["Name"].astype("string")
    df["Surname"] = name.str.split().str[-1].fillna("Unknown").astype(str)
    df["NameMissing"] = df["Name"].isna().astype(int)
    df["SurnameSize"] = df.groupby("Surname")[ID_COL].transform("count") if ID_COL in df.columns else df.groupby("Surname")["Surname"].transform("count")
    df["SurnameSize"] = df["SurnameSize"].astype(int)
    df["IsFamilyGroup"] = (df["SurnameSize"] > 1).astype(int)

    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple age-related features."""
    _require_columns(df, ["Age"])
    df = copy_dataframe(df)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["IsChild"] = (df["Age"] < 13).astype(int)
    df["IsTeen"] = ((df["Age"] >= 13) & (df["Age"] < 18)).astype(int)
    df["IsAdult"] = ((df["Age"] >= 18) & (df["Age"] < 60)).astype(int)
    df["IsSenior"] = (df["Age"] >= 60).astype(int)

    age_filled = df["Age"].fillna(-1)
    df["AgeGroup"] = pd.cut(
        age_filled,
        bins=[-2, -0.5, 12, 17, 30, 45, 60, np.inf],
        labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features motivated by EDA and model performance."""
    required = ["CryoSleep", "HomePlanet", "Destination", "Deck", "Side", "NoSpend", "TotalSpend"]
    _require_columns(df, required)
    df = copy_dataframe(df)

    cryo = df["CryoSleep"].astype("string").fillna("Unknown")
    home = df["HomePlanet"].astype("string").fillna("Unknown")
    dest = df["Destination"].astype("string").fillna("Unknown")
    deck = df["Deck"].astype("string").fillna("Unknown")
    side = df["Side"].astype("string").fillna("Unknown")

    df["Cryo_NoSpend"] = cryo.astype(str) + "_" + df["NoSpend"].astype(str)
    df["HomePlanet_Deck"] = home.astype(str) + "_" + deck.astype(str)
    df["Destination_Deck"] = dest.astype(str) + "_" + deck.astype(str)
    df["Deck_Side"] = deck.astype(str) + "_" + side.astype(str)
    df["HomePlanet_Destination"] = home.astype(str) + "_" + dest.astype(str)

    # Group-level spending signals.
    if "GroupID" in df.columns:
        df["GroupTotalSpend"] = df.groupby("GroupID")["TotalSpend"].transform("sum")
        df["GroupAvgSpend"] = df.groupby("GroupID")["TotalSpend"].transform("mean")
    else:
        df["GroupTotalSpend"] = df["TotalSpend"]
        df["GroupAvgSpend"] = df["TotalSpend"]

    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact missing-value indicators before imputation or encoding."""
    df = copy_dataframe(df)
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    df["MissingCount"] = df[feature_cols].isna().sum(axis=1)
    df["HasMissing"] = (df["MissingCount"] > 0).astype(int)
    return df


# -----------------------------------------------------------------------------
# Full feature pipeline
# -----------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the shared engineered feature table for one dataframe.

    This function keeps original columns and appends engineered features.
    It does not encode categorical variables because different models use
    different preprocessing strategies.
    """
    df = copy_dataframe(df)

    df = add_group_features(df)
    df = add_cabin_features(df)
    df = add_spending_features(df)
    df = add_name_features(df)
    df = add_age_features(df)
    df = add_interaction_features(df)
    df = add_missing_indicators(df)

    return df


def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features for train and test together, then split them back.

    Combining train and test here is useful for non-target-derived structural
    features such as GroupSize and SurnameSize. The target is never used to
    create features.
    """
    train_df = copy_dataframe(train_df)
    test_df = copy_dataframe(test_df)

    train_df["__is_train__"] = 1
    test_df["__is_train__"] = 0

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = build_features(combined)

    train_features = combined[combined["__is_train__"] == 1].drop(columns=["__is_train__"])
    test_features = combined[combined["__is_train__"] == 0].drop(columns=["__is_train__"])

    train_features = train_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)

    return train_features, test_features


# -----------------------------------------------------------------------------
# Model input helpers
# -----------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame, drop_cols: Optional[Iterable[str]] = None) -> list[str]:
    """Return model feature columns after excluding identifiers, target, and custom columns."""
    default_drop = {TARGET_COL, ID_COL, "Name", "Cabin"}
    if drop_cols is not None:
        default_drop.update(drop_cols)
    return [col for col in df.columns if col not in default_drop]


def get_categorical_columns(df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> list[str]:
    """Detect categorical columns for CatBoost or encoder-based pipelines."""
    cols = list(feature_cols) if feature_cols is not None else list(df.columns)
    categorical_cols = []
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
    return categorical_cols


def prepare_catboost_frame(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    drop_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[int]]:
    """
    Prepare feature matrices and categorical indices for CatBoost.

    CatBoost can handle categorical columns directly, but missing categorical
    values should be represented as strings.
    """
    if TARGET_COL not in train_features.columns:
        raise ValueError(f"{TARGET_COL} column is required in train_features.")

    feature_cols = get_feature_columns(train_features, drop_cols=drop_cols)
    X_train = train_features[feature_cols].copy()
    y_train = train_features[TARGET_COL].astype(int)
    X_test = test_features[feature_cols].copy()

    categorical_cols = get_categorical_columns(X_train, feature_cols)
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str).fillna("Missing")
        X_test[col] = X_test[col].astype(str).fillna("Missing")

    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
    return X_train, y_train, X_test, categorical_cols, cat_indices


def prepare_submission(
    passenger_ids: pd.Series,
    predictions: Iterable[bool | int | float],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a Kaggle-ready submission dataframe.

    Predictions are converted to Boolean labels to match the competition format.
    """
    pred_series = pd.Series(predictions)
    if pred_series.dtype != bool:
        pred_series = pred_series.astype(int).astype(bool)

    submission = pd.DataFrame({
        ID_COL: passenger_ids.values,
        TARGET_COL: pred_series.values,
    })

    if output_path is not None:
        submission.to_csv(output_path, index=False)

    return submission
