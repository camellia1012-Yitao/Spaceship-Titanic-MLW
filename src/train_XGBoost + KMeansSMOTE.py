from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from xgboost import XGBClassifier


TARGET = "Transported"
EXPENSE_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM_COLS = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
CAT_COLS = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
DROP_LIST = [
    "ShoppingMall",
    "Age",
    "CryoSleep_True",
    "HomePlanet_Earth",
    "HomePlanet_Europa",
    "VIP_True",
    "HomePlanet_Mars",
    "Destination_PSO J318.5-22",
    "VIP_False",
    "Destination_55 Cancri e",
    "FoodCourt",
    "Destination_TRAPPIST-1e",
]

XGB_PARAMS = {
    "reg_lambda": 3.0610042624477543,
    "reg_alpha": 4.581902571574289,
    "colsample_bytree": 0.9241969052729379,
    "subsample": 0.9527591724824661,
    "learning_rate": 0.06672065863100594,
    "n_estimators": 730,
    "max_depth": 5,
    "min_child_weight": 1,
    "num_parallel_tree": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}


def notebook_preprocess(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    full = pd.concat([train, test], ignore_index=True)

    full.loc[full["CryoSleep"].eq(True), EXPENSE_COLS] = 0
    full["Expenses"] = full[EXPENSE_COLS].sum(axis=1)
    full.loc[full["CryoSleep"].isna() & full["Expenses"].eq(0), "CryoSleep"] = True

    full["Name"] = full["Name"].fillna("Unknown Unknown")
    full["Room"] = full["PassengerId"].str[:4]

    for col in ["VIP", "Cabin", "HomePlanet", "Destination"]:
        guide = full[["Room", col]].dropna().drop_duplicates("Room")
        full = full.merge(guide, how="left", on="Room", suffixes=("", "_guide"))
        full[col] = full[col].fillna(full[f"{col}_guide"])
        full = full.drop(columns=[f"{col}_guide"])

    cabin = full["Cabin"].str.split("/", expand=True)
    full["Cabin_1"] = cabin[0]
    full["Cabin_2"] = cabin[1]
    full["Cabin_3"] = cabin[2]

    selected = full[NUM_COLS + CAT_COLS + [TARGET]].copy()
    num_imp = SimpleImputer(strategy="mean")
    cat_imp = SimpleImputer(strategy="most_frequent")
    selected[NUM_COLS] = pd.DataFrame(num_imp.fit_transform(selected[NUM_COLS]), columns=NUM_COLS)
    selected[CAT_COLS] = pd.DataFrame(cat_imp.fit_transform(selected[CAT_COLS]), columns=CAT_COLS)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = pd.DataFrame(encoder.fit_transform(selected[CAT_COLS]), columns=encoder.get_feature_names_out(CAT_COLS))
    selected = pd.concat([selected.drop(columns=CAT_COLS), encoded], axis=1)

    train_processed = selected[selected[TARGET].notna()].copy()
    test_processed = selected[selected[TARGET].isna()].drop(columns=[TARGET]).copy()
    y = train_processed[TARGET].astype(int)
    X = train_processed.drop(columns=[TARGET])
    X, y = shuffle(X, y, random_state=seed)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    test_processed = test_processed.reset_index(drop=True)

    drop_cols = [col for col in DROP_LIST if col in X.columns]
    X = X.drop(columns=drop_cols)
    test_processed = test_processed.drop(columns=[col for col in drop_cols if col in test_processed.columns])
    return X, y, test_processed


def resample(X: pd.DataFrame, y: pd.Series, method: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if method == "none":
        return X, y
    if method == "smote":
        return SMOTE(sampling_strategy=1, random_state=seed).fit_resample(X, y)
    if method == "kmeans":
        sampler = KMeansSMOTE(
            sampling_strategy=1,
            random_state=seed,
            cluster_balance_threshold=0.01,
        )
        return sampler.fit_resample(X, y)
    raise ValueError(method)


def cv_score(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2140)
    scores = cross_val_score(XGBClassifier(**XGB_PARAMS), X, y, scoring="accuracy", cv=cv, n_jobs=1)
    return float(scores.mean()), float(scores.std())


def write_submission(path: Path, passenger_id: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, object]:
    pred = prob >= threshold
    pd.DataFrame({"PassengerId": passenger_id, TARGET: pred.astype(bool)}).to_csv(path, index=False)
    return {"file": path.name, "threshold": threshold, "true_rate": float(pred.mean())}


def threshold_for_true_rate(prob: np.ndarray, true_rate: float) -> float:
    return float(np.quantile(prob, 1.0 - true_rate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="submissions/xgb_optuna_style_kmeans")
    parser.add_argument("--seeds", default="42,123,2023,2140")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    summary_rows = []
    score_rows = []
    for seed in [int(item.strip()) for item in args.seeds.split(",") if item.strip()]:
        X, y, X_test = notebook_preprocess(train, test, seed=seed)
        for method in ["none", "smote", "kmeans"]:
            try:
                X_fit, y_fit = resample(X, y, method, seed)
            except Exception as exc:
                score_rows.append({"seed": seed, "method": method, "cv_mean": np.nan, "cv_std": np.nan, "error": str(exc)})
                print(f"seed={seed} method={method} failed: {exc}")
                continue
            mean, std = cv_score(X_fit, y_fit)
            score_rows.append({"seed": seed, "method": method, "cv_mean": mean, "cv_std": std, "error": ""})
            print(f"seed={seed} method={method} cv={mean:.5f} +/- {std:.5f} shape={X_fit.shape}")

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_fit, y_fit)
            prob = model.predict_proba(X_test)[:, 1]
            pd.DataFrame({"PassengerId": sample["PassengerId"], "probability": prob}).to_csv(
                output_dir / f"prob_seed{seed}_{method}.csv",
                index=False,
            )
            for threshold in [0.50, 0.51, 0.525, 0.535, 0.545]:
                summary_rows.append(
                    write_submission(
                        output_dir / f"submission_xgb_optuna_seed{seed}_{method}_thr{int(threshold * 1000):03d}.csv",
                        sample["PassengerId"],
                        prob,
                        threshold,
                    )
                )
            for true_rate in [0.4865, 0.488, 0.490, 0.492]:
                threshold = threshold_for_true_rate(prob, true_rate)
                summary_rows.append(
                    write_submission(
                        output_dir / f"submission_xgb_optuna_seed{seed}_{method}_rate{int(true_rate * 10000):04d}.csv",
                        sample["PassengerId"],
                        prob,
                        threshold,
                    )
                )

    pd.DataFrame(score_rows).to_csv(output_dir / "cv_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "submission_summary.csv", index=False)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
