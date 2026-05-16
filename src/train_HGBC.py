import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


RANDOM_SEED = 42
SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Create structured features from PassengerId, Cabin, Name, and spending columns."""

    def fit(self, X, y=None):
        df = X.copy()
        surnames = df["Name"].fillna("Unknown Unknown").str.split().str[-1]
        group_ids = df["PassengerId"].str.split("_").str[0]

        self.group_counts_ = group_ids.value_counts().to_dict()
        self.surname_counts_ = surnames.value_counts().to_dict()
        self.age_median_ = pd.to_numeric(df["Age"], errors="coerce").median()
        return self

    def transform(self, X):
        df = X.copy()

        passenger_parts = df["PassengerId"].str.split("_", expand=True)
        df["GroupId"] = passenger_parts[0]
        df["GroupMemberIndex"] = pd.to_numeric(passenger_parts[1], errors="coerce")

        cabin_parts = df["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
        df["CabinDeck"] = cabin_parts[0].replace("Unknown", np.nan)
        df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        df["CabinSide"] = cabin_parts[2].replace("Unknown", np.nan)

        full_name = df["Name"].fillna("Unknown Unknown")
        df["Surname"] = full_name.str.split().str[-1]
        df["NameLength"] = df["Name"].fillna("").str.len()

        spends = df[SPEND_COLUMNS].fillna(0)
        df["TotalSpend"] = spends.sum(axis=1)
        df["LuxurySpend"] = spends[["Spa", "VRDeck", "FoodCourt"]].sum(axis=1)
        df["BasicSpend"] = spends[["RoomService", "ShoppingMall"]].sum(axis=1)
        df["MaxSingleSpend"] = spends.max(axis=1)
        df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

        age = df["Age"].fillna(self.age_median_)
        df["SpendPerAge"] = df["TotalSpend"] / age.clip(lower=1)
        df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
        df["LuxuryShare"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)

        df["GroupSize"] = df["GroupId"].map(self.group_counts_).fillna(1)
        df["SurnameSize"] = df["Surname"].map(self.surname_counts_).fillna(1)
        df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

        df["CryoSleepFlag"] = df["CryoSleep"].astype("string").fillna("Unknown")
        df["VIPFlag"] = df["VIP"].astype("string").fillna("Unknown")
        df["DeckSide"] = (
            df["CabinDeck"].astype("string").fillna("Unknown")
            + "_"
            + df["CabinSide"].astype("string").fillna("Unknown")
        )
        df["HomeDest"] = (
            df["HomePlanet"].astype("string").fillna("Unknown")
            + "_"
            + df["Destination"].astype("string").fillna("Unknown")
        )
        df["AgeBand"] = pd.cut(
            df["Age"],
            bins=[-1, 12, 18, 25, 40, 60, 120],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Mature", "Senior"],
        )
        df["CabinRegion"] = pd.cut(
            df["CabinNum"],
            bins=[-1, 300, 700, 1200, 2000],
            labels=["Front", "Mid", "Rear", "Far"],
        )
        df["CryoNoSpendMatch"] = ((df["CryoSleep"] == True) & (df["TotalSpend"] == 0)).astype(int)
        df["MissingCount"] = df.isna().sum(axis=1)

        return df.drop(columns=["PassengerId", "Cabin", "Name"])


def build_preprocessor(example_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = example_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in example_frame.columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


@dataclass(frozen=True)
class EnsembleConfig:
    weight: float
    model: HistGradientBoostingClassifier


def build_ensemble(preprocessor: ColumnTransformer) -> list[tuple[float, Pipeline]]:
    configs = [
        EnsembleConfig(
            weight=0.6,
            model=HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.04,
                max_iter=350,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=RANDOM_SEED,
            ),
        ),
        EnsembleConfig(
            weight=0.4,
            model=HistGradientBoostingClassifier(
                max_depth=7,
                learning_rate=0.035,
                max_iter=420,
                min_samples_leaf=18,
                l2_regularization=0.8,
                random_state=7,
            ),
        ),
    ]

    ensemble = []
    for config in configs:
        pipeline = Pipeline(
            steps=[
                ("features", FeatureBuilder()),
                ("preprocess", clone(preprocessor)),
                ("model", config.model),
            ]
        )
        ensemble.append((config.weight, pipeline))
    return ensemble


def blended_predict_proba(models: list[tuple[float, Pipeline]], X: pd.DataFrame) -> np.ndarray:
    probabilities = np.zeros(len(X), dtype=float)
    for weight, model in models:
        probabilities += weight * model.predict_proba(X)[:, 1]
    return probabilities


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.45, 0.551, 0.005):
        score = accuracy_score(y_true, probabilities >= threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(np.round(threshold, 3))

    return best_threshold, best_score


def cross_validate(models: list[tuple[float, Pipeline]], X: pd.DataFrame, y: np.ndarray, folds: int) -> tuple[np.ndarray, float, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    oof_probabilities = np.zeros(len(X), dtype=float)

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y[train_index]

        fitted_models: list[tuple[float, Pipeline]] = []
        for weight, model in models:
            current_model = clone(model)
            current_model.fit(X_train, y_train)
            fitted_models.append((weight, current_model))

        fold_probabilities = blended_predict_proba(fitted_models, X_valid)
        oof_probabilities[valid_index] = fold_probabilities
        fold_score = accuracy_score(y[valid_index], fold_probabilities >= 0.5)
        print(f"Fold {fold_index} accuracy @ 0.50: {fold_score:.5f}")

    best_threshold, best_score = find_best_threshold(y, oof_probabilities)
    return oof_probabilities, best_threshold, best_score


def fit_full_models(models: list[tuple[float, Pipeline]], X: pd.DataFrame, y: np.ndarray) -> list[tuple[float, Pipeline]]:
    fitted_models: list[tuple[float, Pipeline]] = []
    for weight, model in models:
        current_model = clone(model)
        current_model.fit(X, y)
        fitted_models.append((weight, current_model))
    return fitted_models


def main():
    parser = argparse.ArgumentParser(description="Spaceship Titanic solution with engineered features and a dual-HGB ensemble.")
    parser.add_argument(
        "--train",
        default=r"C:\Users\16349\Desktop\ML_WS_Kaggle\train.csv",
        help="Path to Kaggle training CSV.",
    )
    parser.add_argument(
        "--test",
        default=r"C:\Users\16349\Desktop\ML_WS_Kaggle\test.csv",
        help="Path to Kaggle test CSV.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\16349\Documents\New project\submission_spaceship_titanic.csv",
        help="Where to save the submission file.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train = train_df.drop(columns=["Transported"])
    y_train = train_df["Transported"].astype(int).to_numpy()

    example_features = FeatureBuilder().fit_transform(X_train)
    preprocessor = build_preprocessor(example_features)
    ensemble = build_ensemble(preprocessor)

    _, best_threshold, best_score = cross_validate(ensemble, X_train, y_train, folds=args.folds)
    print(f"Best CV accuracy: {best_score:.5f}")
    print(f"Best threshold: {best_threshold:.3f}")
    print("Models: weighted ensemble of two HistGradientBoostingClassifier pipelines")

    fitted_ensemble = fit_full_models(ensemble, X_train, y_train)
    test_probabilities = blended_predict_proba(fitted_ensemble, test_df)
    test_predictions = test_probabilities >= best_threshold

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": test_predictions.astype(bool),
        }
    )
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to: {args.output}")


if __name__ == "__main__":
    main()
