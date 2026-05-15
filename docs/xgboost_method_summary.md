# Method Summary

## Task

Predict whether each passenger in the Spaceship Titanic test set was transported. The evaluation metric is classification accuracy.

## Final Score

```text
Kaggle public score: 0.81295
```

Final submission file:

```text
submission_xgb_kmeans_seed2023_thr500_score_81295.csv
```

## Feature Engineering

The pipeline first combines train and test for unsupervised preprocessing only.

### Expense Features

The five spending columns are:

```text
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
```

The total expense feature is:

```text
Expenses = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
```

If `CryoSleep == True`, all spending columns are set to zero because passengers in cryosleep are confined to their cabins.

If `CryoSleep` is missing and `Expenses == 0`, `CryoSleep` is filled as `True`.

### Group-Based Imputation

The `PassengerId` prefix is used as the passenger room/group id:

```text
Room = PassengerId[:4]
```

Within the same room/group, the model fills missing values for:

```text
VIP, Cabin, HomePlanet, Destination
```

This is based on the observation that passengers in the same group often share travel details.

### Cabin Features

`Cabin` is split into:

```text
Cabin_1 = deck
Cabin_2 = cabin number
Cabin_3 = side
```

The final compact model uses `Cabin_1` and `Cabin_3`.

### Final Feature Set

Numerical features:

```text
ShoppingMall, FoodCourt, RoomService, Spa, VRDeck, Expenses, Age
```

Categorical features:

```text
CryoSleep, Cabin_1, Cabin_3, VIP, HomePlanet, Destination
```

Categorical features are one-hot encoded after missing values are imputed.

### Dropped Features

After one-hot encoding, the following low-importance columns are removed:

```text
ShoppingMall
Age
CryoSleep_True
HomePlanet_Earth
HomePlanet_Europa
VIP_True
HomePlanet_Mars
Destination_PSO J318.5-22
VIP_False
Destination_55 Cancri e
FoodCourt
Destination_TRAPPIST-1e
```

## Model

The final model is XGBoost:

```text
reg_lambda = 3.0610042624477543
reg_alpha = 4.581902571574289
colsample_bytree = 0.9241969052729379
subsample = 0.9527591724824661
learning_rate = 0.06672065863100594
n_estimators = 730
max_depth = 5
min_child_weight = 1
num_parallel_tree = 1
```

The selected run uses:

```text
seed = 2023
resampling = KMeansSMOTE
threshold = 0.500
```

## Ablation Results

The script compares no resampling, ordinary SMOTE, and KMeansSMOTE across several random seeds.

## Research Progression

This final version has a clear connection to the earlier non-KMeansSMOTE experiments.

First, we built baseline pipelines with standard preprocessing, one-hot encoding, and common machine learning models such as XGBoost, LightGBM, CatBoost, and ExtraTrees. The robust LightGBM route achieved a public score of `0.80757`, but further parameter tuning and broad OOF blending did not improve public leaderboard performance consistently.

This suggested that simply increasing model complexity was not the main bottleneck. We then returned to a compact XGBoost pipeline using group, cabin, and spending features. The compact XGBoost version without resampling became the baseline for this branch.

Next, we compared three data-balancing strategies:

```text
1. no resampling
2. ordinary SMOTE
3. KMeansSMOTE
```

The motivation was that Spaceship Titanic contains local passenger subgroups, such as cabin decks, cryosleep/no-spending passengers, and group-based travel patterns. Ordinary SMOTE may interpolate samples across less meaningful regions, while KMeansSMOTE first clusters the feature space and then synthesizes samples within local clusters.

For seed `2023`, the comparison was:

```text
XGB compact, no resampling: 0.80823 CV
XGB compact + SMOTE:        0.80665 CV
XGB compact + KMeansSMOTE:  0.80989 CV
```

The final public leaderboard improvement came from this transition from the compact XGBoost baseline to compact XGBoost with KMeansSMOTE:

```text
seed=2023, KMeansSMOTE, threshold=0.500: public score 0.81295
```

Therefore, the final method can be described as the result of a research path: baseline models -> robust LightGBM -> failed high-complexity blending -> compact XGBoost baseline -> resampling ablation -> XGBoost + KMeansSMOTE.

Best local CV from this experiment:

```text
seed=2140, KMeansSMOTE: 0.81114
```

The best Kaggle public result came from:

```text
seed=2023, KMeansSMOTE, threshold=0.500: 0.81295
```

## Interpretation

The improvement came from using a compact feature representation and KMeansSMOTE, rather than a large multi-model ensemble. This route also proved more reliable on Kaggle public leaderboard than broader OOF blending experiments.
