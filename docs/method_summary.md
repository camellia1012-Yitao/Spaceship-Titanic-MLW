# Method Summary

## Project Task

This project is based on the Kaggle Spaceship Titanic binary classification competition.  
The goal is to predict whether each passenger was transported to an alternate dimension.

- Task type: binary classification
- Target variable: `Transported`
- Evaluation metric: classification accuracy
- Final selected model: CatBoost-based blended pipeline
- Best public Kaggle score: **0.81318**

## Dataset and Feature Engineering

The raw dataset contains passenger information, travel information, cabin records, and onboard spending behavior.  
Feature engineering was a key part of the project because several raw columns contained hidden structure.

Main engineered feature groups:

| Raw Source | Engineered Features | Purpose |
|---|---|---|
| `PassengerId` | `GroupID`, `GroupMember`, `GroupSize`, `IsSolo` | Capture passenger group structure |
| `Cabin` | `Deck`, `CabinNum`, `Side`, `Deck_Side`, `CabinRegion` | Capture cabin location information |
| Spending columns | `TotalSpend`, `NoSpend`, `LuxurySpend`, `BasicSpend`, `LogTotalSpend` | Capture onboard activity and spending behavior |
| `Name` | `Surname`, `SurnameSize` | Capture possible family-related signals |
| Interactions | `Cryo_NoSpend`, `HomePlanet_Deck`, `Destination_Deck`, `Deck_Side` | Capture combined effects between important variables |

EDA showed that the target distribution was nearly balanced and that `CryoSleep`, spending behavior, cabin information, and passenger group structure were important signals.

## Implemented Models

The project implemented and compared five model routes:

| Model | Public Score | Role |
|---|---:|---|
| CatBoost | **0.81318** | Final selected model |
| XGBoost + KMeansSMOTE | 0.81295 | Strongest comparison model |
| LightGBM | 0.80734 | Fast boosting benchmark |
| ExtraTrees | 0.80313 | Additional tree-based comparison model |
| HGBC | 0.80266 | Reproducible scikit-learn baseline |

## Final Model: CatBoost-Based Blended Pipeline

CatBoost was selected as the final model because it achieved the best public Kaggle score and matched the structure of the dataset well.

Main reasons for selecting CatBoost:

1. It handles categorical variables more naturally than one-hot based pipelines.
2. It worked well with engineered group, cabin, spending, family, and interaction features.
3. It slightly outperformed the strong XGBoost + KMeansSMOTE route.
4. It provided the best final public leaderboard score among submitted models.

The final CatBoost route used:

- Shared feature engineering through `preprocessing.py`
- Two CatBoost feature-set variants
- 5-fold stratified cross-validation
- Out-of-fold prediction tracking
- Probability blending
- Final submission generation

## Score Progression

The project did not improve in a perfectly monotonic way. Several intermediate models produced lower scores, but they provided useful comparison evidence.

| Stage | Model or Step | Public Score |
|---:|---|---:|
| 1 | Early CatBoost | 0.80710 |
| 2 | Improved CatBoost | 0.80967 |
| 3 | HGBC | 0.80266 |
| 4 | LightGBM | 0.80734 |
| 5 | ExtraTrees | 0.80313 |
| 6 | XGBoost + KMeansSMOTE | 0.81295 |
| 7 | Final CatBoost | **0.81318** |

## Lessons Learned

Key lessons from the project:

- Feature engineering was more important than simply adding model complexity.
- CatBoost was especially suitable for this categorical-heavy tabular task.
- XGBoost + KMeansSMOTE showed that compact feature design and resampling can be highly competitive.
- Local validation and public leaderboard performance did not always align perfectly.
- Reproducible code organization, saved submissions, and result tracking were important for final reporting and presentation.
