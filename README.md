# Spaceship Titanic Classification Project

This repository contains the code and materials for the AI3023 Machine Learning Workshop course project.

## Project Information

- Course: AI3023 Machine Learning Workshop
- Competition: Kaggle Spaceship Titanic
- Team: milkloong@MLW
- Task: Binary Classification
- Target variable: `Transported`
- Best public Kaggle score: `0.81295`
- Final selected model: XGBoost + KMeansSMOTE

## Project Overview

The goal of this project is to predict whether a passenger was transported to an alternate dimension based on passenger information, travel status, cabin location, and onboard spending behavior.

We built a complete machine learning workflow, including:

1. Exploratory Data Analysis
2. Data preprocessing
3. Feature engineering
4. Model training and validation
5. Model comparison
6. Kaggle submission generation

Five model routes were implemented and compared:

- CatBoost
- XGBoost + KMeansSMOTE
- LightGBM
- ExtraTrees
- HistGradientBoostingClassifier

The final selected model is XGBoost + KMeansSMOTE because it achieved the strongest reproducible public Kaggle score among our implemented routes.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
├── figures/
├── notebooks/
├── result/
├── src/
└── submissions/
```

## Folder Description

### `docs/`

Contains final project documents, such as the final report, presentation slides, and Kaggle submission log screenshots.

### `figures/`

Contains figures used in the report, including EDA plots, model comparison charts, confusion matrix, ROC curve, and feature importance plots.

### `notebooks/`

Contains Jupyter notebooks used for experiments, validation, visualization, and final model reproduction.

### `result/`

Contains model result files, validation outputs, probability files, and intermediate experiment summaries.

### `src/`

Contains Python source code for preprocessing and model training.

Current training scripts include:

```text
preprocessing.py
train_CatBoost.py
train_ExtraTree.py
train_HGBC.py
train_LightGBM.py
train_XGBoost_KMeansSMOTE.py
```

### `submissions/`

Contains generated Kaggle submission files. The final submission file should be placed here.

## Data Preparation

The original Kaggle dataset files are not included in this repository.

Please download the dataset from the Kaggle Spaceship Titanic competition page:

https://www.kaggle.com/competitions/spaceship-titanic

Required files:

```text
train.csv
test.csv
sample_submission.csv
```

Place these three CSV files in the project root directory before running the final model.

Expected local structure:

```text
.
├── train.csv
├── test.csv
├── sample_submission.csv
├── README.md
├── requirements.txt
├── src/
└── submissions/
```

The Kaggle raw data files should not be uploaded to GitHub.

## Environment Setup

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- numpy
- pandas
- scikit-learn
- xgboost
- imbalanced-learn
- lightgbm
- catboost
- matplotlib
- seaborn
- jupyter

## How to Run the Final Model

To run the final XGBoost + KMeansSMOTE pipeline, use:

```bash
python src/train_XGBoost_KMeansSMOTE.py
```

If the filename still contains spaces or the plus sign, rename it first:

```text
train_XGBoost + KMeansSMOTE.py
```

to:

```text
train_XGBoost_KMeansSMOTE.py
```

The generated submission files will be saved in:

```text
submissions/
```

## How to Run Other Model Routes

CatBoost:

```bash
python src/train_CatBoost.py
```

LightGBM:

```bash
python src/train_LightGBM.py
```

ExtraTrees:

```bash
python src/train_ExtraTree.py
```

HistGradientBoostingClassifier:

```bash
python src/train_HGBC.py
```

Some scripts may generate additional validation outputs or probability files in the `result/` or `submissions/` folder.

## Final Result

The best public Kaggle score achieved by our final selected model was:

```text
0.81295
```

The final model was selected based on:

- Kaggle public score
- Reproducibility
- Pipeline clarity
- Model suitability
- Practical submission reliability

## Model Summary

| Model | Public Score | Role |
|---|---:|---|
| XGBoost + KMeansSMOTE | 0.81295 | Final selected model |
| Improved CatBoost | 0.80967 | Strong categorical-feature route |
| LightGBM | 0.80734 | Fast boosting benchmark |
| ExtraTrees | 0.80313 | Tree ensemble comparison |
| HGBC | 0.80266 | Reproducible baseline |

## Feature Engineering Summary

The project used several groups of engineered features:

- Passenger group features from `PassengerId`
- Cabin location features from `Cabin`
- Spending behavior features from service expense columns
- Family-related features from `Name`
- Interaction features such as CryoSleep-spending and HomePlanet-deck combinations

These features were designed based on EDA findings and were used differently across the five model routes.

## Report and Presentation

The final report, presentation slides, and Kaggle submission screenshots should be placed in:

```text
docs/
```

Recommended files:

```text
docs/final_report.pdf
docs/final_presentation.pptx
docs/kaggle_submission_log.png
```

## Team Contribution

- Lu Kejie: ExtraTrees model implementation, comparison analysis, and final presentation support.
- Hou Rui: LightGBM benchmark model, validation experiments, and feature engineering support.
- Yang Zhongxing: XGBoost + KMeansSMOTE final modeling, submission generation, and comparison experiments.
- Guo Yitao: CatBoost route analysis, GitHub organization, report integration, and presentation integration.
- Chen Ziheng: HGBC baseline, supporting analysis, and challenge discussion.
- Whole team: EDA discussion, feature engineering design, model comparison, final review, and presentation preparation.

## Notes

This repository does not include the original Kaggle dataset files. Please download `train.csv`, `test.csv`, and `sample_submission.csv` from Kaggle and place them locally before running the code.

The final ranking is not reported because the leaderboard ranking was not available at the time of final report preparation.
