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
├── notebooks/
├── src/
├── submissions/
├── figures/
├── report/
└── slides/
```

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
├── src/
└── notebooks/
```

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
- matplotlib
- seaborn
- jupyter

## How to Run

To run the final XGBoost + KMeansSMOTE pipeline, use:

```bash
python src/train_XGBoost_KMeansSMOTE.py
```

Alternatively, open the final notebook:

```text
notebooks/xgboost_kmeanssmote_final.ipynb
```

and run all cells from top to bottom.

Generated submission files will be saved in:

```text
submissions/
```

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

## Team Contribution

- Lu Kejie: ExtraTrees model implementation, comparison analysis, and final presentation support.
- Hou Rui: LightGBM benchmark model, validation experiments, and feature engineering support.
- Yang Zhongxing: XGBoost + KMeansSMOTE final modeling, submission generation, and comparison experiments.
- Guo Yitao: CatBoost route analysis, GitHub organization, report integration, and presentation integration.
- Chen Ziheng: HGBC baseline, supporting analysis, and challenge discussion.
- Whole team: EDA discussion, feature engineering design, model comparison, final review, and presentation preparation.

## Notes

The Kaggle raw data files are not uploaded to this repository. Please download them from Kaggle and place them locally before running the code.
