# Spaceship Titanic Classification Project

This repository contains the source code and project materials for the AI3023 Machine Learning Workshop course project. The task is based on the Kaggle Spaceship Titanic competition, where the objective is to predict whether each passenger was transported to an alternate dimension.

## Project Overview

This project follows a standard machine learning workflow:

1. Data loading and inspection
2. Data preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model training and validation
6. Model comparison
7. Kaggle submission generation

The final selected model is a CatBoost-based blended pipeline. The best public Kaggle score achieved by our team is:

```text
0.80967
```

## Team Information

- Course: AI3023 Machine Learning Workshop
- Competition: Kaggle Spaceship Titanic
- Team name: milkloong @MLW
- Task type: Binary classification

## Repository Structure

```text
Spaceship-Titanic-MLW/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_catboost_final_model.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_lightgbm_model.ipynb
│   └── 05_hgbc_model.ipynb
|
├── src/
│   ├── preprocessing.py
│   ├── train_catboost.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── train_hgbc.py
|
├── submissions/
│   ├── catboost_last_try_submission.csv
│   ├── xgboost_submission.csv
│   ├── lightgbm_submission.csv
│   └── hgbc_submission.csv
|
├── figures/
│   ├── missing_values.png
│   ├── target_distribution.png
│   ├── spending_distribution.png
│   └── model_comparison.png
|
├── report/
│   └── final_report.pdf
|
└── slides/
    └── presentation.pptx
```

## Dataset

The dataset is from the Kaggle Spaceship Titanic competition.

Please download the following files from Kaggle and place them in a local `data/` directory:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

The raw Kaggle dataset is not included in this repository.

## Implemented Models

This project implements and compares four machine learning models:

1. CatBoost
2. XGBoost
3. LightGBM
4. Histogram Gradient Boosting Classifier

Although several models were implemented for comparison, the final selected submission model is the CatBoost blended pipeline because it achieved the best public leaderboard score among our experiments.

## Final Selected Model

The final pipeline is based on CatBoost and includes:

- Missing value handling
- Categorical feature processing
- Passenger group feature extraction
- Cabin feature decomposition
- Spending-related feature engineering
- Cross-validated model training
- Probability blending between CatBoost feature sets
- Kaggle submission generation

## Environment Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

The project was developed with Python 3.10+.

## How to Run

1. Download the Kaggle dataset.
2. Create a local `data/` folder.
3. Place the competition files into the folder:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

4. Open the final CatBoost notebook:

```text
notebooks/02_catboost_final_model.ipynb
```

5. Run all cells from top to bottom.
6. The final submission file will be generated as:

```text
catboost_last_try_submission.csv
```

## Comparison Models

The comparison models can be reproduced using the following notebooks:

```text
notebooks/03_xgboost_model.ipynb
notebooks/04_lightgbm_model.ipynb
notebooks/05_hgbc_model.ipynb
```

These models are included to support the required model comparison and experimental analysis.

## Main Results

| Model | Role | Notes |
|---|---|---|
| CatBoost | Final selected model | Best public leaderboard score |
| XGBoost | Comparison model | Strong gradient boosting baseline |
| LightGBM | Comparison model | Efficient histogram-based boosting model |
| HGBC | Comparison model | Scikit-learn histogram gradient boosting baseline |

Final public Kaggle score:

```text
0.80967
```

## Reproducibility Notes

The notebook uses fixed random seeds where applicable. Due to small differences in package versions, hardware, or floating-point computation, minor variations in cross-validation scores may occur.

## Project Materials

The final project submission includes:

- Source code and notebooks
- Final Kaggle submission file
- Final report
- Presentation slides
- Figures used in the report and presentation

## License

This repository is prepared for academic coursework. The Kaggle dataset should be downloaded from the official competition page and used according to Kaggle's terms.
