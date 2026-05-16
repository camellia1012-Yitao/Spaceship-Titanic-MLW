# Spaceship Titanic Classification Project

This repository contains the source code and project materials for the AI3023 Machine Learning Workshop course project. The task is based on the Kaggle Spaceship Titanic competition, where the objective is to predict whether each passenger was transported to an alternate dimension.

## Project Overview

This project follows a complete machine learning workflow:

1. Data loading and inspection
2. Data preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model training and validation
6. Model comparison
7. Kaggle submission generation
8. Result tracking and presentation preparation

The final selected model is a CatBoost-based blended pipeline. The best public Kaggle score achieved by our team is:

```text
0.81318
```

## Team Information

- Course: AI3023 Machine Learning Workshop
- Competition: Kaggle Spaceship Titanic
- Team name: milkloong @MLW
- Task type: Binary classification
- Evaluation metric: Classification accuracy

## Repository Structure

```text
Spaceship-Titanic-MLW/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── notebooks/
│   ├── catboost_model.ipynb
│   ├── xgboost_kmeanssmote_model.ipynb
│   ├── lightgbm_model.ipynb
│   ├── hgbc_model.ipynb
│   └── extratrees_model.ipynb
|
├── src/
│   ├── preprocessing.py
│   ├── train_CatBoost.py
│   ├── train_XGBoost + KMeansSMOTE.py
│   ├── train_LightGBM.py
│   ├── train_HGBC.py
│   └── train_ExtraTree.py
|
├── submissions/
│   ├── catboost_submission_0.81318.csv
│   ├── XGBoost + KMeansSMOTE_submission_0.81295.csv
│   ├── LightGBM_submission_0.80734.csv
│   ├── ExtraTree_submission_0.80313.csv
│   └── HGBC_submission_0.80266.csv
|
├── results/
│   ├── model_comparison.csv
│   ├── model_ranking.csv
│   └── score_progression.csv
|
├── figures/
│   ├── target_distribution.png
│   ├── cryosleep_transport_rate.png
│   ├── missing_value_ratio.png
│   ├── public_score_comparison.png
│   └── kaggle_score_progression.png
|
├── docs/
│   ├── method_summary.md
│   ├── github_upload_checklist.md
│   └── presentation_outline.md
|
├── report/
│   └── final_report.pdf
|
└── slides/
    └── final_presentation.pptx
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

This project implements and compares five machine learning model routes:

1. CatBoost
2. XGBoost + KMeansSMOTE
3. LightGBM
4. Histogram Gradient Boosting Classifier
5. ExtraTrees

The four main comparison models are CatBoost, XGBoost + KMeansSMOTE, LightGBM, and HGBC. ExtraTrees was added as an additional tree-based comparison model.

## Final Selected Model

The final pipeline is based on CatBoost and includes:

- Missing value handling
- Categorical feature processing
- Passenger group feature extraction
- Cabin feature decomposition
- Spending-related feature engineering
- Name and surname feature extraction
- Interaction feature construction
- Cross-validated model training
- Probability blending between CatBoost feature sets
- Kaggle submission generation

CatBoost was selected as the final model because it achieved the best public leaderboard score among our submitted models and handled the engineered categorical features effectively.

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

4. Run the final CatBoost notebook:

```text
notebooks/catboost_model.ipynb
```

or run the CatBoost training script:

```bash
python "src/train_CatBoost.py"
```

5. The final submission file is stored in:

```text
submissions/catboost_submission_0.81318.csv
```

## Comparison Models

The comparison models can be reviewed using the following notebooks:

```text
notebooks/xgboost_kmeanssmote_model.ipynb
notebooks/lightgbm_model.ipynb
notebooks/hgbc_model.ipynb
notebooks/extratrees_model.ipynb
```

The corresponding scripts are stored in the `src/` folder.

## Main Results

| Rank | Model | Public Score | Role |
|---:|---|---:|---|
| 1 | CatBoost | 0.81318 | Final selected model |
| 2 | XGBoost + KMeansSMOTE | 0.81295 | Strongest comparison model |
| 3 | LightGBM | 0.80734 | Fast boosting benchmark |
| 4 | ExtraTrees | 0.80313 | Additional comparison model |
| 5 | HGBC | 0.80266 | Reproducible baseline |

Final public Kaggle score:

```text
0.81318
```

## Result Files

The model comparison records are stored in:

```text
results/model_comparison.csv
results/model_ranking.csv
results/score_progression.csv
```

These files summarize the public leaderboard scores, model roles, and score progression used in the final report and presentation.

## Reproducibility Notes

The notebooks and scripts use fixed random seeds where applicable. Due to differences in package versions, hardware, or floating-point computation, minor variations in cross-validation scores may occur.

The raw Kaggle data files are intentionally excluded from the repository. Users should download the dataset from Kaggle and place it under the local `data/` folder before running the notebooks or scripts.

## Project Materials

The final project submission includes:

- Source code and notebooks
- Model submission files
- Result tracking CSV files
- Final report
- Presentation slides
- Figures used in the report and presentation

## License

This repository is prepared for academic coursework. The Kaggle dataset should be downloaded from the official competition page and used according to Kaggle's terms.
