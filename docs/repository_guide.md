# Repository Guide

This document explains the structure of the Spaceship Titanic project repository.

## Root Files

| File | Purpose |
|---|---|
| `README.md` | Main project overview and instructions |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Files and folders excluded from GitHub |

## Folder Structure

```text
Spaceship-Titanic-MLW/
|
├── notebooks/
├── src/
├── submissions/
├── results/
├── figures/
├── docs/
├── slides/
└── report/
```

## `notebooks/`

This folder contains Jupyter notebooks for reviewing and demonstrating each model route.

Recommended contents:

```text
catboost_model.ipynb
xgboost_kmeanssmote_model.ipynb
lightgbm_model.ipynb
hgbc_model.ipynb
extratrees_model.ipynb
```

The notebooks are mainly for explanation, review, and reproduction.

## `src/`

This folder contains Python scripts used for model training and reusable preprocessing.

Recommended contents:

```text
preprocessing.py
train_CatBoost.py
train_XGBoost + KMeansSMOTE.py
train_LightGBM.py
train_HGBC.py
train_ExtraTree.py
```

`preprocessing.py` contains shared feature engineering utilities.  
The model training scripts generate predictions and submission files.

## `submissions/`

This folder contains Kaggle submission CSV files.

Recommended contents:

```text
catboost_submission_0.81318.csv
XGBoost + KMeansSMOTE_submission_0.81295.csv
LightGBM_submission_0.80734.csv
ExtraTree_submission_0.80313.csv
HGBC_submission_0.80266.csv
```

The final selected submission is:

```text
catboost_submission_0.81318.csv
```

## `results/`

This folder contains result tracking files used in the report and presentation.

Recommended contents:

```text
model_comparison.csv
model_ranking.csv
score_progression.csv
```

These files summarize model scores, ranking, roles, and score progression.

## `figures/`

This folder contains figures used in the report and presentation.

Recommended contents:

```text
target_distribution.png
missing_value_ratio.png
cryosleep_transport_rate.png
public_score_comparison.png
kaggle_score_progression.png
```

These figures should be based on real data or recorded project results.

## `docs/`

This folder contains supplementary project documentation.

Recommended contents:

```text
method_summary.md
repository_guide.md
```

These files help readers understand the methodology and repository organization.

## `slides/`

This folder contains the final presentation slides.

Recommended contents:

```text
final_presentation.pptx
```

## `report/`

This folder contains the final project report.

Recommended contents:

```text
final_report.pdf
```

## Data Folder

The raw Kaggle data should be placed locally in:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

The `data/` folder should not be uploaded to GitHub because the Kaggle dataset should be downloaded from the official competition page.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final model script from the project root:

```bash
python "src/train_CatBoost.py"
```

Or open the final model notebook:

```text
notebooks/catboost_model.ipynb
```

## Notes

- Use the CatBoost submission as the final selected result.
- Keep result CSV files in `results/` so the model comparison is easy to verify.
- Keep figures in `figures/` so the report and presentation have traceable visual evidence.
