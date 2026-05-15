# GitHub Upload Checklist

Upload these folders and files to the repository root:

- README.md
- requirements.txt
- .gitignore
- notebooks/
- src/
- submissions/
- results/
- docs/
- screenshots/  (after adding Kaggle screenshots)
- figures/      (after adding EDA/model comparison figures)

Do not upload raw Kaggle data:
- data/train.csv
- data/test.csv
- data/sample_submission.csv

Recommended final repository structure:

Spaceship-Titanic-MLW/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 02_catboost_final_model.ipynb
├── src/
│   ├── train_catboost.py            (optional if notebook is used)
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── train_hgbc.py
├── submissions/
│   ├── catboost_final_submission.csv
│   ├── xgboost_kmeans_submission_81295.csv
│   ├── lightgbm_submission.csv
│   ├── hgbc_submission.csv
│   └── hgbc_3model_submission.csv
├── results/
│   ├── model_comparison_template.csv
│   ├── xgboost_cv_scores.csv
│   └── xgboost_submission_summary.csv
└── docs/
    ├── xgboost_method_summary.md
    ├── xgboost_originality_and_compliance.md
    ├── xgboost_course_compliance_check.md
    └── xgboost_high_score_notebook_case_study.md
