# Course Compliance Check

This document checks the `xgb_kmeans_81295_package` against the AI3023 project requirements.

## Requirement Checklist

### Kaggle Competition

Status: compliant.

The project uses the required Kaggle competition:

```text
Spaceship Titanic
```

The final submission file is:

```text
submission/submission_xgb_kmeans_seed2023_thr500_score_81295.csv
```

Known public leaderboard score:

```text
0.81295
```

### Data Preparation And Preprocessing

Status: compliant.

The code performs:

- missing value handling
- categorical feature encoding
- group-based imputation using `PassengerId`
- cabin feature extraction
- spending consistency cleanup
- KMeansSMOTE resampling

### EDA Requirement

Status: must be completed in the report.

This package contains modeling code and experiment outputs. The final report still needs EDA figures and explanations, such as:

- target distribution
- missing value patterns
- spending distributions by `Transported`
- categorical feature relationships
- cabin deck/side patterns

### Model Comparison Requirement

Status: mostly satisfied by project experiments, but must be summarized in the report.

The project has tried multiple models and directions:

- Logistic/baseline models
- XGBoost
- LightGBM
- CatBoost
- ExtraTrees
- OOF blending
- native categorical LightGBM
- XGB + SMOTE / KMeansSMOTE

The final report should include a comparison table with CV score, Kaggle public score where available, and short discussion.

### Hyperparameter Tuning

Status: compliant.

The final XGBoost parameters are tuned parameters inspired by Optuna-style searches, and additional local seed/resampling comparisons were conducted.

### Proper Validation

Status: partially compliant; must be explained carefully.

The script records 10-fold stratified CV for the compact XGB pipeline. The report should also mention that additional experiments used group-aware CV in the LightGBM route. Since passengers in the same group can be correlated, this limitation should be acknowledged.

### Code Quality And Reproducibility

Status: compliant.

The package contains:

- runnable source code
- dependency list
- README instructions
- final submission file
- CV result tables
- method summary

### Plagiarism / Originality

Status: acceptable if cited and explained honestly.

Risk level: medium if submitted without attribution; low-to-medium if cited properly.

Reason:

- The final method is inspired by public Kaggle notebooks, especially the compact XGB + drop-list + KMeansSMOTE idea.
- The submitted code is not copied cell-by-cell; it is a standalone implementation with functions, seed comparisons, resampling ablations, and documentation.
- The report should explicitly cite the public notebooks as references under "Related Work / Existing Techniques".

Recommended wording:

```text
We reviewed public Kaggle notebooks to identify common preprocessing and feature-engineering strategies. We independently reimplemented a compact XGBoost pipeline, compared resampling methods including no resampling, SMOTE, and KMeansSMOTE, and selected the final configuration based on our own experiments and Kaggle validation.
```

Avoid wording such as:

```text
All feature engineering ideas were designed entirely from scratch.
```

### Similarity And AI Content Checks

Status: requires human rewriting before final submission.

The PDF states:

```text
similarity check must not exceed 20%
AI generation content check must not exceed 25%
```

Therefore:

- Do not paste these markdown documents directly as the final report.
- Use them as notes.
- Rewrite the report in the team's own words.
- Add your own figures, tables, screenshots, and interpretation.
- Include citations for public notebooks and any AI assistance.

## Final Recommendation

This package is usable as the main modeling route if the report is written transparently:

- cite public notebooks as inspiration
- explain independent reimplementation
- include ablation experiments
- include failed model directions
- avoid claiming full originality for common Kaggle techniques

The final method does not use hidden test labels, hard-coded answers, external high-score submissions, or leaderboard row overrides.
