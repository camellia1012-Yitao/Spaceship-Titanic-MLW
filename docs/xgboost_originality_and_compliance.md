# Originality And Compliance Notes

## Summary

This package uses public Kaggle notebooks as references for common feature-engineering ideas, but the submitted code is an independent reimplementation and experiment pipeline.

It does not use:

- external high-scoring submissions as labels
- hard-coded answer tables
- leaderboard-derived row overrides
- OR-gating multiple public submissions
- copied notebook cells as the final project notebook

## What Was Inspired By Public Work

The following ideas are common in public Spaceship Titanic notebooks and were used as references:

- using `PassengerId` group prefix
- deriving total passenger expenses
- using cryosleep/spending consistency
- splitting `Cabin`
- using XGBoost with tuned hyperparameters
- dropping low-importance one-hot features
- trying KMeansSMOTE

## What We Added

The project work here includes:

- a standalone Python script with reusable functions
- comparison across seeds
- comparison of no resampling, SMOTE, and KMeansSMOTE
- generation of CV score tables
- generation of multiple controlled threshold candidates
- comparison against previous LightGBM, CatBoost, ensemble, native categorical, and OOF blend experiments
- documentation of failed directions and ablations

## Suggested Report Wording

Use wording like this in the report:

```text
We reviewed public Kaggle notebooks to identify common feature-engineering strategies for the Spaceship Titanic dataset. We then independently reimplemented the preprocessing and model training pipeline, compared several model families and resampling methods, and selected a compact XGBoost + KMeansSMOTE model based on our experiments.
```

Avoid claiming that every idea was invented from scratch. The safer and more honest framing is independent implementation plus experimental validation.

## Why This Is Suitable For The Course Project

The final submission is generated from:

- the official train/test files
- deterministic feature engineering
- cross-validation and ablation experiments
- a trained XGBoost model

No test labels or answer-table style information are used.

