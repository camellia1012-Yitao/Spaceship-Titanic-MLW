# High Score Notebook Case Study

更新时间：2026-05-14

用户提供了三个 public score 约 `0.814-0.815+` 的 notebook：

- `space-titanic-eda-advanced-feature-engineering (1).ipynb`
- `0-814-optuna-xgb-space-titanic.ipynb`
- `spaceshiptitanic-xgboost-score-81-5.ipynb`

## 合规性检查

### Advanced EDA / Feature Engineering

可借鉴：

- `expenditure = VRDeck + Spa + RoomService`
- Last name / group / cabin features
- Target/count/WOE encoding 思路
- 多模型权重优化思路

不能作为主方法照用：

- notebook 末尾读取多个外部高分 submission：
  - `XGB_best.csv`
  - `solution/submission.csv`
  - `0-81669.../submission.csv`
- 这种做法不适合课程报告主方法。

### 0.814 Optuna XGB / 81.5 XGBoost

核心合法思路：

- 很窄的特征集，而不是复杂大特征：
  - spending columns
  - `Expenses`
  - `Age`
  - `CryoSleep`
  - `Cabin` deck/side
  - `VIP`
  - `HomePlanet`
  - `Destination`
- 通过 `PassengerId[:4]` 的 Room/Group 填补：
  - `VIP`
  - `Cabin`
  - `HomePlanet`
  - `Destination`
- 固定 drop list：
  - `ShoppingMall`
  - `Age`
  - `CryoSleep_True`
  - `HomePlanet_Earth`
  - `HomePlanet_Europa`
  - `VIP_True`
  - `HomePlanet_Mars`
  - `Destination_PSO J318.5-22`
  - `VIP_False`
  - `Destination_55 Cancri e`
  - `FoodCourt`
  - `Destination_TRAPPIST-1e`
- Optuna/XGB 参数：
  - `reg_lambda=3.061`
  - `reg_alpha=4.582`
  - `colsample_bytree=0.924`
  - `subsample=0.953`
  - `learning_rate=0.0667`
  - `n_estimators=730`
  - `max_depth=5`
  - `min_child_weight=1`
- `KMeansSMOTE` oversampling 是 0.814 notebook 和我们旧复刻的一个关键差异。

## Clean Reimplementation

新增脚本：

- `xgb_optuna_style_kmeans.py`

这个脚本用自己的结构重新实现上述合法路线，没有复制外部 submission，也没有 hard-coded label。

本地 CV：

```text
seed=42   none    0.80824
seed=42   smote   0.80893
seed=42   kmeans  0.80931
seed=123  kmeans  0.80909
seed=2023 kmeans  0.80989
seed=2140 kmeans  0.81114
```

## Candidate Submissions

这条 XGB/KMeansSMOTE 路线与当前 `0.80757` LGBM 文件差异很大，约 `480-500` 行。因此它不是稳健微调，而是一条独立高风险候选路线。

如果要测试，建议最多测试：

1. `submissions/xgb_optuna_style_kmeans/submission_xgb_optuna_seed2140_kmeans_thr500.csv`
2. `submissions/xgb_optuna_style_kmeans/submission_xgb_optuna_seed2140_kmeans_thr510.csv`
3. `submissions/xgb_optuna_style_kmeans/submission_xgb_optuna_seed2023_kmeans_thr500.csv`

原因：

- `seed2140_kmeans` 本地 CV 最高。
- notebook 原始提交方式等价于 `predict` / `threshold=0.5`，不是强行压低 True 比例。

## Current Judgment

这几个 high-score notebook 给出的新启发是：

- public `0.81+` 的合法路线可能不是更复杂，而是“极简 XGB + drop list + KMeansSMOTE”。
- 但这条路线和当前最高 LGBM 结果差异很大，风险也大。
- 它值得作为独立候选测试 1-2 次；如果 public 不升，就停止，不要继续围绕它微调。
