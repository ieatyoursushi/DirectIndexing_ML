# ML Mathematical Derivations (Simplified Version)

## Purpose

This document explains the ML pipeline from first principles without measure theory or heavy notation.

The overall flow is:

```text
Portfolio State
      ↓
 Lot Features (X)
      ↓
 Oracle Rule
      ↓
 Soft Labels
      ↓
 ML Models
      ↓
 Predicted Harvest Probability
```

---

# 1. Feature Space (X)

Each row in the dataset represents:

> One lot, on one day.

The feature vector contains 15 numeric features.

## Lot Features

These describe the lot itself:

| Feature | Meaning |
|----------|----------|
| Unrealized Return | Current gain/loss % |
| Holding Days | Days held |
| Long-Term Flag | Held > 365 days |
| Cost Basis | Purchase price |
| Portfolio Weight | Position size |
| Lot Count | Number of open lots of same stock |

## Portfolio Features

These describe the portfolio environment:

| Feature | Meaning |
|----------|----------|
| YTD Gains | Realized gains this year |
| Tracking Error | Portfolio deviation from benchmark |
| Wash Sale Days | Days since last harvest |

## Asset Features

These describe current market behavior:

| Feature | Meaning |
|----------|----------|
| Daily Return | Today's return |
| Intraday Range | High-Low volatility |
| MA50 Deviation | Distance from 50-day MA |
| MA200 Deviation | Distance from 200-day MA |

## Derived Features

| Feature | Meaning |
|----------|----------|
| Tax Alpha | Estimated tax benefit |
| Days To Year End | Calendar timing feature |

So mathematically:

```text
X = [15-dimensional feature vector]
```

---

# 2. Oracle Label

The oracle is the hand-written harvesting rule.

A lot is harvestable if:

1. Loss exceeds 2%
2. Tracking error is below 5%
3. Portfolio has realized gains
4. Wash-sale window has cleared

Written as:

```text
Harvest = 1 if ALL conditions are true
Harvest = 0 otherwise
```

This produces the hard label:

```text
Y ∈ {0,1}
```

---

# 3. Soft Backtest Label

The oracle only answers:

```text
Harvest now?
```

The soft label asks:

```text
How often will this lot be harvestable
during the next 30 trading days?
```

Procedure:

1. Look forward 30 days.
2. Apply oracle each day.
3. Count harvestable days.
4. Divide by 30.

Formula:

```text
soft_bt = (# oracle fires) / 30
```

Examples:

| Oracle Fires | soft_bt |
|--------------|---------|
| 0 days | 0.00 |
| 6 days | 0.20 |
| 15 days | 0.50 |
| 30 days | 1.00 |

Interpretation:

- 0 → almost never harvestable
- 1 → harvestable nearly all the time

Think of it as a harvest urgency score.

---

# 4. GBM Soft Label

Instead of using the real future path:

```text
Actual Prices
```

simulate many possible futures:

```text
GBM Path 1
GBM Path 2
...
GBM Path 200
```

For each path:

- Run the oracle
- Check whether it fires

The label becomes:

```text
Probability Oracle Fires
```

Estimated as:

```text
soft_gbm =
(firing paths)/(total paths)
```

This is a Monte Carlo probability estimate.

---

# 5. Data Preparation

Every supervised model uses the same pipeline.

## Step 1

Train/Test Split

```text
80% Train
20% Test
```

## Step 2

Median Imputation

Replace missing values with training-set medians.

## Step 3

Class Balancing

Harvest events are rare.

Increase their weight during training.

## Step 4

5-Fold Cross Validation

Choose best hyperparameters.

## Step 5

Refit Best Model

Train on all training data.

## Step 6

Evaluate Once

Use the test set exactly once.

---

# 6. Logistic Regression

Model:

```text
Score = w·x + b
```

Convert score into probability:

```text
p = 1/(1 + e^(-Score))
```

Interpretation:

- Large positive score → harvest likely
- Large negative score → harvest unlikely

Advantages:

- Fast
- Interpretable
- Well calibrated

---

# 7. Elastic Net Logistic Regression

Same logistic model:

```text
p = sigmoid(w·x + b)
```

But adds penalties:

```text
L1 penalty
```

encourages sparse models.

```text
L2 penalty
```

shrinks coefficients.

Result:

- Less overfitting
- Automatic feature selection

---

# 8. Gradient Boosted Trees

Idea:

Build many small trees sequentially.

Tree 1:
fixes some errors

Tree 2:
fixes remaining errors

Tree 3:
fixes remaining errors

...

Final prediction:

```text
Prediction =
Tree1 +
Tree2 +
...
+ TreeM
```

Usually strongest supervised baseline.

---

# 9. Random Forest

Instead of correcting mistakes:

build many independent trees.

```text
Tree 1
Tree 2
Tree 3
...
Tree T
```

Final prediction:

```text
Average of all trees
```

Benefits:

- Low variance
- Handles nonlinear relationships
- Harder to overfit

---

# 10. Linear Regression (Negative Control)

Model:

```text
ŷ = w·x + b
```

Problem:

Outputs can be:

```text
-0.4
1.7
2.3
```

These are not valid probabilities.

Logistic regression fixes this by forcing:

```text
0 ≤ p ≤ 1
```

Therefore linear regression is intentionally included as a poor-fit benchmark.

---

# 11. Evaluation Metrics

## ROC-AUC

Measures ranking quality.

Question:

> Does the model rank positives above negatives?

Higher is better.

---

## PR-AUC

Measures:

```text
Precision vs Recall
```

Important because harvest events are rare.

This is the primary model-selection metric.

---

## F1 Score

Balances:

```text
Precision
Recall
```

Formula:

```text
F1 =
2PR/(P+R)
```

---

# 12. PCA

Goal:

Reduce dimensionality.

Starting space:

```text
15 features
```

Find directions explaining the most variance.

Keep enough components to explain:

```text
95% variance
```

Result:

```text
15 dimensions
      ↓
r dimensions
```

where r is much smaller.

---

# 13. K-Means

Goal:

Group similar stocks together.

Algorithm:

1. Choose k centers
2. Assign stocks to nearest center
3. Recompute centers
4. Repeat until stable

Produces:

```text
Cluster 1
Cluster 2
...
Cluster k
```

Used for discovery rather than prediction.

---

# 14. Big Picture

The entire project can be summarized as:

```text
Oracle Rules
      ↓
Forward-Looking Soft Labels
      ↓
Machine Learning Models
      ↓
Harvest Probability Estimates
```

or mathematically:

```text
Oracle
   ↓
soft_bt
   ↓
Learned Posterior
```

The ML model's job is not to replace the oracle.

Its job is to rank lots inside the oracle region and estimate harvest urgency.
