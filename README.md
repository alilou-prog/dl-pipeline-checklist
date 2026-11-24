# End-to-End Deep Learning Pipeline Checklist

## 1. Data Exploration
- [ ] Run EDA (data profiling)
- [ ] Detect potential dataset issues
- [ ] Discuss class imbalance

## 2. Data Preparation *(build as a pipeline)*
- [ ] **Dataset splitting:** stratified `train / val / test` split, ensuring no data leakage (fit imputers/scalers on train only)
- [ ] **Feature engineering:**
  - Feature selection (relevance, real-world availability, potential issues)
  - Feature extraction (PCA)
- [ ] **Data quality:**
  - Missing values (imputation or explicit `"unknown"` class)
  - Outliers, skewness, and potentially problematic values
- [ ] **Categorical encoding:** nominal & ordinal
- [ ] **Feature scaling**

## 3. Baseline Models
- [ ] Majority class classifier
- [ ] Logistic Regression
- [ ] Tree-based ensemble learning

## 4. Deep NN Design *(train a separate model per tweak to isolate its effect)*
- [ ] Architecture & activation functions
- [ ] Optimization methods & learning rates
- [ ] Weight initialization
- [ ] L2 regularization and/or Dropout
- [ ] Early stopping
- [ ] Addressing class imbalance
- [ ] Investigate train vs. validation curves

## 5. Final Evaluation & Business Decisions
- [ ] Evaluate the **best 3 finalized models** once on the held-out **test set** (all relevant metrics + confusion matrix)
- [ ] **Business challenge (predictions -> decisions):**
  - Understand prediction errors (FP vs. FN)
  - Precision vs. Recall trade-off
  - Limited business resources (how the model helps prioritize)
  - Business impact
  - Final recommendation (use, drop, hybrid, ...) based on test evaluation, baseline performance, business constraints, and potential impact
