# PRCP-1003: Customer Transaction Prediction

A banking machine learning project that predicts which customers will make a transaction in the future, regardless of the amount.

## Problem
- Binary classification: `target = 1` means the customer will transact, `0` means they will not.
- Data: 200,000 customers, 200 anonymized numeric features (`var_0` to `var_199`).
- The target is imbalanced (about 90% class 0, 10% class 1).

## Approach
1. **Data analysis:** no missing values or duplicates, nearly independent bell-shaped features, very weak correlation with the target, rare outliers (kept).
2. **Imbalance handling:** stratified split, balanced class and sample weights, threshold tuning, ROC-AUC as the main metric.
3. **Models compared:** Logistic Regression, Naive Bayes, LDA, Decision Tree, Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost.
4. **Feature engineering:** squared and duplicate-flag features (linear models) and "value occurs only once" features (boosting models).
5. **Tuning:** lower learning rate with early stopping.

## Results (validation set, 40,000 customers)

| Model | ROC-AUC | F1 |
|---|---|---|
| **CatBoost (Tuned + Features)** | **0.9017** | 0.5765 |
| Gradient Boosting (Tuned + Features) | 0.8987 | 0.5741 |
| CatBoost + Features | 0.8931 | 0.5564 |
| Logistic Regression + Features | 0.8863 | 0.5447 |
| Gaussian Naive Bayes | 0.8822 | 0.5394 |

The final model, CatBoost (Tuned + Features), has Recall 0.5749 and Precision 0.5780 at a threshold of about 0.68.

## How to run
1. Download `train.csv` from the dataset link and place it in Google Drive (or the same folder as the notebook).
2. Open the notebook in Google Colab or Jupyter.
3. Update the data path in the "Load the Dataset" cell.
4. Run all cells from top to bottom. The tuning cells take several minutes.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost joblib
```

## Limitations
- The settings and threshold were chosen on the same validation set used to report scores, so performance on new customers may be slightly lower.
- The features are anonymized, so the model cannot be explained in business terms.

## Author
(Your name)
