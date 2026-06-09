# Not All NaNs Are Equal — Ames Housing Deep Dive

## Ames Housing Price Prediction — End-to-End Machine Learning Project

## Project Objective

This project implements a full machine learning pipeline using the Ames Housing dataset.

The goal is to predict house prices from structured tabular data while building a complete and reproducible ML workflow.

Focus areas:
- Exploratory Data Analysis (EDA)
- Missing value interpretation and handling
- Feature engineering
- Categorical encoding
- Model benchmarking
- Hyperparameter optimization
- Model evaluation

---

## Dataset

- Source: Kaggle Ames Housing Dataset  
- Kaggle competition: https://www.kaggle.com/code/iriyablood/not-all-nans-are-equal-ames-housing-deep-dive  
- Target variable: `SalePrice`  
- Problem type: Regression  

---

## Workflow

### 1. Data Exploration

- Distribution analysis of target variable
- Outlier detection
- Correlation analysis
- Identification of skewed distributions

---

### 2. Missing Value Handling

Key assumption:

Missing values often represent **absence of a feature**, not missing data.

Examples:
- No garage → encoded as `HasGarage = 0`
- No pool → encoded as `HasPool = 0`
- No fireplace → encoded as `HasFireplace = 0`

Missingness was treated as a structural signal rather than noise.

---

### 3. Feature Engineering

Created features:

- `HouseAge`
- `RemodAge`
- `GarageAge`
- `HasGarage`
- `HasPool`
- `HasFireplace`
- `TotalBath`
- `Remodeled`

These features were designed to capture structural and temporal properties of houses.

---

### 4. Encoding

- One-Hot Encoding for nominal categorical variables
- Ordinal Encoding for quality-related features:
  - ExterQual
  - KitchenQual
  - BsmtQual
  - GarageQual

Ordinal structure was preserved to improve model consistency.

---

### 5. Modeling

Models evaluated:

- Linear Regression
- Ridge / Lasso / ElasticNet
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- SVR variants

---

### 6. Model Selection

Best performing model:

CatBoost Regressor

---

### 7. Hyperparameter Tuning

Optimized using RandomizedSearchCV

Best parameters:
- depth: 4  
- learning_rate: 0.0378  
- iterations: 674  
- l2_leaf_reg: 7.1  
- border_count: 120  

---

## Results

- Cross-validation RMSE: 23462.27  
- Hold-out test RMSE: 21142.79  
- Kaggle public score: 0.1299  

---

## Key Insights

- Missing values encode structural information in housing datasets
- Feature engineering had greater impact than model complexity
- CatBoost handled categorical structure effectively
- Encoding strategy significantly affects model performance
- Tree-based models outperform linear baselines for nonlinear interactions

---

## Limitations

- No stacking or blending ensembles
- Limited experimentation with log-transformed target space
- No SHAP-based interpretability analysis
- No uncertainty estimation

---

## Future Improvements

- Stacking ensemble (CatBoost + LightGBM + linear models)
- SHAP-based feature importance analysis
- Log-space modeling experiments
- Bayesian hyperparameter optimization
- Residual diagnostics and error analysis

---

## Kaggle Notebook

https://www.kaggle.com/code/iriyablood/not-all-nans-are-equal-ames-housing-deep-dive
