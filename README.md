# Used Car Price Prediction

End-to-end machine learning project for predicting used car prices using historical transaction data.  
The project compares an interpretable linear baseline (Ridge regression) with a more flexible tree-based model (Random Forest).

---

## Business Objective
Accurate pricing of used vehicles is a key challenge for marketplaces, dealerships, and auction platforms.  
The goal of this project is to build a regression model that can estimate the selling price of a used car based on its characteristics, supporting pricing decisions and market analysis.

---

## Project Scope
The project focuses on:
- understanding the structure and quality of real-world tabular data,
- identifying key price drivers through exploratory analysis,
- building and evaluating predictive models with proper validation,
- comparing model performance against interpretability trade-offs.

---

## Data Overview
The dataset contains historical used car transactions with features such as:
- vehicle characteristics (make, body type, color, condition),
- usage metrics (odometer, vehicle age, miles per year),
- market-related information (MMR).

Raw data are excluded from the repository, while processed, model-ready datasets are included to ensure reproducibility.

---

## Methodology

### Exploratory Data Analysis (EDA)
- Analysis of numerical and categorical feature distributions
- Handling missing values and outliers
- Identification of highly skewed and high-cardinality categorical variables

### Feature Engineering & Preprocessing
- Creation of derived features (e.g. vehicle age, miles per year)
- Model-specific preprocessing pipelines:
  - One-Hot Encoding and scaling for linear models
  - Ordinal encoding for tree-based models
- All preprocessing steps are fitted exclusively on the training set to prevent data leakage

### Modeling
Two models were trained and compared:
- **Ridge Regression** – interpretable linear baseline
- **Random Forest Regressor** – non-linear model capturing complex interactions

Hyperparameters were optimized using cross-validation.

---

## Evaluation
Models were evaluated on a held-out test set using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Test Set Performance
- Ridge Regression: MAE ≈ 1,213 | RMSE ≈ 2,685  
- Random Forest (tuned): MAE ≈ 992 | RMSE ≈ 1,993  

The Random Forest model achieved substantially better predictive performance, at the cost of reduced interpretability.

---

## Feature Importance
Feature importance analysis from the Random Forest model indicates that market valuation (MMR), vehicle condition, usage intensity, and manufacturer are the strongest price drivers.

---

## Repository Structure
- `eda.ipynb` – exploratory data analysis and data preparation
- `models.ipynb` – preprocessing pipelines, model training, tuning, and evaluation
- `used_cars_model_ready_ridge.csv` – processed dataset for linear models
- `used_cars_model_ready_rf.csv` – processed dataset for tree-based models

---

## Key Takeaways
- Proper preprocessing and leakage prevention are critical for realistic model evaluation
- Tree-based models outperform linear baselines on complex tabular data
- Interpretability–performance trade-offs should be considered depending on business needs

---

## Tools & Technologies
- Python
- pandas, NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook
