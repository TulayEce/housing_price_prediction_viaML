# Housing Price Prediction - King County, WA 🏠

**Course:** Machine and Deep Learning  
**Student:** Tulay Ece Yildirim  
**Academic Year:** Fall 2025/2026

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Reproducibility Notes](#reproducibility-notes)

---

## Overview

This project develops a robust machine learning framework for predicting residential property prices in King County, Washington. The analysis follows a comprehensive data science workflow including exploratory data analysis, feature engineering with target encoding for ZIP codes, outlier removal using IQR-based methods, and model comparison across multiple regression algorithms. Four distinct models—**Ridge Regression, Decision Trees, Random Forest, and Gradient Boosting**—are trained and optimized using **GridSearchCV** to determine which architecture best captures the non-linear patterns of the housing market.

---

## Dataset

- **Source:** `kc_house_data.csv` (King County housing sales, 2014–2015)
- **Size:** 21,613 observations with 21 variables
- **Target Variable:** `price` (sale price in USD)
- **Key Feature Groups:**
  - Physical attributes: square footage (living, lot, above, basement), bedrooms, bathrooms, floors
  - Quality indicators: grade (construction quality), condition (maintenance state)
  - Location: zipcode, latitude, longitude
  - Special features: waterfront, view
  - Temporal: year built, year renovated, sale date

---

## Methodology

- **Data Cleaning:** Removal of missing values, invalid entries (bedrooms = 33, bathrooms = 0), and duplicate records
- **Feature Engineering:** Derived features (house age, sale month/year, square meters conversion, renovation indicator), target encoding for ZIP codes
- **Outlier Handling:** IQR-based removal for numerical features (bedrooms, bathrooms, sqft_living, grade, price)
- **Preprocessing Pipeline:** ColumnTransformer with OneHotEncoder (categorical) and StandardScaler (numerical)
- **Models Trained:** Ridge Regression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
- **Hyperparameter Tuning:** GridSearchCV with 3-fold cross-validation
- **Evaluation Metrics:** RMSE, MAE, R²

---

## Results

**Test Set Performance (sorted by RMSE):**

| Model              | RMSE        | MAE        | R²     |
|--------------------|-------------|------------|--------|
| **Gradient Boosting** | **72,089.43** | **49,902.94** | **0.8721** |
| Random Forest      | 76,249.10   | 52,051.73  | 0.8569 |
| Ridge Regression   | 83,430.08   | 60,990.56  | 0.8287 |
| Decision Tree      | 92,651.93   | 65,826.42  | 0.7888 |

**Best Model:** Gradient Boosting Regressor achieved the highest R² (0.872) and lowest error metrics, demonstrating superior capability in capturing complex, non-linear relationships in the housing data.

---

## Project Structure

```
housing_price_prediction_viaML/
│
├── data/
│   └── kc_house_data.csv          # King County housing dataset
│
├── housing_price_prediction.ipynb  # Main Jupyter notebook
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd housing_price_prediction_viaML
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open and run `housing_price_prediction.ipynb`**

---

## Reproducibility Notes

- All models use `random_state=42` for consistent train/test splits and model initialization
- GridSearchCV uses 3-fold cross-validation (`cv=3`) for hyperparameter optimization
- Outlier removal and feature engineering steps are deterministic and reproducible
- The notebook includes an interactive prediction widget for exploring model predictions with custom inputs

---

**Project Repository:** https://github.com/TulayEce/housing_price_prediction_viaML.git 

