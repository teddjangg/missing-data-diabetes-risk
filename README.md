# Missing Data Imputation and Diabetes Risk Prediction

A comparative study examining how different missing data imputation strategies affect variable distributions and downstream classification performance using a diabetes dataset.

---

## Overview

Real-world clinical datasets often contain substantial missing values. The choice of imputation method can distort variable distributions, alter predictor relationships, and potentially affect model performance. This project systematically compares **four imputation methods** — from simple baselines to advanced multivariate approaches — and evaluates their impact on logistic regression-based diabetes prediction.

---

## Project Structure

```
├── MissingData_Imputation__and_Diabetes_Risk_Prediction.ipynb
├── Diabetes.csv
└── README.md
```

---

## Methodology

### 1. Missing Data Exploration
- **Summary statistics**: missingness counts and percentages per variable
- **UpSet plot**: visualizes combination patterns of co-missing variables
- **Missingness matrix & heatmap** (`missingno`): reveals strong co-missingness between `Insulin` and `SkinThickness` (~0.7 correlation), suggesting a structured MAR mechanism rather than MCAR
- **KDE shadow analysis**: distribution of observed variables split by `Insulin` missingness status — repeated distributional shifts across `Glucose`, `BMI`, `SkinThickness`, and `BloodPressure` support a MAR assumption

### 2. Basic Imputation Methods
| Method | Strategy |
|---|---|
| Mean | Replace with column mean |
| Median | Replace with column median |
| Mode | Replace with most frequent value |
| Constant | Replace with 0 |

KDE plots confirm that mean/median imputation produces a sharper peak around the central tendency, indicating **variance shrinkage**.

### 3. Advanced Imputation Methods
| Method | Strategy |
|---|---|
| **KNN** (`n_neighbors=5`) | Imputes using values from the k most similar observations — preserves local structure |
| **MICE** (`IterativeImputer`, `max_iter=10`) | Iteratively models each missing variable as a function of all others — preserves global multivariate relationships |

Distribution comparison (KDE on `SkinThickness`) shows KNN and MICE produce distributions much closer to the complete-case pattern than mean imputation.

### 4. Model Evaluation
- **Logistic regression** trained on each imputed dataset (80/20 train-test split, `StandardScaler`)
- Metrics: **Accuracy** and **AUC**
- **Adjusted R²** comparison (OLS on `Glucose`) to assess preservation of predictor relationships
- **VIF analysis** to rule out multicollinearity as a confound
- **Coefficient stability analysis**: logistic regression coefficients compared across imputation methods

---

## Key Findings

**Distribution quality**: Mean imputation noticeably distorts variable distributions; KNN and MICE preserve the original structure far better. MICE achieves the highest adjusted R² in the OLS comparison, indicating stronger preservation of multivariate relationships.

**Prediction performance**: Despite clear distributional differences, logistic regression Accuracy and AUC remain nearly identical across all four imputation methods.

**Why?** VIF analysis rules out multicollinearity. Coefficient stability analysis shows that strong predictors — `Glucose`, `BMI`, `Age`, `Pregnancies` — are estimated almost identically regardless of imputation method. The variables with the heaviest missingness (`Insulin`, `SkinThickness`) appear to be weak predictors of the diabetes outcome, making downstream classification insensitive to how they are imputed.

**Main takeaway**: When heavily missing variables are weak predictors of the target, simple mean imputation may be sufficient for prediction tasks. However, KNN and MICE are clearly preferable when the goal is to preserve the original data distribution or conduct analysis that depends on accurate variable relationships.

---

## Limitations

- Results are based on a single dataset; findings may not generalize to settings where high-missingness variables are stronger predictors of the outcome.
- The MAR mechanism was inferred through exploratory visualization and cannot be formally proven.

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
missingno
upsetplot
scikit-learn
statsmodels
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn missingno upsetplot scikit-learn statsmodels
```

---

## Usage

```bash
git clone https://github.com/teddjangg/<repo-name>.git
cd <missing-data-diabetes-risk>
jupyter notebook MissingData_Imputation__and_Diabetes_Risk_Prediction.ipynb
```

Make sure `Diabetes.csv` is in the same directory as the notebook before running.
