![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Model-Machine%20Learning-green)
![Climate](https://img.shields.io/badge/Domain-Climate%20Agriculture-orange)
![Interpretability](https://img.shields.io/badge/Focus-Interpretability-purple)

# Climate-Smart Crop Modeling: Predicting Bioactive Compounds under SSP Scenarios

> Predicting and stabilizing bioactive compound levels under climate change scenarios using interpretable machine learning and robust generalization strategies.

A machine learning project that models the impact of climate change on bioactive compounds (TPC, TFC) in *Cnidium officinale*.  
This study integrates environmental variables, physiological indicators, and SSP climate scenarios to build predictive models that support climate-resilient agricultural strategies.

---

## Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `notebooks/` | End-to-end pipeline (preprocessing → modeling → evaluation → prediction) |
| `docs/final_report.pdf` | Final report summarizing methodology and results |
| `docs/interim_presentation.pdf` | Early-stage EDA and hypothesis exploration |
| `data/` | Dataset description only (data not included) |

---

## Analysis Overview

### 1. Problem Definition

Climate change introduces variability in environmental conditions (temperature, humidity, CO₂, VPD),  
leading to instability in bioactive compounds (TPC, TFC) in medicinal crops.

This project aims to:

- Predict compound levels under different SSP scenarios  
- Identify key environmental drivers  
- Provide insights for **stable, high-quality crop production**

---

### 2. Data

| Category | Description |
|----------|-------------|
| Environmental Variables | Temperature, Humidity, VPD, CO₂, PAR, Rainfall |
| Physiological Indicators | Chl_a, Chl_b, TChl, Car, Fv/Fm, PI_abs, etc. |
| Target Variables | Leaf_TPC, Root_TPC, Leaf_TFC, Root_TFC |
| Scenario | SSP 1-2.6, 3-7.0, 5-8.5 |

---

### 3. Methodology

| Step | Description |
|------|-------------|
| Preprocessing | Missing value handling, outlier removal, feature engineering |
| Model Comparison | Ridge, Lasso, ElasticNet, PLS, RandomForest, CatBoost, GAM |
| Validation Strategy | **LOGO (Leave-One-Group-Out)** based on SSP scenarios |
| Model Selection | Based on generalization across unseen scenarios |
| Interpretation | Feature importance, SHAP, and Bayesian analysis |

---

### Key Approach: LOGO Validation

Unlike standard cross-validation, this project uses:

- **Leave-One-Group-Out (LOGO)** based on SSP scenarios  
- Each scenario is completely excluded during training and used for testing  

**This ensures:**
- Robust evaluation under unseen climate conditions  
- Stronger generalization for future environmental changes  

---

## Key Findings

- **Best Model**: CatBoost with LOGO validation  
  - R² ≈ 0.91, lowest MAE and RMSE across models  
- Key drivers of compound variation:
  - CO₂, Temperature, PI_abs, Fv/Fm  
- Environmental variables influence different targets differently:
  - VPD and Humidity have stronger effects on Root_TPC stability  
- The model successfully captures non-linear relationships between climate and plant physiology  
- The framework supports:
  → Climate-resilient crop management  
  → Data-driven agricultural decision-making  

---

## Advanced Analysis

### Bayesian Hierarchical Modeling

- PyMC-based hierarchical model  
- Random slopes per SSP scenario  
- Provides:
  - Effect direction  
  - Uncertainty (95% CI)  
  - Scenario-specific sensitivity  

->  Unlike black-box models:
- Enables **interpretable and probabilistic insights**

---

## Limitations & Future Work

- External variables (e.g., soil conditions, management practices) not included  
- Structural shifts under extreme climate scenarios may not be fully captured  
- Future extensions:
  - Multivariate models with exogenous variables  
  - Integration with real-time climate data  
  - Expansion to other crops and agricultural systems  

---

## Data Availability

The dataset used in this project was provided for internal competition purposes and cannot be publicly shared.

All code is provided for reproducibility.

---

## Environment

- **Language**: Python 3.10  
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, catboost, statsmodels, shap  
- **Frameworks**: PyMC (Bayesian modeling)  

---

## Summary

This project goes beyond standard predictive modeling by:

- Focusing on **generalization under climate scenarios**  
- Combining **performance and interpretability**  
- Providing **actionable insights for climate-smart agriculture**
  -> A step toward data-driven, sustainable crop management.
