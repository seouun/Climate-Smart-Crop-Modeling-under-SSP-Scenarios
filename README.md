![Python](https://img.shields.io/badge/Python-3.10-blue)
![Competition](https://img.shields.io/badge/Competition-KISTI%20DATA·AI%202025-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Climate%20×%20Agriculture-orange)
![Interpretability](https://img.shields.io/badge/Approach-Interpretable%20ML-purple)
 
# Climate-Smart Crop Modeling: Predicting Bioactive Compounds in *Cnidium officinale* under SSP Climate Scenarios
 
**2025 KISTI DATA·AI Analysis Competition**
 
A machine learning study on how climate change scenarios (SSP1-2.6 / SSP3-7.0 / SSP5-8.5) alter the bioactive compound content (Total Phenolic Content, Total Flavonoid Content) of medicinal crop *Cnidium officinale* (천궁), with a focus on **scenario-level generalization** and **interpretable causal analysis**.
 
---
 
## Research Question
 
> *Can a machine learning model trained on two climate scenarios reliably predict bioactive compound levels in a completely unseen third scenario — and what environmental factors drive these changes?*
 
Standard crop models optimize for within-sample accuracy. This project is designed around a harder, more realistic objective: **scenario extrapolation**. Models are evaluated using **Leave-One-Group-Out (LOGO) cross-validation** at the SSP scenario level, simulating the challenge of predicting under genuinely novel future climate conditions.
 
---
 
## Background
 
*Cnidium officinale* is a high-value medicinal crop used in traditional medicine, functional foods, and pharmaceutical ingredients. Its bioactive quality is sensitive to environmental stress — yet most agricultural AI focuses on yield, not compound stability.
 
| Scenario | Description | CO₂ by 2100 |
|---|---|---|
| SSP1-2.6 | Sustainable growth, low emissions | 432 ppm |
| SSP3-7.0 | Fragmented policy, high vulnerability | 834 ppm |
| SSP5-8.5 | Fossil-fuel intensive, urban-led development | 1,089 ppm |
 
This framing connects directly to climate impact assessment under IPCC's Shared Socioeconomic Pathways — the same framework used in integrated assessment modeling (IAM) research.
 
---
 
## Repository Structure
 
```
├── notebooks/
│   ├── 01_eda_correlation.ipynb        # EDA: distribution, correlation, PCA
│   ├── 02_baseline_regression.ipynb    # Linear, Ridge, Lasso, ElasticNet, PLS
│   ├── 03_baseline_tree.ipynb          # RandomForest, XGBoost + tuning
│   ├── 04_final_model_blend.ipynb      # Blending + linear calibration + SHAP
│   └── 05_bayesian_hierarchical.ipynb  # PyMC Bayesian hierarchical model
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_train.py
│   ├── run_tune_catboost_LOGO.py
│   └── run_bayes_analysis.py
├── docs/
│   ├── interim_presentation.pdf        # Mid-project EDA & baseline results
│   └── final_report.pdf
└── data/
    └── README.md                       # Data description (raw data not included)
```
 
---
 
## Data
 
Data collected from **SPDS (Soil Plant Daylit System) chambers** at three SSP-simulated climate conditions.
 
| Category | Variables |
|---|---|
| Environmental | Temperature, Humidity, VPD, CO₂ppm, PAR, Rainfall |
| Physiological (chlorophyll/pigment) | Chl_a, Chl_b, TChl, Car, Chl_a_b, TCh-Car |
| Physiological (photosynthesis efficiency) | Fv/Fm, PI_abs, SFI_abs, DF_abs |
| Photosystem reaction center | ABS-RC, Tro-RC, Dio-RC, Eto-RC |
| Extraction yield | Leaf_ExtractionYield, Root_ExtractionYield |
| **Target variables** | **Leaf_TPC, Root_TPC, Leaf_TFC, Root_TFC** |
 
> Raw data provided by the competition organizers and cannot be publicly shared. All code is available for reproducibility.
 
---
 
## Methodology
 
### Pipeline
 
```
Raw Data
   │
   ├─ EDA
   │    ├─ Scenario-stratified distribution analysis (ANOVA, Kruskal-Wallis)
   │    ├─ Correlation heatmaps per SSP scenario (SSP1 / SSP3 / SSP5)
   │    └─ PCA: SSP1 ↔ SSP5 separation on photosynthesis + seasonality axis (PC1: 53%)
   │
   ├─ Preprocessing
   │    ├─ No missing values; outlier clipping via IQR (Chl_a_b selective)
   │    ├─ VIF-based multicollinearity reduction (target VIF < 10; 8 variables removed)
   │    ├─ RobustScaler (robust to extreme values under SSP5 stress conditions)
   │    └─ Month OneHot Encoding (captures seasonality as a quantitative signal)
   │
   ├─ Baseline Modeling
   │    ├─ Regression: Linear, Ridge, Lasso, ElasticNet, PLS2/PLS3
   │    └─ Tree: RandomForest, XGBoost (RandomizedSearch → fine-grained tuning)
   │
   ├─ Final Model
   │    ├─ Blend: XGB(0.55) + Ridge(0.45) — OOF weight optimization
   │    ├─ Linear Calibration — corrects scale/offset mismatch per scenario × target
   │    └─ Interpretation: SHAP (beeswarm) + Permutation Importance (LOGO-based)
   │
   └─ Bayesian Hierarchical Model (PyMC)
        ├─ Random slopes for Temp, Humid, CO₂ per SSP scenario
        └─ Reports effect direction + 95% CI per scenario (WAIC/LOO for fit assessment)
```
 
### Validation Strategy: Scenario-Level LOGO
 
Standard K-Fold cross-validation does not simulate a realistic climate adaptation problem — it leaks scenario-level information into training. This project uses **Leave-One-Group-Out (LOGO)** where each SSP scenario is entirely withheld as a test set in turn.
 
This tests the harder question: *can the model extrapolate to a climate it has never seen?*
 
---
 
## Key Findings
 
### 1. SSP-Dependent Compound Dynamics
 
EDA revealed statistically significant differences across scenarios (Leaf_TPC: ANOVA p<0.001; Root_TFC: Kruskal-Wallis p<0.001). The pattern is consistent:
 
> **As climate intensifies (SSP1 → SSP5), above-ground (leaf) compound accumulation becomes unstable, while below-ground (root) compounds increasingly dominate under environmental stress.**
 
| Scenario | Pattern |
|---|---|
| SSP1 (mild) | Photosynthetic efficiency drives compound accumulation — standard physiology |
| SSP3 (intermediate) | Leaf-compound correlation weakens; root retains photosynthetic linkage |
| SSP5 (high stress) | Leaf TPC loses predictability; Root TPC/TFC governed directly by VPD and Temperature |
 
This shift has direct implications for **harvest strategy under future climate conditions**.
 
### 2. Model Performance (LOGO Cross-Validation)
 
Final model: Blend(XGB 0.55 + Ridge 0.45) with linear calibration
 
| Scenario | MAE | RMSE | R² |
|---|---|---|---|
| SSP1 | 0.386 | 0.483 | 0.826 |
| SSP3 | 0.474 | 0.646 | 0.613 |
| SSP5 | 0.380 | 0.548 | 0.797 |
| **Average** | **0.413** | **0.559** | **0.745** |
 
In-sample R² exceeds 0.99 for most targets. The gap to LOGO performance reflects the genuine difficulty of **scenario extrapolation** — predicting across a qualitatively different climate regime the model has never seen. SSP3's lower R² (0.613) is consistent with its high internal variability (bimodal distributions, wide IQR).
 
### 3. Key Drivers (Feature Importance + SHAP)
 
| Target | Primary Drivers |
|---|---|
| Leaf_TPC | Leaf_ExtractionYield, TCh-Car, Dio-RC |
| Root_TPC | Temperature, CO₂ppm, Leaf_ExtractionYield |
| Leaf_TFC | Leaf_ExtractionYield, Temperature |
| Root_TFC | Leaf_ExtractionYield, Temp, Scenario offset (SSP3/SSP5) |
 
With `month` excluded, **physiological stress indicators (Dio-RC, Eto-RC, PI_abs) and thermal variables emerge as stable core predictors**, offering more mechanistic interpretability than seasonality-anchored models.
 
### 4. Bayesian Hierarchical Analysis
 
PyMC-based model estimated SSP-specific random slopes for Temperature, Humidity, and CO₂. Key finding: **the effect direction of Temperature on compound content reverses between SSP1 and SSP5**, consistent with a threshold-based physiological stress response. All chains converged (target_accept=0.97, zero divergences).
 
---
 
## Limitations
 
- Small experimental dataset from controlled chamber conditions — field generalizability is untested
- **SSP3 remains the hardest scenario to generalize** (LOGO R² ≈ 0.61): its transitional stress regime is underrepresented relative to SSP1 and SSP5
- Leaf_TPC shows near-zero Pearson r in some LOGO folds — the model tracks directional trends but struggles with absolute magnitude in low-variance targets
- Soil properties, irrigation management, and genotype variation are not captured
 
---
 
## Implications for Climate-Resilient Agriculture
 
1. **Harvest strategy**: Under SSP3/SSP5, root-based extraction yields more stable compound levels than leaf
2. **Early warning indicators**: PI_abs and Fv/Fm are reliable leading signals of compound quality decline under thermal stress
3. **Extensibility**: The LOGO validation framework and SSP-stratified analysis pipeline generalizes to other medicinal crops under climate pressure
 
---
 
## Environment
 
| Item | Spec |
|---|---|
| Language | Python 3.10 (Anaconda 3) |
| Core Libraries | pandas 2.1.4, numpy 1.26.0, scikit-learn 1.4.2, catboost 1.2.5 |
| Visualization | matplotlib 3.8.2, seaborn 0.13.1 |
| Interpretability | shap 0.45.0 |
| Bayesian Modeling | statsmodels 0.14.1 (PyMC) |
| Hardware | Intel i7-14700, 32GB RAM, Windows 11 Pro |
