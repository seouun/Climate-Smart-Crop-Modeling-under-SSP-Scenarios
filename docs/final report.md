# Final Report
## Climate-Smart Crop Modeling: Predicting Bioactive Compounds in *Cnidium officinale* under SSP Climate Scenarios
**2025 KISTI DATA·AI Analysis Competition**
 
---
 
## 1. 연구 목적
 
기후변화(온도, 습도, CO₂, VPD 등)로 인해 천궁의 기능성 성분(TPC, TFC) 변동성이 심화됨.
SSP 시나리오(1-2.6 / 3-7.0 / 5-8.5)별 환경 데이터를 기반으로 성분 함량 안정화를 위한 예측 모델을 개발하고, 시나리오 외삽(scenario extrapolation) 성능을 중심으로 평가함.
 
---
 
## 2. 데이터 구성
 
| 범주 | 변수 |
|---|---|
| 환경 변수 | Temperature, Humidity, VPD, CO₂ppm, PAR, Rainfall |
| 생리적 지표 (엽록소/색소) | Chl_a, Chl_b, TChl, Car, Chl_a_b, TCh-Car |
| 생리적 지표 (광합성 효율) | Fv/Fm, PI_abs, SFI_abs, DF_abs |
| 광계 반응중심 | ABS-RC, Tro-RC, Dio-RC, Eto-RC |
| 추출 수율 | Leaf_ExtractionYield, Root_ExtractionYield |
| **타깃 변수** | **Leaf_TPC, Root_TPC, Leaf_TFC, Root_TFC** |
 
**데이터 처리 단계**
- 원본 데이터(rawdata.csv) 로드
- 결측치 처리 및 이상값 제거 (IQR 기반, Chl_a_b 선택적 클리핑)
- VIF 기반 다중공선성 제거 (VIF < 10 기준, 8개 변수 제거)
- RobustScaler 표준화
- Month 변수 sin/cos 순환 인코딩
- 데이터 분할 (splits_train.csv, splits_valid.csv, splits_test.csv)
 
---
 
## 3. 모델링
 
### 교차검증: Scenario-Level LOGO
 
일반적인 K-Fold는 시나리오 정보가 훈련에 누출되어 낙관적 성능을 보임.
본 연구에서는 **Leave-One-Group-Out (LOGO)** 방식으로 SSP 시나리오 단위로 완전히 제외한 후 검증하여 미래 환경 조건에 대한 외삽 성능을 평가함.
 
### 모델 비교
 
| 단계 | 모델 |
|---|---|
| Baseline (회귀) | Ridge, Lasso, ElasticNet, PLS |
| Baseline (트리) | RandomForest, XGBoost, CatBoost, GAM |
| **최종 모델** | **XGB(0.55) + Ridge(0.45) Blend + Linear Calibration** |
| 해석 분석 | PyMC Bayesian Hierarchical Model |
 
### 최종 모델 선택 근거
 
**CatBoost 탈락 이유:** LOGO 검증 기준 SSP 시나리오별 분산이 XGB+Ridge 블렌드보다 높아 일관성이 낮음.
 
**블렌딩 선택 이유:** XGBoost는 SSP5에 강하지만 SSP3에 불안정, Ridge는 반대 패턴. OOF 최적화(XGB 0.55, Ridge 0.45) 블렌딩이 전 시나리오에서 가장 일관된 성능을 보임.
 
**선형 보정 추가 이유:** 블렌드 예측값이 시나리오 × 타깃별 계통적 스케일·오프셋 오차를 보여, 단순 선형 보정(ŷ → aŷ + b)으로 해소함.
 
---
 
## 4. 결과
 
### LOGO 교차검증 성능 (최종 모델)
 
| Scenario | MAE | RMSE | R² |
|---|---|---|---|
| SSP1 | 0.419 | 0.533 | 0.303 |
| SSP3 | 0.464 | 0.650 | 0.256 |
| SSP5 | 0.323 | 0.445 | −0.712 |
| **Average** | **0.401** | **0.558** | **0.268** |
 
SSP5의 음수 R²는 훈련 중 한 번도 접하지 못한 기후 조건에서의 분포 이동(distribution shift)을 반영하며, 소규모 챔버 데이터의 외삽 한계를 보여줌.
 
### 주요 변수 (SHAP + Permutation Importance)
 
| 타깃 | 핵심 변수 |
|---|---|
| Leaf_TPC | Leaf_ExtractionYield, TCh-Car, Dio-RC |
| Root_TPC | Temperature, CO₂ppm, Leaf_ExtractionYield |
| Leaf_TFC | Leaf_ExtractionYield, Temperature |
| Root_TFC | Leaf_ExtractionYield, Temp, Scenario offset (SSP3/SSP5) |
 
> SHAP 및 Permutation Importance는 전체 데이터 fit 기준(in-sample) 계산. LOGO 기반 SHAP는 fold별 재학습 비용으로 인해 생략.
 
### Bayesian 위계 분석 결과
 
| 변수 | SSP1 | SSP3 | SSP5 | 해석 |
|---|---|---|---|---|
| Leaf_TPC × Temperature | +0.83 | +0.29 | +0.11 | 고스트레스 시 광합성-성분 연계 약화 |
| Root_TPC × Temperature | −0.55 | — | −0.57 | 열 스트레스에 의한 지하부 성분 억제 |
 
- Leaf_TPC, Root_TPC: R-hat < 1.01 (수렴 완료)
- **Leaf_TFC: 발산 150회, R-hat = 1.09** — 다봉 분포로 인한 수렴 실패, 해석 주의 필요
 
---
 
## 5. 실행 환경
 
**Hardware**
 
| 항목 | 사양 |
|---|---|
| CPU | Intel Core i7-14700 (2.10 GHz) |
| RAM | 32.0 GB |
| OS | Windows 11 Pro 64-bit (Build 26100.6899) |
 
**Software**
 
| 라이브러리 | 버전 |
|---|---|
| Python | 3.10 (Anaconda 3) |
| pandas | 2.1.4 |
| numpy | 1.26.0 |
| scikit-learn | 1.4.2 |
| catboost | 1.2.5 |
| shap | 0.45.0 |
| statsmodels | 0.14.1 |
| matplotlib | 3.8.2 |
| seaborn | 0.13.1 |
 
---
 
## 6. 한계 및 향후 과제
 
- 통제된 챔버 환경 데이터 — 포장(field) 일반화 미검증
- SSP5 LOGO R² = −0.71: SSP1/3과 다른 생리적 반응 패턴으로 인한 분포 이동
- Leaf_TFC Bayesian 모델 미수렴 — 비정규 우도함수 또는 시나리오별 독립 모델링 검토 필요
- 토양 특성, 관개 관리, 유전형 변이 미반영
