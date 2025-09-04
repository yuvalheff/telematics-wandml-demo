# Exploration Experiments Summary

## Overview
Before creating the final experiment plan, I conducted a series of lightweight experiments to test key hypotheses about optimal preprocessing and modeling approaches for the vehicle collision prediction task. The dataset presents significant challenges with severe class imbalance (95.2% no collisions) and requires careful handling.

## Experiment Results

### 1. Baseline Model Performance
**Approach**: Random Forest with minimal preprocessing (median imputation, label encoding)
**Result**: Macro PR-AUC = 0.4205
**Key Findings**: 
- Drive hours, miles, and count_trip emerged as most important features
- Baseline performance establishes reasonable starting point
- Model struggles with extremely rare Class 2 (2 collisions)

### 2. Class Imbalance Handling Strategies
**Tested Approaches**:
- SMOTE oversampling: Macro PR-AUC = 0.4083
- Balanced Random Forest: Macro PR-AUC = 0.3601  
- Class weights: Macro PR-AUC = 0.4205

**Key Findings**:
- **Class weights performed best** - simple but effective for this dataset
- SMOTE actually degraded performance slightly, possibly due to synthetic sample quality issues with extreme imbalance
- Balanced Random Forest underperformed, likely due to aggressive undersampling

### 3. Feature Scaling and Transformation
**Tested Approaches**:
- No preprocessing: Macro PR-AUC = 0.4202
- Log transformation only: Macro PR-AUC = 0.4205
- Standard scaling only: Macro PR-AUC = 0.4208
- **Both transform + scaling: Macro PR-AUC = 0.4215 (BEST)**

**Key Findings**:
- Combined log transformation and standard scaling provided slight but consistent improvement
- Log transformation helps with heavily right-skewed features like miles, drive_hours
- Standard scaling beneficial for ensemble methods and regularized models

### 4. Algorithm Comparison
**Results**:
- **Random Forest: 0.4215 (BEST)**
- XGBoost: 0.4106
- Gradient Boosting: 0.4054
- Logistic Regression: 0.3751

**Key Findings**:
- Random Forest consistently outperformed other algorithms
- Tree-based methods handle missing values and feature interactions well
- Ensemble of RF + XGBoost achieved 0.4222, marginal improvement

### 5. Final Model Analysis
**Best Configuration**: Random Forest with class weights, log transformation, and standard scaling
- Macro PR-AUC: 0.3894 (with hyperparameter tuning)
- Class-wise performance: Class 0: 0.991, Class 1: 0.174, Class 2: 0.003
- Severe difficulty predicting Class 2 due to only 3 test samples

## Key Insights for Experiment Plan

1. **Feature Engineering**: Log transformation of skewed features + standard scaling is beneficial
2. **Class Imbalance**: Class weights are most effective approach for this dataset
3. **Algorithm Selection**: Random Forest is optimal base algorithm
4. **Evaluation Strategy**: Need specialized metrics and analysis for extreme imbalance
5. **Missing Values**: Median imputation is sufficient given low missing rates
6. **Feature Selection**: Strong multicollinearity exists, but tree-based methods handle this naturally

## Recommendations for Main Experiment

1. Use Random Forest as primary algorithm with class weights
2. Apply log transformation + standard scaling preprocessing pipeline
3. Focus evaluation on Class 1 performance (Class 2 too rare for reliable metrics)
4. Include error analysis to understand prediction failures
5. Consider ensemble methods as potential improvement
6. Implement robust cross-validation strategy accounting for class imbalance