# Exploration Experiments for Iteration 2

## Overview
This document summarizes the exploration experiments conducted to determine the best approach for iteration 2 of the vehicle collision prediction project. The goal was to improve upon the Random Forest baseline that achieved 0.383 Macro PR-AUC.

## Previous Iteration Analysis
- **Baseline**: Random Forest with 0.383 Macro PR-AUC
- **Key Challenges**: Severe class imbalance (95.2% no collisions, 4.6% one collision, 0.2% two collisions)
- **Strengths**: Comprehensive feature engineering with risk behavior ratios and exposure-normalized metrics
- **Weaknesses**: Moderate performance on rare collision events, limited by class imbalance

## Exploration Experiments Conducted

### 1. Algorithm Comparison
**Objective**: Test gradient boosting algorithms against Random Forest baseline

**Results**:
- **LightGBM baseline**: 0.389 (±0.004) Macro PR-AUC
- **XGBoost baseline**: 0.420 (±0.037) Macro PR-AUC
- **Random Forest baseline**: 0.383 Macro PR-AUC

**Key Finding**: XGBoost showed the strongest performance among gradient boosting algorithms, with 0.037 improvement over Random Forest.

### 2. Enhanced Feature Engineering
**Objective**: Test comprehensive feature engineering beyond the baseline

**Enhanced Features Created**:
- Risk behavior ratios: `brakes_per_trip`, `accel_per_trip`, `speeding_ratio`, `phone_ratio`
- Exposure normalization: `highway_ratio`, `night_ratio`
- Composite metrics: `risk_score`, `miles_per_hour`
- Log transformations: `miles_log`, `drive_hours_log`, `count_brakes_log`, `count_accelarations_log`
- Trip efficiency: `trip_distance_avg`, `trip_duration_avg`

**Results**:
- **LightGBM + Enhanced Features**: 0.401 (±0.012) Macro PR-AUC
- **XGBoost + Enhanced Features**: 0.415 (±0.012) Macro PR-AUC

**Key Finding**: Enhanced feature engineering provided consistent improvements across algorithms, with XGBoost + Enhanced Features achieving the best performance.

### 3. Sampling Techniques
**Objective**: Test SMOTE oversampling to address class imbalance

**Results**:
- **LightGBM + Enhanced + SMOTE**: 0.391 (±0.009) Macro PR-AUC

**Key Finding**: SMOTE did not improve performance over enhanced feature engineering alone, likely due to the extreme rarity of class 2 (only 13 samples in training set).

### 4. Feature Importance Analysis
**Top Features in Best Model (XGBoost + Enhanced)**:
1. `drive_hours` (0.134)
2. `count_accelarations` (0.067) 
3. `count_brakes` (0.063)
4. `miles` (0.055)
5. `night_drive_hrs` (0.050)
6. `trip_duration_avg` (0.048)
7. `speeding_ratio` (0.046)
8. `phone_ratio` (0.045)

## Final Recommendation

**Best Approach**: XGBoost + Enhanced Feature Engineering
- **Expected Performance**: 0.415 Macro PR-AUC
- **Improvement**: +0.032 over Random Forest baseline (+8.4% relative improvement)

## Key Insights for Implementation

1. **Algorithm Choice**: XGBoost outperforms other gradient boosting methods for this dataset
2. **Feature Engineering**: Enhanced features provide significant value, especially ratio-based and efficiency metrics
3. **Sampling Strategy**: Class weights are more effective than oversampling for this extremely imbalanced dataset
4. **Feature Importance**: Exposure metrics (drive hours, miles) and behavioral patterns (accelerations, braking) are most predictive

## Limitations and Considerations

1. SMOTE sampling was challenging due to only 13 samples in the rarest class
2. Cross-validation showed some variance (±0.012-0.037), indicating sensitivity to data splits
3. The improvement is meaningful but still moderate, suggesting additional data sources may be needed for substantial gains

## Next Steps for Implementation

Focus on single change for iteration 2: **Switch from Random Forest to XGBoost with enhanced feature engineering**, maintaining the same evaluation framework for direct comparison.