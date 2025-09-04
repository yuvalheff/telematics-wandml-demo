# Vehicle Collision Prediction - Experiment 1 Plan

## Experiment Overview
**Name**: RF_Baseline_Collision_Prediction  
**Task Type**: Multi-class classification (imbalanced)  
**Target Column**: collisions  
**Primary Metric**: PR-AUC (Precision-Recall Area Under Curve) - Macro-averaged  

## 1. Data Preprocessing Steps

### 1.1 Data Cleaning
1. **Remove identifier column**: Drop 'driver_id' from both train and test sets
2. **Categorical encoding**: Encode 'month' column using LabelEncoder fitted on train set, applied to both train and test sets

### 1.2 Missing Value Handling
- Use **median imputation** for all numeric features
- Calculate median values from train set only, apply to both train and test
- **Affected columns**: 'count_trip', 'miles', 'drive_hours', 'count_brakes', 'count_accelarations', 'time_speeding_hours', 'time_phoneuse_hours', 'highway_miles', 'night_drive_hrs', 'maximum_speed'

### 1.3 Feature Transformation
1. **Log transformation**: Apply log1p transformation to right-skewed features:
   - 'miles', 'drive_hours', 'count_brakes', 'count_accelarations', 'time_speeding_hours', 'time_phoneuse_hours', 'highway_miles', 'night_drive_hrs'

2. **Feature scaling**: Apply StandardScaler to all numeric features
   - Fit scaler on train set only
   - Transform both train and test sets

## 2. Feature Engineering Steps

### 2.1 Exposure-Normalized Features
Create trip-normalized metrics to reduce multicollinearity:
- **brakes_per_trip** = count_brakes / count_trip
- **accel_per_trip** = count_accelarations / count_trip  
- **miles_per_trip** = miles / count_trip
- **hours_per_trip** = drive_hours / count_trip

*Note: Handle division by zero by setting result to 0 when count_trip is 0*

### 2.2 Risk Behavior Ratios
- **highway_ratio** = highway_miles / miles (set to 0 when miles is 0)
- **speeding_ratio** = time_speeding_hours / drive_hours (set to 0 when drive_hours is 0)  
- **phone_ratio** = time_phoneuse_hours / drive_hours (set to 0 when drive_hours is 0)

### 2.3 Composite Risk Score
- **risk_score** = standardized sum of time_speeding_hours, time_phoneuse_hours, and night_drive_hrs

### 2.4 Final Feature Set
Retain both original preprocessed features and new engineered features for comprehensive model training.

## 3. Model Selection Strategy

### 3.1 Primary Algorithm
**RandomForestClassifier** with parameters:
- n_estimators=200
- max_depth=12
- min_samples_split=10
- min_samples_leaf=5
- class_weight='balanced'
- random_state=42

### 3.2 Hyperparameter Tuning
Use **stratified 5-fold cross-validation** with PR-AUC scoring:
- **n_estimators**: [100, 200, 300]
- **max_depth**: [8, 10, 12, 15]
- **min_samples_split**: [5, 10, 20]
- **min_samples_leaf**: [2, 5, 10]

### 3.3 Alternative Models for Comparison
1. **XGBClassifier** with sample weights derived from balanced class weights
2. **GradientBoostingClassifier** with balanced class weighting strategy

### 3.4 Ensemble Approach
- Simple voting ensemble of top 2 performing individual models
- Use probability averaging for final predictions

### 3.5 Feature Selection
- Apply Recursive Feature Elimination with Cross-Validation (RFECV) on Random Forest
- Retain minimum of 5 features
- Select optimal feature subset based on cross-validation PR-AUC

## 4. Evaluation Strategy

### 4.1 Performance Metrics
- **Primary**: Macro-averaged PR-AUC across classes 0, 1, and 2
- **Secondary**: F1-score (macro and weighted), precision and recall per class
- **Reporting**: Classification report and confusion matrix analysis

### 4.2 Class-Specific Analysis
- Detailed analysis of **Class 1 (1 collision)** performance
- Class 2 analysis with acknowledgment of limited sample reliability
- Precision-recall trade-off analysis and optimal threshold selection

### 4.3 Feature Importance Analysis
- Extract and visualize Random Forest feature importances
- Compare original vs engineered feature rankings
- Identify most predictive driving behaviors for collision risk

### 4.4 Error Analysis
- Characterize misclassified samples
- Analyze false positive and false negative patterns
- Examine prediction confidence distributions across classes

### 4.5 Model Calibration
- Assess prediction probability calibration using reliability diagrams
- Critical for safety applications requiring well-calibrated risk estimates

### 4.6 Business Impact Analysis
- Calculate potential collision prevention rates at different probability thresholds
- Estimate false alarm rates and operational implications
- Simulate deployment scenarios for fleet safety applications

### 4.7 Cross-Validation Robustness
- Report mean and standard deviation of CV scores
- Assess model stability across different data splits
- Use stratified CV to maintain class distribution integrity

## Expected Outputs

1. **Trained model**: Optimally tuned RandomForestClassifier
2. **Performance report**: Comprehensive evaluation metrics and analysis
3. **Feature importance rankings**: Most predictive collision risk factors
4. **Error analysis report**: Patterns in model failures and limitations
5. **Calibration analysis**: Reliability of predicted collision probabilities
6. **Business impact assessment**: Actionable insights for fleet safety implementation

## Success Criteria

- Achieve **macro PR-AUC > 0.40** on test set
- Demonstrate **Class 1 recall > 0.35** (collision detection capability)
- Provide **interpretable feature importance** rankings for business insights
- Establish **calibrated probability thresholds** for operational deployment