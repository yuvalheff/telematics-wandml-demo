# Experiment 2: XGBoost Enhanced Collision Prediction

## Experiment Overview
- **Experiment Name**: XGBoost_Enhanced_Collision_Prediction  
- **Iteration**: 2
- **Primary Change**: Switch from Random Forest to XGBoost with enhanced feature engineering
- **Expected Improvement**: +0.032 Macro PR-AUC (from 0.383 to 0.415)

## Dataset Information
- **Task Type**: Multi-class classification (imbalanced)
- **Target Column**: `collisions`
- **Train Path**: `/Users/yuvalheffetz/ds-agent-projects/session_6a348ddd-12b5-4ee2-af3a-daf992d9a288/data/train_set.csv`
- **Test Path**: `/Users/yuvalheffetz/ds-agent-projects/session_6a348ddd-12b5-4ee2-af3a-daf992d9a288/data/test_set.csv`

## Data Preprocessing Steps

### 1. Missing Value Imputation
- **time_phoneuse_hours**: 13.4% missing → Use median imputation
- **Other numerical features**: 0.7% missing in `count_trip`, `miles`, `drive_hours`, `count_brakes`, `count_accelarations`, `time_speeding_hours`, `highway_miles`, `night_drive_hrs`, `maximum_speed` → Use median imputation
- **Categorical features**: Use mode imputation if any missing values

### 2. Categorical Encoding
- **month**: Apply label encoding to convert month strings to numerical values
- **driver_id**: Remove from features (identifier only)

### 3. Feature Scaling
- **Not required**: XGBoost handles different feature scales naturally

## Feature Engineering Steps

### 1. Risk Behavior Ratios (Exposure-Normalized)
```python
# Risk behavior intensity per trip
brakes_per_trip = count_brakes / max(count_trip, 1)
accel_per_trip = count_accelarations / max(count_trip, 1)

# Risk behavior as proportion of driving time
speeding_ratio = time_speeding_hours / max(drive_hours, 0.1)
phone_ratio = time_phoneuse_hours / max(drive_hours, 0.1)
```

### 2. Exposure Normalization
```python
# Driving environment proportions
highway_ratio = highway_miles / max(miles, 1)
night_ratio = night_drive_hrs / max(drive_hours, 0.1)
```

### 3. Composite Metrics
```python
# Overall risk composite score
risk_score = (brakes_per_trip + accel_per_trip + speeding_ratio + phone_ratio) / 4

# Driving intensity
miles_per_hour = miles / max(drive_hours, 0.1)
```

### 4. Trip Efficiency Metrics
```python
# Trip characteristics
trip_distance_avg = miles / max(count_trip, 1)  # Average distance per trip
trip_duration_avg = drive_hours / max(count_trip, 1)  # Average duration per trip
```

### 5. Log Transformations (Handle Right Skew)
```python
miles_log = log1p(miles)
drive_hours_log = log1p(drive_hours)
count_brakes_log = log1p(count_brakes)
count_accelarations_log = log1p(count_accelarations)
```

## Model Selection and Configuration

### Algorithm: XGBoost Classifier
```python
xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='multi:softprob',
    eval_metric='mlogloss',
    verbosity=0
)
```

### Class Imbalance Handling
- **Method**: Balanced class weights using `sklearn.utils.class_weight.compute_class_weight`
- **Strategy**: `'balanced'` mode to automatically compute weights inversely proportional to class frequencies
- **Rationale**: More effective than SMOTE for extremely imbalanced dataset (only 13 samples in minority class)

### Cross-Validation Strategy
- **Method**: StratifiedKFold with 5 folds
- **Shuffle**: True with random_state=42
- **Purpose**: Maintain class distribution across folds for reliable evaluation

## Evaluation Strategy

### Primary Metric
- **Macro PR-AUC**: Macro-averaged Precision-Recall Area Under Curve across all collision classes

### Secondary Metrics
1. Individual class PR-AUC for classes 0, 1, 2
2. Weighted F1-Score 
3. Macro F1-Score
4. Balanced Accuracy
5. Classification Report with precision, recall, F1 per class

### Diagnostic Analyses

#### 1. Feature Importance Analysis
- Extract XGBoost feature importances
- Create bar chart of top 20 most important features
- Translate technical features to business insights

#### 2. Model Performance Assessment
- **Precision-Recall Curves**: Plot for each class to assess discrimination ability
- **Confusion Matrix**: Analyze prediction patterns across collision classes  
- **Calibration Plot**: Assess probability calibration reliability for each class
- **Cross-Validation Stability**: Evaluate performance consistency across folds

#### 3. Error Analysis
- **Misclassification Analysis**: Identify patterns in incorrectly predicted cases
- **Class Distribution Comparison**: Compare predicted vs actual distributions
- **Threshold Analysis**: Explore optimal thresholds for business decision-making

### Model Interpretation
- **Feature Importance Plot**: Visualize top predictive features
- **Business Insights**: Connect model findings to actionable safety recommendations
- **SHAP Analysis** (optional): Understand individual prediction contributions

## Expected Outputs

### 1. Model Artifacts
- Trained XGBoost model saved with MLflow
- Feature engineering pipeline for consistent preprocessing

### 2. Performance Documentation
- **experiment_summary.json**: Key metrics and findings for tracking
- **experiment_report.md**: Comprehensive technical analysis with business implications

### 3. Visualizations
- `precision_recall_curves.html`: PR curves for all classes
- `confusion_matrix.html`: Prediction accuracy heatmap
- `feature_importance.html`: Feature ranking visualization  
- `calibration_plot.html`: Probability calibration assessment
- `class_distribution.html`: Predicted vs actual class distributions
- `cross_validation_scores.html`: Performance stability across folds

## Success Criteria

### Performance Targets
- **Minimum**: Achieve > 0.400 Macro PR-AUC (improvement over 0.383 baseline)
- **Target**: Achieve ~0.415 Macro PR-AUC based on exploration experiments
- **Consistency**: Cross-validation standard deviation < 0.020

### Interpretability Requirements
- Clear feature importance rankings for business insights
- Actionable recommendations for collision risk reduction

## Implementation Notes

### Critical Requirements
1. **Feature Engineering Order**: Apply after preprocessing, before model training
2. **Validation Strategy**: Use identical 5-fold stratified CV as baseline for fair comparison
3. **Missing Value Consistency**: Ensure same imputation strategy for train and test sets
4. **Class Weight Application**: Calculate from training data only, apply to XGBoost model
5. **Feature Retention**: Keep all engineered features - XGBoost handles high dimensionality effectively

### Quality Assurance
- Verify feature engineering produces expected ranges and distributions
- Confirm class weight computation addresses imbalance appropriately  
- Validate cross-validation maintains stratification across all folds
- Ensure reproducibility with consistent random seeds