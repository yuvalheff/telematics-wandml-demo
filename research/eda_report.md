# Vehicle Collision Prediction - Exploratory Data Analysis Report

## Dataset Overview

**Dataset**: Vehicle collision prediction using telematic data  
**Size**: 7,667 driver-month records with 13 features  
**Task**: Multi-class classification (imbalanced) to predict collision frequency  
**Target**: Number of collisions (0, 1, or 2) per month

## Key Findings Summary

The vehicle collision prediction dataset presents a challenging but workable prediction problem. The analysis reveals significant class imbalance, data quality issues, and strong multicollinearity among features, but also identifies clear predictive signals that can guide successful model development.

## EDA Analysis Steps

### 1. Missing Values Analysis
- **Key Issue**: 13.4% missing values in time_phoneuse_hours feature
- **Pattern**: 0.7% systematic missing values in driving behavior features  
- **Impact**: Requires robust imputation strategy, particularly for phone use data
- **Recommendation**: Consider domain-specific imputation or separate handling for missing phone use data

### 2. Target Distribution Analysis  
- **Severe Class Imbalance**: 95.2% no collisions, 4.6% one collision, 0.2% two collisions
- **Challenge**: Standard accuracy metrics will be misleading
- **Recommendation**: Use PR-AUC, stratified sampling, class weights, or resampling techniques
- **Model Selection**: Algorithms robust to class imbalance (e.g., ensemble methods, cost-sensitive learning)

### 3. Numerical Features Distribution
- **Skewness**: Most features heavily right-skewed with long tails
- **Scale Variance**: Wide range of values (miles: 0-309K, drive_hours: 0-13K)
- **Zero Inflation**: Many features have zero-inflated distributions
- **Recommendation**: Apply log transformation, robust scaling, and consider zero-handling strategies

### 4. Categorical Features Analysis
- **Driver ID**: Very high cardinality (7,663 unique values) with minimal repetition
- **Month**: Even temporal distribution across 12 months (Jan-22 to Dec-22)
- **Data Structure**: Predominantly cross-sectional rather than longitudinal
- **Recommendation**: Driver ID likely not useful as predictor; month may provide seasonal insights

### 5. Feature Correlation Analysis
- **Strongest Predictor**: count_trip shows highest correlation with collisions (r=0.246)
- **Multicollinearity**: Severe correlation between miles, drive_hours, brakes, accelerations (r>0.98)
- **Negative Correlation**: Maximum speed negatively correlated with collisions (r=-0.065)
- **Recommendation**: Feature selection needed to address multicollinearity; focus on trip frequency

### 6. Outlier Detection Analysis  
- **Highest Outliers**: time_speeding_hours (17.0% outliers) indicates extreme behavior
- **General Pattern**: 9-12% outliers in most driving behavior features
- **Clean Features**: Phone use and maximum speed have no outliers
- **Recommendation**: Robust preprocessing with outlier capping or transformation

### 7. Feature-Target Relationship Analysis
- **Clear Progression**: Mean trip counts increase with collision classes (8.1 → 16.4 → 26.0)
- **Strong Separation**: Drivers with 2 collisions consistently have high trip counts (18-30)
- **Risk Pattern**: Trip frequency appears to be exposure-based risk factor
- **Model Insight**: count_trip should be priority feature in model development

## Column Details

| Column | Type | Role | Description | Key Insights |
|--------|------|------|-------------|--------------|
| driver_id | object | identifier | Unique driver identifier | 7,663 unique IDs, minimal repetition |
| month | object | feature | Data collection month | Even temporal distribution, seasonal potential |
| count_trip | float64 | feature | Monthly trip count | Strongest predictor (r=0.246), range 1-31 |
| miles | float64 | feature | Total miles driven | Heavy right-skew, strong multicollinearity |
| drive_hours | float64 | feature | Total driving hours | Near-perfect correlation with miles (r=0.996) |
| count_brakes | float64 | feature | Hard braking events | Exposure-dependent, high correlation with drive time |
| count_accelarations | float64 | feature | Hard acceleration events | Nearly identical to braking pattern |
| time_speeding_hours | float64 | feature | Hours spent speeding | 17% outliers, extreme behavior indicator |
| time_phoneuse_hours | float64 | feature | Phone use while driving | 13.4% missing values, distraction metric |
| highway_miles | float64 | feature | Highway driving miles | Subset of total miles, environment indicator |
| night_drive_hrs | float64 | feature | Night driving hours | Low average with 2.5% outliers |
| maximum_speed | float64 | feature | Maximum speed reached | Controlled range 60-90 mph, negative correlation |
| collisions | int64 | target | Monthly collision count | Severe imbalance: 95.2% zero, 4.8% non-zero |

## Recommendations for ML Pipeline

### Preprocessing Requirements
1. **Missing Value Handling**: Implement robust imputation for phone use data (13.4% missing)
2. **Feature Scaling**: Apply robust scaling or log transformation for skewed features
3. **Outlier Treatment**: Cap or transform extreme values, especially in speeding hours
4. **Feature Selection**: Address multicollinearity by selecting representative features

### Model Development Strategy
1. **Class Imbalance**: Use stratified sampling, class weights, or SMOTE-based resampling
2. **Evaluation Metrics**: Focus on PR-AUC rather than accuracy given severe imbalance
3. **Feature Engineering**: Consider interaction terms between trip frequency and risk behaviors
4. **Model Selection**: Prioritize algorithms robust to class imbalance (Random Forest, XGBoost)

### Key Predictive Features
1. **Primary**: count_trip (strongest predictor, clear relationship with collisions)
2. **Secondary**: time_speeding_hours (risk behavior indicator)
3. **Environmental**: month (potential seasonal effects), highway_miles (driving environment)
4. **Avoid**: Highly correlated features (choose one from miles/drive_hours/brakes cluster)

## Data Quality Assessment

**Overall Quality**: Good with manageable issues  
**Completeness**: 86.6% complete (missing phone use data)  
**Consistency**: High (structured telematic data)  
**Validity**: Good (reasonable ranges for all features)  
**Reliability**: High (objective telematic measurements)  

The dataset provides a solid foundation for collision prediction modeling with clear predictive signals and manageable preprocessing requirements.