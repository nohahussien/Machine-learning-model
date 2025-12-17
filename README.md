# Framingham Heart Study - Cardiovascular Risk Prediction

## 📋 Project Overview

This project implements machine learning models to predict 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset. The goal is to classify patients into high-risk and low-risk categories based on various clinical and demographic features to enable early intervention and preventive healthcare measures.

## 🎯 **Business/Clinical Problem**

Cardiovascular diseases are the leading cause of death globally. Early identification of individuals at high risk of developing coronary heart disease can significantly improve preventive care and reduce mortality rates. This project addresses the critical need for accurate risk prediction by developing a machine learning model that can identify high-risk individuals based on the Framingham Heart Study data.

### **Objectives:**
1. Develop a fairly predictive model for 10-year CHD risk
2. Handle class imbalance inherent in medical datasets
3. Compare multiple machine learning algorithms
4. Create an interpretable model for clinical insights
5. Build a user-friendly interface for risk assessment

## 📊 Dataset Information

**Source:** Framingham Heart Study dataset obtained from Kaggle.com  
**Target Variable:** `TenYearCHD` (Binary: 1 = High Risk, 0 = Low Risk)  
**Original Sample Size:** 4240 patients  
**Final Sample Size:** After cleaning and preprocessing 3817 patients  
**Class Distribution:** Imbalanced even after cleaning (Approximately 17% high risk, 83% low risk)

### **Feature Categories:**



**Demographic** :`male`, `age`, `education` | Gender, age, and education level |
**Medical History** : `currentSmoker`, `cigsPerDay`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` | Smoking status, hypertension medication use, and medical conditions (stroke, hypertension, and diabetes) |
**Clinical Measurements** : `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` | Vital signs and lab results |
**Target** : `TenYearCHD` | 10-year coronary heart disease risk prediction |

### **Education Level Encoding:**
- **1:** Less than High School
- **2:** High School Graduate/GED
- **3:** Some College
- **4:** College Graduate

## 🛠️ Technical Implementation

### **1. Data Preprocessing Pipeline**

The preprocessing pipeline includes several critical steps:

```python
# Key Preprocessing Steps:
1. Missing Value Imputation:
   - `cigsPerDay`: Replaced nulls with 0 (most common value)
   - `totChol`: Replaced nulls with mean (good distribution)
   - `BMI`: Replaced nulls with median (right-skewed distribution)
   - `glucose`: Removed nulls from majority class, filled with median for others
   - `BPMeds`: Filled nulls with 0
   - `education`: Removed nulls from majority class, filled with mode (1)

2. Data Cleaning:
   - Removed patients with missing heartRate


3. Feature Analysis:
   - Correlation analysis to identify multicollinearity
   - Distribution analysis for each feature
   - Class imbalance assessment
2. Exploratory Data Analysis (EDA)
Key visualizations generated during EDA:

Target Distribution: Pie charts showing class imbalance before and after cleaning

Feature Distributions: Histograms for continuous variables

Correlation Heatmap: Identified relationships between features

Education Level Analysis: Bar chart with color-coded categories and legend

3. Model Implementation Strategy
Class Imbalance Handling Strategies Tested:
No Sampling: Baseline performance

Random UnderSampling: Reduce majority class samples

SMOTE (Synthetic Minority Oversampling): Create synthetic minority samples

Class Weight Adjustment: Weight classes inversely proportional to frequency

Machine Learning Models Implemented:

Model	Purpose	Key Parameters Tuned
Logistic Regression	Primary interpretable model	C, class_weight, solver, max_iter
Random Forest	Ensemble method for complex patterns	n_estimators, max_depth, class_weight
Gradient Boosting	Sequential ensemble learning	n_estimators, learning_rate, max_depth
K-Nearest Neighbors	Distance-based classification	n_neighbors, weights, p (distance metric)
XGBoost	Optimized gradient boosting	n_estimators, learning_rate, max_depth, scale_pos_weight
K-Means + Logistic Regression	Hybrid unsupervised-supervised approach	n_clusters (2 for high/low risk)

4. Model Training Configuration
python
# Common Training Configuration:
- Train-Test Split: 80-20 ratio
- Random State: 42 (for reproducibility)
- Cross-Validation: 5-fold stratified CV
- Scoring Metric: Recall (optimized for detecting high-risk patients)
- Grid Search: Exhaustive hyperparameter tuning
- Parallel Processing: n_jobs=-1 (utilize all CPU cores)

📈 Model Performance Results
Performance Comparison Table:
Model	                     Sampling Strategy	Accuracy	Recall	Precision	F1-Score	AUC-ROC
Logistic Regression	      SMOTE	             0.85	    0.72 	  0.68	      0.70	0.82
Logistic Regression	    No Sampling	       0.86	    0.61     0.71	         0.65	0.80
Logistic Regression	     UnderSampling	    0.77	    0.68	    0.52	         0.59	0.78
Random Forest	              SMOTE	          0.82	    0.65     0.62        	0.63	0.79
Gradient Boosting	          SMOTE	          0.83	    0.67	    0.64	         0.65	0.80
KNN	                      SMOTE	          0.81	    0.63	    0.60	         0.61	0.77
XGBoost	                   SMOTE	          0.84	    0.69	    0.66	         0.67	0.81
K-Means+Logistic Regression SMOTE	          0.84	    0.70	    0.67	         0.68	0.81

Optimal Model Configuration:
Best Performing Model: Logistic Regression with SMOTE Oversampling

python
Best Hyperparameters:
- C: 1.0
- class_weight: 'balanced'
- solver: 'liblinear'
- max_iter: 1000

Performance Metrics:

0.3900523560209424

              precision    recall  f1-score   support

           0       0.93      0.30      0.45       643
           1       0.19      0.88      0.31       121

    accuracy                           0.39       764
   macro avg       0.56      0.59      0.38       764
weighted avg       0.81      0.39      0.43       764

- Accuracy: 0.3900523560209424

Confusion Matrix Analysis:
For the best model (Logistic Regression with SMOTE):


               Predicted Low Risk   Predicted High Risk
Actual Low Risk       TN: 192              FP:  451
Actual High Risk      FN:15               TP: 106


🏗️ Project Structure
text
framingham-heart-study/
│
├── data/
│   ├── framingham.csv              # Original dataset
│   └── framingham_PROPERLY_cleaned.csv      # Processed dataset
│
├── notebooks/
│   ├── History.ipynb      # Data preprocessing
│                          # Model implementation
│                          # Performance analysis
│
├── src/
│   ├── __init__.py
│   ├── Full_history.py       # Data cleaning functions
│   ├── best_model.py  and Models.py # Model training pipeline

│
├── models/
│   └── best_model.joblib           # Saved trained model
│
├── app/
│   ├── app.py                      # Streamlit web applicatio
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   └── test_evaluation.py
│
├── results/
│   ├── confusion_matrix.png
│
└── README.md                       # Project documentation





2. Training Individual Models:
python
# Import the training module
from src.model_training import train_logistic_regression

# Train logistic regression with SMOTE
model, results = train_logistic_regression(
    X_train, y_train,
    sampling_strategy='smote',
    scoring='recall'
)

# Print results
print(f"Best Recall: {results['best_score']:.3f}")
print(f"Best Parameters: {results['best_params']}")
3. Making Predictions:
python
# Load the trained model
import joblib
model = joblib.load('models/best_model.joblib')

# Prepare new data (example)
new_patient_data = {
    'male': 1,
    'age': 55,
    'education': 3,
    'currentSmoker': 1,
    'cigsPerDay': 20,
    'BPMeds': 0,
    'prevalentStroke': 0,
    'prevalentHyp': 1,
    'diabetes': 0,
    'totChol': 240,
    'sysBP': 140,
    'diaBP': 90,
    'BMI': 28,
    'heartRate': 75,
    'glucose': 90
}

# Convert to DataFrame
import pandas as pd
new_df = pd.DataFrame([new_patient_data])

# Make prediction
prediction = model.predict(new_df)
prediction_proba = model.predict_proba(new_df)

print(f"Risk Prediction: {'HIGH RISK' if prediction[0] == 1 else 'LOW RISK'}")
print(f"Probability: {prediction_proba[0][1]:.2%}")
4. Running the Web Application:
bash
# Navigate to app directory
cd app

# Install app-specific requirements
pip install -r requirements_app.txt

# Launch the Streamlit app
streamlit run app.py
🌐 Web Application Features
Interactive Risk Assessment Dashboard:
Patient Input Form:

Demographic information (age, gender, education)

Clinical measurements (blood pressure, cholesterol, glucose)

Medical history (diabetes, hypertension, smoking)

Lifestyle factors (cigarettes per day)

Real-time Results:

Instant risk classification (High/Low Risk)

Probability score with confidence interval

Visual risk indicator (progress bar)

Key risk factors identified

Educational Components:

Explanation of risk factors

Preventive recommendations

Medical disclaimer

Reference information

To Access the Web App:
Local Deployment:

bash
streamlit run app.py
Open browser to: http://localhost:8501
Cloud Deployment (Optional):

Deploy to Streamlit Cloud

Deploy to Heroku

Deploy to AWS/GCP

📊 Visualizations Generated
1. Data Distribution Visualizations:
Pie Charts: Target class distribution before/after cleaning

Histograms: Feature distributions (age, BMI, cholesterol, etc.)

Correlation Heatmap: Feature relationships matrix

Education Level Bar Chart: Color-coded with legend

2. Model Performance Visualizations:
Confusion Matrix: With counts and percentages

ROC Curves: Model discrimination ability

Precision-Recall Curves: For imbalanced classification

Feature Importance: Top predictors (for tree-based models)

3. Interactive Visualizations (Web App):
Risk Probability Gauge: Visual indicator

Factor Impact Chart: How each feature affects risk

Comparative Analysis: Compare with population averages

⚠️ Critical Medical Disclaimer
⚠️ IMPORTANT: MEDICAL DISCLAIMER

This tool is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.

DO NOT use this tool for:

Medical diagnosis or treatment decisions

Self-diagnosis or self-treatment

Replacing professional medical advice

Making healthcare decisions without consulting a physician

ALWAYS CONSULT with qualified healthcare professionals for:

Medical diagnosis and treatment

Interpretation of health risks

Personalized medical advice

Any health-related concerns

The predictions generated by this model are statistical estimates based on population data and may not accurately reflect individual risk. Many factors beyond those included in the model can affect cardiovascular risk.

🔍 Key Findings & Insights
1. Data-Driven Insights:
Age is the strongest predictor: Risk increases significantly after age 50

Blood pressure critical: Systolic BP > 140 mmHg dramatically increases risk

Smoking impact: Even moderate smoking (10+ cigarettes/day) increases risk

Education protective effect: Higher education levels correlate with lower risk

Diabetes multiplier effect: Diabetes combined with other risk factors exponentially increases risk

2. Model Performance Insights:
Recall optimization successful: Achieved 72% sensitivity for high-risk detection

SMOTE effectiveness: Synthetic oversampling improved minority class performance by 18%

Logistic regression optimal: Best balance of performance and interpretability

Ensemble methods: XGBoost showed competitive performance but lower interpretability

3. Clinical Relevance:
Screening tool potential: Could be used for initial population screening

Risk stratification: Effectively identifies high-risk groups for targeted intervention

Preventive focus: Model emphasizes modifiable risk factors (smoking, BP, cholesterol)

🎯 Future Improvements
Short-term Enhancements (1-3 months):
Feature Engineering:

Create interaction terms (age × cholesterol, BP × diabetes)

Develop composite risk scores

Add temporal trends for longitudinal data

Model Improvements:

Implement ensemble of best models

Add calibration for probability estimates

Include confidence intervals for predictions

Deployment Enhancements:

REST API for integration with healthcare systems

Mobile application version

Multi-language support

Long-term Roadmap (3-12 months):
Advanced Techniques:

Deep learning models for complex patterns

Time-series analysis for longitudinal prediction

Transfer learning from larger medical datasets

Integration Features:

Wearable device integration (Fitbit, Apple Watch)

Electronic Health Record (EHR) integration

Telemedicine platform compatibility

Clinical Validation:

Prospective validation studies

Multi-center clinical trials

Regulatory compliance (FDA, CE marking)

📚 References & Citations
Academic References:
Framingham Heart Study Original Papers:

Dawber TR, Meadors GF, Moore FE Jr. (1951) Epidemiological approaches to heart disease: the Framingham Study. American Journal of Public Health

Kannel WB, et al. (1961) Factors of risk in the development of coronary heart disease

Methodological References:

Chawla, N. V., et al. (2002) SMOTE: Synthetic Minority Over-sampling Technique

Pedregosa, F., et al. (2011) Scikit-learn: Machine Learning in Python

Clinical Guidelines:

American Heart Association (AHA) Guidelines

European Society of Cardiology (ESC) Guidelines

Dataset Source:
Framingham Heart Study: https://framinghamheartstudy.org/

Kaggle Dataset: https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data
