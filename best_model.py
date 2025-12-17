##########################################
####### Framingham Heart Study ###########
##########################################
# importing needed libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. SUPERVISED MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 2. UNSUPERVISED
from sklearn.cluster import KMeans

# 3. SAMPLING TECHNIQUES
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek

# 4. PIPELINE
from imblearn.pipeline import Pipeline

#######################################
# Importing dataset
#######################################
df = pd.read_csv("data/framingham_cleaned.csv")


##########################################################
###############   Model: Logistic regression  ############
###################  oversampling: SMOTE    ##############
##########################################################

X = df.drop(["TenYearCHD"], axis = 1)
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 11)


pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])



param_grid = {
    'model__C': [0.01, 0.05, 1, 10],
    'model__class_weight': [{0:1, 1:3}, {0:1, 1:5}, 'balanced'],
    'model__solver': ['liblinear', 'lbfgs'],
    'model__max_iter': [100, 1000, 10000]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
   )

grid.fit(X_train, y_train)


print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

joblib.dump(best_model, "best_model.joblib")


pred = best_model.predict(X_test)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##############################################
############  Confusion matrix  ##############
class_names = {
    0: 'No Heart Disease\n(Low Risk)',
    1: 'Heart Disease\n(High Risk)'
}

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 6))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Reds',
            cbar=True,
            cbar_kws={'label': 'Number of Cases'},
            square=True)

plt.title('Heart Disease Prediction - Confusion Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Diagnosis', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=12, fontweight='bold')

# Use descriptive labels
plt.xticks([0.5, 1.5], [class_names[0], class_names[1]], fontsize=11)
plt.yticks([0.5, 1.5], [class_names[0], class_names[1]], fontsize=11, rotation=0)

plt.tight_layout()
plt.show();






