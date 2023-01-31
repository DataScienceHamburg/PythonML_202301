#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# %% raw data import
mushrooms = pd.read_csv('../data/mushrooms.csv')

#%% Missing Data?
mushrooms.isnull().sum()
# %% EDA
mushrooms.head()
# %% Data Preparation
# Target Variable
class_map = {"p": 0, "e":1} 
mushrooms['class_num'] = [class_map[i] for i in mushrooms['class']] 
mushrooms.drop(columns='class', inplace=True)
mushrooms= mushrooms[['class_num', 'cap-shape', 'gill-spacing']]

#%% Target Variable Distribution
sns.countplot(data=mushrooms, x='class_num')
# both features are nearly balanced

#%% get count per class number
from collections import Counter
Counter(mushrooms['class_num'])

# %% Independent Features
# convert all categorical features with dummy variables
mushrooms_filt = pd.get_dummies(mushrooms)
# %% Separate Independent and Dependent Features
X, y = mushrooms_filt.drop(columns='class_num'), mushrooms_filt['class_num']

# %% Train / Test Split of Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}\nX_test: {X_test.shape}, y_test: {y_test.shape}")
# %% scale the data
scaler = StandardScaler()
# scale train
X_train_scaled = scaler.fit_transform(X_train)
# apply scaling from train to test
X_test_scaled = scaler.transform(X_test)
# %% Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

#%% Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

# %% create preds
y_pred_logreg_test = log_reg.predict(X_test_scaled)
y_pred_rf_test = rf.predict(X_test_scaled)

#%% Feature Importance
feature_importance = pd.DataFrame(
{'feature':list(X_train.columns),
'feature_importance':[abs(i) for i in rf.feature_importances_]}
)
sns.set(rc={'figure.figsize':(16,8)})
p = sns.barplot(data=feature_importance, x='feature', y='feature_importance')
p.set_xticklabels(p.get_xticklabels(),rotation=90)
# %% Model Evaluation
# calculate normalized confusion matrix
test_cnt = len(y_test)
cm = confusion_matrix(y_pred=y_pred_logreg_test, y_true=y_test) / test_cnt

sns.heatmap(cm, annot=True, fmt='.2f')

# %% Accuracy Score
accuracy_score(y_pred_logreg_test, y_test)
# %% Classification Report
print(classification_report(y_test, y_pred_logreg_test))

# %% Create ROC curve
y_pred_logreg_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_logreg_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_prob)
# %%
plt.plot(fpr_logreg,tpr_logreg)
plt.plot(fpr_rf,tpr_rf, 'r')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# %%
