#%% packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# %%
diabetes = pd.read_csv('../data/diabetes.csv')
# %%
X, y = diabetes.drop(['Outcome'], axis=1), diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
# %%
steps = [
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=300))
]

pipeline = Pipeline(steps)
clf = pipeline.fit(X_train, y_train)
# %%
y_test_pred = clf.predict(X_test)
# %%
confusion_matrix(y_test_pred, y_test)
# %%
accuracy_score(y_test_pred, y_test)
#%% Variable Importances
importances = RandomForestClassifier(n_estimators=300, random_state=42, bootstrap=True).fit(X_train, y_train).feature_importances_

# %%
feature_importances = pd.DataFrame({'features': list(X_train.columns), 'importances': importances})
feature_importances
# %%
feature_importances = feature_importances.sort_values(by=['importances'], ascending=False)
g = sns.barplot(data=feature_importances, x='features', y='importances')
g.set_xticklabels(labels=feature_importances['features'], rotation=90)
plt.show()
# %%
