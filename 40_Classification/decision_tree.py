#%% packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# %%
diabetes = pd.read_csv('../data/diabetes.csv')
# %% correlation
corr = diabetes.corr()
sns.heatmap(corr, annot=corr)
plt.show()
# %%
X, y = diabetes.drop(['Outcome'], axis=1), diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# %%
steps = [
    ('scaler', StandardScaler()),
    ('decision_tree', DecisionTreeClassifier())
]

pipeline = Pipeline(steps)
clf = pipeline.fit(X_train, y_train)

# %%
y_test_pred = clf.predict(X_test)
# %%
cm = confusion_matrix(y_pred=y_test_pred, y_true=y_test)
cm
# %% Baseline classifier
from collections import Counter
Counter(y_test)
50/(50+27)  # accuracy for baseline classifier
# %%
accuracy_score(y_true=y_test, y_pred=y_test_pred)
# %%
print(classification_report(y_true=y_test, y_pred=y_test_pred))
# %%
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
plot_tree(dt)
plt.show()
# %%
