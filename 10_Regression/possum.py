#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_smooth, geom_text, geom_point, geom_line
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
possum =pd.read_csv("../data/possum.csv")
#sns.pairplot(possum.iloc[:, 1:12], hue="totlngth")
possum = pd.get_dummies(possum)
possum.dropna(inplace=True)
corr = possum.corr()
sns.heatmap(corr, vmin=-1, vmax=1, annot=np.round(corr, 2))
possum_X = possum.loc[:, possum.columns != "totlngth"]
X = np.array(possum_X)
y = np.array(possum["totlngth"]).reshape(-1, 1)
print(f"X shape: {X.shape}, y shape: {y.shape}")
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf.get_n_splits(X)
print(kf)
scores = []
model = LinearRegression()
for train_index, test_index in kf.split(X):
    #print(f"{train_index.shape}, {test_index.shape}")
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(np.median(scores))
#%%
steps = [
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=300)),
    #("rf", RandomForestClassifier())
]
pipeline = Pipeline(steps)
#%%
clf = pipeline.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
#%%
print(confusion_matrix(y_true=y_test, y_pred=y_test_pred))
print(accuracy_score(y_test, y_test_pred))
# Variable Importances
importances = RandomForestClassifier(n_estimators=300, random_state=42, bootstrap=True).fit(X_train, y_train).feature_importances_
feature_importances = pd.DataFrame({"features":list(X_train.columns),"importances": importances}).sort_values(by=["importances"], ascending=False)
sns.barplot(data=feature_importances, x ="features", y="importances")
# %%
from scikitplot.metrics import plot_roc_curve
# %%
plot_roc_curve()