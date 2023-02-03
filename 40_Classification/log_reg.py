#%% package
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier

from plotnine import ggplot, aes, geom_bar, labs
# %%
banking = pd.read_csv('../data/direct_marketing.csv', sep=';')
# %%
banking.head()
# %% Target variable
ggplot(data=banking) + aes(x='y') + geom_bar()

# %%
from collections import Counter
Counter(banking['y'])
# %% Naive model
39922 / (39922 + 5289)

# %%
banking_numerical = banking[['age', 'balance', 'day', 'campaign', 'previous', 'duration', 'pdays']]
# %% treatment of categorical features
banking_cat = pd.get_dummies(banking[['default', 'loan', 'marital', 'education', 'job', 'poutcome', 'housing']])
banking_cat

# %% separate independent / dependent features 
X = pd.concat([banking_numerical, banking_cat], axis=1)
y = [0 if i=='no' else 1 for i in banking['y']]
# y = banking['y'].apply(lambda x: 1 if x == 'yes' else 0).tolist()
# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# %% Pipeline
steps = [
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
    # ('rf', RandomForestClassifier())
]

pipeline = Pipeline(steps)
clf = pipeline.fit(X_train, y_train)

# %% create predictions
y_test_pred = clf.predict(X_test)


# %% Model Evaluation
confusion_matrix(y_true=y_test, y_pred=y_test_pred)

# %%
accuracy_score(y_test, y_test_pred)
# %%
