#%% Logistic Regression

# The dataset is provided by UCI Machine Learning repository and deals with direct marketing of a bank. The target variable describes a customer subscribing (1) to a deposit or not (0).


# %% Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from plotnine import ggplot, aes, geom_bar, labs

#%% Data Preparation

banking = pd.read_csv("../data/direct_marketing.csv", sep=';')

#%% 
banking.describe()

#%% 
banking.head()

#%%  Target Variable
(ggplot(data=banking) +
 aes(x='y') +
 geom_bar() +
 labs(title = "Target Variable Count", y = "Count", x = "Target Variable")
)
# %% Filter Data

# The new object banking_filt only holds these information:

cols_to_keep = ['age','balance','day', 'campaign', 'previous']
banking_numerical = banking[cols_to_keep]
y = banking['y'].apply(lambda x: 1 if x == 'yes' else 0).tolist()

# %% transform categorical data into numerical
banking_cat = pd.get_dummies(banking[['default', 'housing', 'loan','marital', 'education', 'job', 'poutcome']])

#%% combine numeric and categorical columns
X = pd.concat([banking_numerical, banking_cat], axis=1)



#%% Perform Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Set up Model Pipeline with StandardScaler, LogisticRegression
steps = [
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
]

pipeline = Pipeline(steps)

clf = pipeline.fit(X_train, y_train)


#%% Calculate Predictions on Test data
y_pred = clf.predict(X_test)


#%% Calculate Baseline Classifier
1 - np.sum(y_test) / len(y_test)

#%% Calculate Confusion Matrix on Test data
confusion_matrix(y_true=y_test, y_pred=y_pred)

#%% Compare our Model Accuracy to Baseline Model Accuracy
accuracy_score(y_test, y_pred)
# %%
print(classification_report(y_test, y_pred))
# %%
