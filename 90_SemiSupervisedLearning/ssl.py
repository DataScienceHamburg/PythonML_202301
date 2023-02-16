#%% package import
# data prep
import pandas as pd
import numpy as np
import random
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# modeling
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# %% load the data
X, y = fetch_california_housing(return_X_y=True)

n_max = 10000
X,y = X[:n_max, :], y [:n_max]

# %% train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# %% scale the data
scaler = StandardScaler()
# scale train
X_train_scaled = scaler.fit_transform(X_train)
# apply scaling from train to test
X_test_scaled = scaler.transform(X_test)

# %%
model_list = [
    LinearRegression(),
    RandomForestRegressor(),
    XGBRegressor()
]

for m in model_list:
    n_folds = 5
    scores = cross_val_score(estimator=m, X=X_train_scaled, y=y_train, cv=n_folds, scoring='r2')
    print(f"{m.__class__.__name__}: R**2 {np.round(scores.mean(), 3)}")
# %%
class PseudoLabel(BaseEstimator, RegressorMixin):
    """Pseudo Labeler Implementation for Regression Problem

    Args:
        BaseEstimator (Sklearn Model): Baseline Regression Model
        RegressorMixin (_type_): _description_
    """
    def __init__(self, model, unlabeled_data, sample_rate) -> None:
        super().__init__()
        self.model = model
        self.unlabeled_data = unlabeled_data
        self.sample_rate = sample_rate
    
    def fit(self, X, y):
        """Fit with Pseudo Labeling

        Args:
            X (np.array): independent features
            y (np.array): dependent feature
        """
        aug_train = self.create_augmented_train(X, y)
        self.model.fit(
            aug_train[:, :-1],
            aug_train[:, -1]
        )
        return self
        
    def create_augmented_train(self, X, y):
        n_total = self.unlabeled_data.shape[0]
        n_samples = int(n_total * self.sample_rate)
        
        # train model based on labeled data
        self.model.fit(X, y)
        
        # predict pseudo labels
        pseudo_labels = self.model.predict(self.unlabeled_data)
        
        # create "labeled" dataset incl. pseudo labels
        pseudo_data = self.unlabeled_data.copy()
        
        # take subset of test with pseudo labels, append onto training data
        sampled_pseudo = np.concatenate((pseudo_data, pseudo_labels.reshape(-1, 1)), axis=1)
        sample_row_ids = random.sample(range(n_total), n_samples)
        sampled_pseudo = sampled_pseudo[sample_row_ids, :]
        
        temp_train = np.concatenate((X, y.reshape(-1, 1)), axis = 1)
        
        augmented_train = np.concatenate((sampled_pseudo, temp_train), axis=0)
        
        return shuffle(augmented_train)
        
    def predict(self, X):
        """Create predictions based on trained model

        Args:
            X (np.array): Array with predictions
        """
        return self.model.predict(X)
        
        
        
# %%
model_list = [
    RandomForestRegressor(),
    PseudoLabel(RandomForestRegressor(), X_test_scaled, sample_rate=0.2)
]

for m in model_list:
    n_folds = 5
    scores = cross_val_score(estimator=m, X=X_train_scaled, y=y_train, cv=n_folds, scoring='r2')
    print(f"{m.__class__.__name__}: R**2 {np.round(scores.mean(), 3)}")

# %%
