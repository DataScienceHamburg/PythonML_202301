#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import random
# %%
my_range = np.linspace(start=-10, stop=10, num=100000)
x1 = random.sample(population=set(my_range), k=1000)
x2 = random.sample(population=set(my_range), k=1000)
z = [x1**2+ x2**2 for x1, x2 in zip(x1, x2)]
z_class = [1 if i < 50 else 0 for i in z]
# %%
sns.scatterplot(x=x1, y=x2, hue=z_class)
plt.show()
# %% separate  independent / dependent
X = np.array([x1, x2], )
y = z_class


# %%
clf = SVC(kernel='rbf')
clf.fit(X, y)
# %%
y_pred = clf.predict(X)
# %%
sns.scatterplot(x=x1, y=x2, hue=y_pred)
plt.show()
# %%
