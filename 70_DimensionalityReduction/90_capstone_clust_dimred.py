#%% Data Description
#
# source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
# 
# Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.

# Attributes

# People

# ID: Customer's unique identifier
# Year_Birth: Customer's birth year
# Education: Customer's education level
# Marital_Status: Customer's marital status
# Income: Customer's yearly household income
# Kidhome: Number of children in customer's household
# Teenhome: Number of teenagers in customer's household
# Dt_Customer: Date of customer's enrollment with the company
# Recency: Number of days since customer's last purchase
# Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# Products

# MntWines: Amount spent on wine in last 2 years
# MntFruits: Amount spent on fruits in last 2 years
# MntMeatProducts: Amount spent on meat in last 2 years
# MntFishProducts: Amount spent on fish in last 2 years
# MntSweetProducts: Amount spent on sweets in last 2 years
# MntGoldProds: Amount spent on gold in last 2 years
# Promotion

# NumDealsPurchases: Number of purchases made with a discount
# AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# Place

# NumWebPurchases: Number of purchases made through the company’s website
# NumCatalogPurchases: Number of purchases made using a catalogue
# NumStorePurchases: Number of purchases made directly in stores
# NumWebVisitsMonth: Number of visits to company’s website in the last month
# Target
# Need to perform clustering to summarize customer segments.

#%% packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
#%% Data Import
#--------------
df_raw = pd.read_csv('data//marketing_campaign.csv', sep='\t')
# %% Exploartory Data Analysis
#-----------------------------
# shape of dataframe
df_raw.shape

# %% Head of Data
df_raw.head()

#%%  get more info about data
df_raw.info()
# %% Clean the data by deleting rows with NaNs
df_filt = df_raw.copy().dropna()
df_filt.shape
# %% Feature Engineering
#-----------------------

# %% create column 'Age' from birth year
df_filt['Age'] = 2022 - df_filt['Year_Birth']

# %% create column 'MoneyTotal' from columns starting with Mnt...
df_filt['MoneyTotal'] = df_filt["MntWines"] + df_filt["MntFruits"] + df_filt["MntMeatProducts"] + df_filt["MntFishProducts"] + df_filt["MntSweetProducts"] + df_filt["MntGoldProds"]
# %% create column 'KidsTotal' from 'KidHome' and 'TeenHome'
df_filt['KidsTotal'] = df_filt['Kidhome'] + df_filt['Teenhome']
# %% create column 'IsParent' 
# hint: use list comprehension or np.where()
df_filt['IsParent'] = [1 if i > 0 else 0 for i in df_filt['KidsTotal']]


#%% column Education
def EducationToNum(education):
    if education == 'Basic':
        return 0
    elif education == '2n Cycle':
        return 1
    elif education == 'Graduation':
        return 2
    elif education == 'Master':
        return 3
    elif education == 'PhD':
        return 4
    
df_filt['EducationNum'] = df_filt['Education'].apply(EducationToNum)
    
#%% Feature Correlation
cols_to_plot = ['MoneyTotal', 'KidsTotal', 'Income', 'Age', 'IsParent']
sns.pairplot(df_filt[cols_to_plot], hue='IsParent')
plt.show()
# %% Can you spot outliers? If so, filter these
sns.displot(df_filt, x='Age', kind='kde') 
plt.show()
df_filt = df_filt[df_filt['Age'] < 85]

# %%
sns.displot(df_filt, x='Income', kind='kde') 
plt.show()
df_filt = df_filt[df_filt['Income'] < 200000]

#%% delete columns that are not needed any more
df_filt.drop(columns=['ID', 'Z_CostContact', 'Education'], inplace=True)

# %% Heatmap showing feature correlation
cols_to_plot = ['MoneyTotal', 'KidsTotal', 'Income', 'Age', 'IsParent', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases']
corr = df_filt[cols_to_plot].corr()
plt.figure(figsize=(10,10))  
sns.heatmap(corr, annot=True, center=0)
plt.show()
#%% 
cols_to_keep = ['Income', 'NumWebPurchases','NumCatalogPurchases',
       'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'MoneyTotal', 'KidsTotal',
       'IsParent', 'EducationNum']

# %% Modeling
#------------
#%% Scaling
pca = PCA(n_components=3)
steps = [
    ('scalar', StandardScaler()),
    ('pca', pca)
]



pipeline = Pipeline(steps)

prin_comp = pipeline.fit_transform(df_filt[cols_to_keep])

#%% PCA
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
pca_loading_matrix = pd.DataFrame(pca_loadings, columns =    
                        ['PC{}'.format(i) for i in range(1, 4)], index = df_filt[cols_to_keep].columns                         
                        )
sns.heatmap(pca_loading_matrix, cmap=sns.diverging_palette(200, 30, n=200), annot=True)
plt.show()
# %%
prin_df = pd.DataFrame(data = prin_comp
             , columns = ['PC1', 'PC2', 'PC3'])


#%% Elbow Method
elbow = KElbowVisualizer(KMeans(), k=10)
elbow.fit(prin_df)
elbow.show()

# %% Kmeans clustering
clustering = KMeans(n_clusters=4).fit(prin_comp)

# %%
df_filt['cluster']  = clustering.labels_
df_filt['cluster'] = df_filt['cluster'].astype("category")
# %% Result Evaluation
#---------------------

sns.countplot(df_filt['cluster'])
plt.show()
# %%
sns.scatterplot(data= df_filt, x = 'Income', y='MoneyTotal', hue='cluster', alpha=.2)
plt.show()
# %%
sns.violinplot(data=df_filt, x='cluster', y='MoneyTotal')
plt.show()
# %%
