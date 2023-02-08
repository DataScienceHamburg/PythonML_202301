#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from plotnine import ggplot, aes, geom_col, theme, element_text
# %%
retail = pd.read_excel('../data/OnlineRetail.xlsx')
# %%
retail.head()

# %%
retail['Description'] = retail['Description'].str.strip()
# %%
retail.shape
# %% lösche Zeile ohne Rechnungsnummer
retail.dropna(subset=['InvoiceNo'], axis=0, inplace=True)
retail.shape

# %% Modellierung
# fokussiere auf deutschen Markt
basket_germany = (retail[retail['Country'] == 'Germany'].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

# %%
def hot_encode(x):
    if (x <= 0):
        return 0 
    if (x >= 1):
        return 1
# %%
basket_germany = basket_germany.applymap(hot_encode)
# %% Apriori
freq_items = pd.DataFrame(apriori(basket_germany, min_support=0.05, use_colnames=True).sort_values('support', ascending=False))

# %%
ggplot(data = freq_items) + aes(x='itemsets', y='support') + geom_col() + theme(axis_text=element_text(rotation=90))

#%% Rules
rules = association_rules(freq_items, metric='lift')
# %% alle Sets mit >= 2 Produkten, Support mind. 10 %
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))

# %%
freq_items[(freq_items['length']>=2) & (freq_items['support']>= 0.1)]
# %%
