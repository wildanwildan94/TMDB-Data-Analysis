# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import scipy.stats as stats
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from collections import Counter
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import itertools
plt.rcParams['figure.dpi'] = 100
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from IPython.display import HTML, Math
display(HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
             "latest.js?config=default'></script>"))
Math(r"e^\alpha")


### Load Data


train_adj_d=pd.read_csv('train_adj_r10_t01_d.csv')
val_adj_d=pd.read_csv('test_adj_r10_t01_d.csv')


print "---"
print "Q: What is the size of train_adj_d?"
print train_adj_d.shape
print "---"

## Split data into train and test
train_split_adj_d, test_split_adj_d=train_test_split(train_adj_d, test_size=0.1)


print "---"
print "Q: What is the shape of train_split_adj_d?"
print train_split_adj_d.shape
print "---"
print "---"
print "Q: What is the shape of test_split_adj_d?"
print test_split_adj_d.shape
print "---"


## Reset index
train_split_adj_d.reset_index(inplace=True, drop=True)
test_split_adj_d.reset_index(inplace=True, drop=True)

print "---"
print "Q: How does train_split_adj_d look like?"
print train_split_adj_d.iloc[0]
print "---"
print "Q: How does test_split_adj_d look like?"
print test_split_adj_d.iloc[0]
print "---"






### Fill NaN of runtime train_split_adj_d

train_split_adj_d["runtime"]=train_split_adj_d["runtime"].fillna(np.mean(train_split_adj_d["runtime"]))

### Construct dataframes with sqrt transformation; Both train and test


 
## (b) Create a list of attribute labels, except for movie_index and revenue
train_split_cols=train_split_adj_d.columns.tolist()
train_split_cols.remove("movie_index")
test_split_cols=test_split_adj_d.columns.tolist()
test_split_cols.remove("movie_index")

print "---"
print "Q: How does train_split_drop_cols look like?"
print train_split_cols
print "---"


print "---"
print "Q: How does train_split_adj_d look like?"
print train_split_adj_d.iloc[0]
print "---"
## (b) Perform square root transformation of each attribute, for both train and test data

train_split_sqrt_adj_d=train_split_adj_d.copy(deep=True)
test_split_sqrt_adj_d=test_split_adj_d.copy(deep=True)
train_split_sqrt_adj_d[train_split_cols]=train_split_adj_d[train_split_cols].apply(np.sqrt)
test_split_sqrt_adj_d[test_split_cols]=test_split_adj_d[test_split_cols].apply(np.sqrt)

print "---"
print "Q: How does train_split_drop_sqrt_adj_d look like?"
print train_split_sqrt_adj_d.head(3)
print train_split_sqrt_adj_d.iloc[0]
print "---"
print "---"
print "Q: How does test_split_sqrt_adj_d look like?"
print test_split_sqrt_adj_d.head(3)
print test_split_sqrt_adj_d.iloc[0]
print "---"


### Visualize Correlation Matrix for Training Data (sqrt)

# (a) Visualize heatmap for correlation
# How does train_split_drop_sqrt_adj_d

fig, ax = plt.subplots()
train_split_sqrt_adj_nomi_d=train_split_sqrt_adj_d.drop("movie_index", axis=1)
sns.heatmap(train_split_sqrt_adj_nomi_d.corr(),
           xticklabels=[x.replace("_"," ").title() for x in train_split_sqrt_adj_nomi_d.columns],
           yticklabels=[x.replace("_"," ").title() for x in train_split_sqrt_adj_nomi_d.columns],
           cmap="Blues",
           annot=True,
           linecolor="black",
           linewidths=1.2,
           cbar_kws={'label':'Correlation'},
           ax=ax)

fig.subplots_adjust(right=0.7, left=-0.7, bottom=-0.7, top=0.7)
ax.set_title("Correlation Between Different Attributes")
ax.tick_params(left=False, bottom=False)
fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "Revenue is generally highly \n" \
                  "correlated with other attributes",
                  r"$\cdot$ " "Moderate to high correlation \n" \
                  "among revenue values for different \n" \
                  "crew job types - might lead to high \n" \
                  "collinearity in modelling",
                  r"$\cdot$ " "High correlation between budget \n" \
                  "and revenue values, as expected",
                  r"$\cdot$ " "Low correlation between revenue \n"\
                  "and runtime, which imply that runtime \n" \
                  "is a weak attribute to model revenue"))

fig.text(0.65, 0, com_res, bbox=dict(boxstyle="round",
                                     edgecolor="black",
                                     facecolor="wheat"),
        fontsize=12, family="serif")

