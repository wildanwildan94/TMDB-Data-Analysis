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


### Compute Model Performance Measures for Each Subset of Attributes
### For MSE, AIC and BIC

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

# (a) Define data for linear model
# Subtract test set by the mean of train set, to preserve 
# attribute transformations

train_split_sqrt_red_adj_d=train_split_sqrt_adj_d.drop(cook_cutoff_threem_exc_id, axis=0)
X_train_red_df=train_split_sqrt_red_adj_d.drop(["movie_index", "revenue"], axis=1)
y_train_red=train_split_sqrt_red_adj_d["revenue"].as_matrix()
X_train_cols=X_train_df.columns
X_train_red=X_train_red_df.as_matrix()

X_train_red_df_cols=X_train_red_df.columns.tolist()


X_train_red_center_df=X_train_red_df-X_train_red_df.mean()
X_train_red_center=X_train_red_center_df.as_matrix()


X_test_df=test_split_sqrt_adj_d.drop(["movie_index", "revenue"], axis=1)
X_test_center_df=X_test_df-X_train_red_df.mean()
X_test_center=X_test_center_df.as_matrix()
y_test=test_split_sqrt_adj_d["revenue"].as_matrix()


n_train=X_train_red_center_df.shape[0]
n_test=X_test_center_df.shape[0]

# (b) For each subset of attributes, construct a model, with respect to
# the attributes, train on the training data and compute the 
# MSE, AIC and BIC on the training and test data, based on the trained model
# store the results as results_measures

AIC_train_array=[]
AIC_test_array=[]
BIC_train_array=[]
BIC_test_array=[]
testMSE=[]
trainMSE=[]
subset_array=[]
subset_size_array=[]

subsets=[x for x in powerset(X_train_red_df_cols)]
print subsets
subsets_n=len(subsets)
for i in range(subsets_n):
  subset_cols=subsets[i]
  subset_cols_list=list(subset_cols)
  p_m=len(subset_cols)
  X_train_subset=X_train_red_center_df[subset_cols_list].as_matrix()
  X_train_subset_ones=sm.add_constant(X_train_subset)
  X_test_subset=X_test_center_df[subset_cols_list].as_matrix()
  X_test_subset_ones=sm.add_constant(X_test_subset)
    
  model=sm.OLS(y_train_red, X_train_subset_ones)
  results=model.fit()
    
  y_pred_train=np.array(results.predict(X_train_subset_ones))
  y_pred_test=np.array(results.predict(X_test_subset_ones))
    
  RSS_train=np.sum((y_pred_train-y_train_red)**2)
  RSS_test=np.sum((y_pred_test-y_test)**2)
    
  tempMSE_train=RSS_train/float(n_train)
  tempMSE_test=RSS_test/float(n_test)
     
  AIC_train_temp=n_train*np.log(RSS_train)+2*p_m
  AIC_test_temp=n_test*np.log(RSS_train)+2*p_m
    
  BIC_train_temp=n_train*np.log(RSS_train)+p_m*np.log(n_train)
  BIC_test_temp=n_test*np.log(RSS_test)+p_m*np.log(n_test)
    
  AIC_train_array.append(AIC_train_temp)
  AIC_test_array.append(AIC_test_temp)
    
  BIC_train_array.append(BIC_train_temp)
  BIC_test_array.append(BIC_test_temp)
    
  trainMSE.append(tempMSE_train)
  testMSE.append(tempMSE_test)
  subset_array.append(subset_cols)
  subset_size_array.append(p_m)

    
results_measures=pd.DataFrame({'AIC_train':AIC_train_array,
                              'AIC_test':AIC_test_array,
                              'BIC_train':BIC_train_array,
                              'BIC_test':BIC_test_array,
                              'trainMSE':trainMSE,
                              'testMSE':testMSE,
                              'subset_cols':subset_array,
                              'subset_size':subset_size_array})

results_measures.to_csv('results_measures.csv', index=False)
    
    
    
    
    
    
    
    
