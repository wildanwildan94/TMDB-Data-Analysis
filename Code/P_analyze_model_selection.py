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

### Analyze results from measures


results_measures_df=pd.read_csv('results_measures.csv')

print "---"
print "Q: How does results_measures_df look like?"
print results_measures_df.iloc[0]
print "---"


# (a) Visualize AIC; BIC and MSE 

fig_meas, ax_meas=plt.subplots(3,2, sharex=True)

AIC_train_lower=[]
AIC_test_lower=[]
BIC_train_lower=[]
BIC_test_lower=[]
MSEtrain_lower=[]
MSEtest_lower=[]
subsetsize_lower=[]
for a, b in results_measures_df.groupby("subset_size"):
  n=b.shape[0]
  ax_meas[0, 0].scatter([int(a)+1]*n, b["AIC_train"], color="cornflowerblue")
  ax_meas[0,1].scatter([int(a)+1]*n, b["AIC_test"], color="indianred")
  AIC_train_min=min(b["AIC_train"])
  AIC_test_min=min(b["AIC_test"])
  AIC_train_lower.append(AIC_train_min)
  AIC_test_lower.append(AIC_test_min)

  
  ax_meas[1,0].scatter([int(a)+1]*n, b["BIC_train"], color="cornflowerblue")
  ax_meas[1,1].scatter([int(a)+1]*n, b["BIC_test"], color="indianred")
  ax_meas[1,0].set_ylabel("BIC")
  BIC_train_min=min(b["BIC_train"])
  BIC_test_min=min(b["BIC_test"])
  BIC_train_lower.append(BIC_train_min)
  BIC_test_lower.append(BIC_test_min)
  
  ax_meas[2,0].scatter([int(a)+1]*n, b["trainMSE"], color="cornflowerblue")
  ax_meas[2,1].scatter([int(a)+1]*n, b["testMSE"], color="indianred")
  MSEtrain_min=min(b["trainMSE"])
  MSEtest_min=min(b["testMSE"])
  MSEtrain_lower.append(MSEtrain_min)
  MSEtest_lower.append(MSEtest_min)
  
  subsetsize_lower.append(int(a))
  
  
subsetsize_lower=[x+1 for x in subsetsize_lower]
ax_meas[0,0].plot(subsetsize_lower, AIC_train_lower, color="black", lw=4)
ax_meas[0,1].plot(subsetsize_lower, AIC_test_lower, color="black", lw=4)
ax_meas[1,0].plot(subsetsize_lower, BIC_train_lower, color="black", lw=4)
ax_meas[1,1].plot(subsetsize_lower, BIC_test_lower, color="black", lw=4)
ax_meas[2,0].plot(subsetsize_lower, MSEtrain_lower, color="black", lw=4)
ax_meas[2,1].plot(subsetsize_lower, MSEtest_lower, color="black", lw=4)

for index, axes in enumerate(fig_meas.axes):
  axes.set_xticks(subsetsize_lower)



ax_meas[0, 0].set_ylabel("AIC")
ax_meas[2,0].set_ylabel("MSE")
ax_meas[2,0].set_xlabel("Subset Size")
ax_meas[2,1].set_xlabel("Subset Size")

ax_meas[0,0].set_title("AIC for Training Data")
ax_meas[0,1].set_title("AIC for Testing Data")
ax_meas[1,0].set_title("BIC for Training Data")
ax_meas[1,1].set_title("BIC for Testing Data")
ax_meas[2,0].set_title("MSE for Training Data")
ax_meas[2,1].set_title("MSE for Testing Data")

ax_meas[0,0].set_facecolor("navajowhite")
ax_meas[0,1].set_facecolor("navajowhite")
ax_meas[1,0].set_facecolor("navajowhite")
ax_meas[1,1].set_facecolor("navajowhite")
ax_meas[2,0].set_facecolor("navajowhite")
ax_meas[2,1].set_facecolor("navajowhite")

fig_meas.set_facecolor("floralwhite")


  
  
fig_meas.subplots_adjust(left=-0.6, right=0.6, top=0.6, bottom=-0.6, hspace=0.3)

com_AIC="\n".join((r"$\cdot$ (AIC) " "The AIC for the testing data \n" \
                  "seems to reach around 10 attributes (+const) \n" \
                  "before becoming stagnant",
                  r"$\cdot$ " "Note how this imply a rather \n" \
                  "large model, but significantly less than \n" \
                  "the total amount of attributes, 17 (+const)"))
com_BIC="\n".join((r"$\cdot$ (BIC) " "The BIC for the testing data \n" \
                  "seems to have a minimum at 8 attributes \n" \
                  "(+const) before increasing", 
                 r"$\cdot$ " "Hence, the BIC suggests a model with \n" \
                  "moderate amount of attributes, a bit less \n" \
                  "than the amount of attributes that the \n" \
                  "AIC model suggests"))
com_MSE="\n".join((r"$\cdot$ (MSE) " "The MSE of the testing data \n" \
                  "seems to drop off rather fast, and \n" \
                  "becoming stagnant around 4 attributes (+const)",
                  r"$\cdot$ " "This suggests a lower number of attributes, \n" \
                  "in constrast to all attributes, and the \n" \
                  "amount of attributes as suggested by \n" \
                  "AIC and BIC",
                  r"$\cdot$ " "Hence, from the performance measures \n" \
                  "AIC, BIC and MSE, we have three possible, well \n" \
                  "performing models with varying amount \n" \
                  "of number of attributes"))

com_alg="\n".join((r"$\cdot$ " "For each subset of attributes",
                  "   Remove 5% of points as determined by Cook's \n" \
                  "   cutoff, apply square root transformation, center \n" \
                  "   attributes around their means",
                  "   Fit a linear model to data",
                  "   Evaluate AIC, BIC and MSE for the data \n" \
                  "   and another set of data, called the test data, \n" \
                  "   (which is independent of the training data)"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig_meas.text(0.62, 0.38, com_AIC, bbox=box)
fig_meas.text(0.62, 0.02, com_BIC, bbox=box)
fig_meas.text(0.62, -0.52, com_MSE, bbox=box)
fig_meas.text(0, -1.03, com_alg, bbox=box)
