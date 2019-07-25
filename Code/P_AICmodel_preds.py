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


### Compute statistics for the chosen optimal models

# (a) Define data for linear model

train_split_sqrt_red_adj_d=train_split_sqrt_adj_d.drop(cook_cutoff_threem_exc_id,
                                                       axis=0)
X_train_red_df=train_split_sqrt_red_adj_d.drop(["movie_index", "revenue"], axis=1)
y_train_red=train_split_sqrt_red_adj_d["revenue"].as_matrix()
X_train_cols=X_train_df.columns
X_train_red=X_train_red_df.as_matrix()

X_train_red_df_cols=X_train_red_df.columns.tolist()
print "---"
print "Q: What are the columns of X_train_red_df_cols?"
print X_train_red_df_cols
print "---"

X_train_red_center_df=X_train_red_df-X_train_red_df.mean()
X_train_red_center=X_train_red_center_df.as_matrix()



X_test_df=test_split_sqrt_adj_d.drop(["movie_index", "revenue"], axis=1)
X_test_center_df=X_test_df-X_train_red_df.mean()
X_test_center=X_test_center_df.as_matrix()
y_test=test_split_sqrt_adj_d["revenue"].as_matrix()


# (a) For the AIC optimal model

X_train_AIC=X_train_red_center_df[optmodel_AIC].as_matrix()
X_train_AIC_ones=sm.add_constant(X_train_AIC)
X_test_AIC=X_test_center_df[optmodel_AIC].as_matrix()
X_test_AIC_ones=sm.add_constant(X_test_AIC)


mod_AIC=sm.OLS(y_train_red, X_train_AIC_ones)
res_AIC=mod_AIC.fit()
rsq_adj_AIC=res_AIC.rsquared_adj
y_pred_test_AIC=res_AIC.predict(X_test_AIC_ones)
print "---"
print "Q: What is the R-squared adjusted for AIC model?"
print rsq_adj_AIC
print "---"


# (b) For the BIC optimal model

X_train_BIC=X_train_red_center_df[optmodel_BIC].as_matrix()
X_train_BIC_ones=sm.add_constant(X_train_BIC)
X_test_BIC=X_test_center_df[optmodel_BIC].as_matrix()
X_test_BIC_ones=sm.add_constant(X_test_BIC)

mod_BIC=sm.OLS(y_train_red, X_train_BIC_ones)
res_BIC=mod_BIC.fit()
rsq_adj_BIC=res_BIC.rsquared_adj
y_pred_test_BIC=res_BIC.predict(X_test_BIC_ones)

print "---"
print "Q: What is the R-squared adjusted for BIC model?"
print rsq_adj_BIC
print "---"

# (c) For the MSE model

X_train_MSE=X_train_red_center_df[optmodel_MSE].as_matrix()
X_train_MSE_ones=sm.add_constant(X_train_MSE)
X_test_MSE=X_test_center_df[optmodel_MSE].as_matrix()
X_test_MSE_ones=sm.add_constant(X_test_MSE)

mod_MSE=sm.OLS(y_train_red, X_train_MSE_ones)
res_MSE=mod_MSE.fit()
rsq_adj_MSE=res_MSE.rsquared_adj
y_pred_test_MSE=res_MSE.predict(X_test_MSE_ones)

print "---"
print "Q: What is the R-squared adjusted for MSE?"
print rsq_adj_MSE
print "---"


### Analysis of AIC model on validation data

# (a) Define data for linear model

train_split_sqrt_red_adj_d=train_split_sqrt_adj_d.drop(cook_cutoff_threem_exc_id,
                                                       axis=0)
X_train_red_df=train_split_sqrt_red_adj_d.drop(["movie_index", "revenue"], axis=1)
y_train_red=train_split_sqrt_red_adj_d["revenue"].as_matrix()
X_train_cols=X_train_df.columns
X_train_red=X_train_red_df.as_matrix()

X_train_red_df_cols=X_train_red_df.columns.tolist()
print "---"
print "Q: What are the columns of X_train_red_df_cols?"
print X_train_red_df_cols
print "---"

X_train_red_center_df=X_train_red_df-X_train_red_df.mean()
X_train_red_center=X_train_red_center_df.as_matrix()


val_adj_sqrt_d=val_adj_d.copy(deep=True).drop("movie_index", axis=1)
val_adj_sqrt_d=val_adj_sqrt_d.apply(np.sqrt)
print "---"
print "Q: What are the columns of val_adj_sqrt_d?"
print val_adj_sqrt_d.columns

X_val_adj_sqrt=(val_adj_sqrt_d.drop("revenue", axis=1)-X_train_red_df.mean())
y_val_adj_sqrt=val_adj_sqrt_d["revenue"].as_matrix()


# (a) For the AIC optimal model

X_train_AIC=X_train_red_center_df[optmodel_AIC].as_matrix()
X_train_AIC_ones=sm.add_constant(X_train_AIC)
X_val_AIC=X_val_adj_sqrt[optmodel_AIC].as_matrix()
X_val_AIC_ones=sm.add_constant(X_val_AIC)


mod_AIC=sm.OLS(y_train_red, X_train_AIC_ones)
res_AIC=mod_AIC.fit()
rsq_adj_AIC=res_AIC.rsquared_adj
y_pred_val_AIC=res_AIC.predict(X_val_AIC_ones)
print "---"
print "Q: What is the R-squared adjusted for AIC model?"
print rsq_adj_AIC
print "---"


# (b) Transform back to correct units
y_val_adj=y_val_adj_sqrt**2
y_pred_val=y_pred_val_AIC**2

# (c) Consider the values which have true value below 0.5e9,
# which corresponds to low/moderate values
index_moderate_rev=np.argwhere(y_val_adj<0.5e09)
y_val_moderev_adj=y_val_adj[index_moderate_rev]
y_pred_moderev_val=y_pred_val[index_moderate_rev]


# (d) Visualize the true vs. predicted, for both the untouched data
# and the moderate revenue values


fig, ax= plt.subplots(1,2)

ax[0].scatter(y_pred_val, y_val_adj, facecolor="cornflowerblue", edgecolor="black")
ax[1].scatter(y_pred_moderev_val, y_val_moderev_adj, facecolor="cornflowerblue",
             edgecolor="black")

ax[0].set_xlim((-0.05e9, 1.1e9))
ax[0].set_ylim((-0.05e9, 1.1e9))
ax[1].set_xlim((-0.2e8, 4.5e8))
ax[1].set_ylim((-0.2e8, 4.5e8))

fig.suptitle("True vs. Predicted Revenue for AIC Model", x=-0.1, y=1)
ax[0].set_title("For All Revenue Values")
ax[1].set_title("For Revenue Values with \n True Revenue <0.5e9")
fig.set_facecolor("floralwhite")

for index, axes in enumerate(fig.axes):
  axes.set_facecolor("navajowhite")
  axes.set_xlabel("Predicted")
  axes.set_ylabel("True")
  
fig.subplots_adjust(right=0.8, left=-0.8, wspace=0.1)

com_full_A="\n".join((r"$\cdot$ " "Visually, the predictions \n" \
                   "seems to be able to capture the \n" \
                   "true values in some cases well, but \n" \
                   "in some cases not so well",
                   r"$\cdot$ " "A problem is predicted values \n" \
                   "for true values close to zero, \n" \
                   "where it is not able to predict \n" \
                   "it well all the time"))
                   
com_full_B="\n".join((r"$\cdot$ " "However, it can be seen \n" \
                   "that the model performs pretty well \n"
                   "for moderate true revenue values, and \n" \
                   "in some cases where the revenue is low",
                   r"$\cdot$ " "The linear model doesn't seem \n" \
                   "to be able to capture movies with higher \n" \
                   "revenues, but the predicted value isn't \n" \
                   "too far away"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(-0.8, -0.35, com_full_A, bbox=box)
fig.text(0, -0.33, com_full_B, bbox=box)
