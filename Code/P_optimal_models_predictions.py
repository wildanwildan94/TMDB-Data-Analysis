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


### Prediction on Validation of all Models Considered
### and presentation of statistics

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



# (b) For the BIC optimal model

X_train_BIC=X_train_red_center_df[optmodel_BIC].as_matrix()
X_train_BIC_ones=sm.add_constant(X_train_BIC)
X_val_BIC=X_val_adj_sqrt[optmodel_BIC].as_matrix()
X_val_BIC_ones=sm.add_constant(X_val_BIC)

mod_BIC=sm.OLS(y_train_red, X_train_BIC_ones)
res_BIC=mod_BIC.fit()
rsq_adj_BIC=res_BIC.rsquared_adj
y_pred_val_BIC=res_BIC.predict(X_val_BIC_ones)

print "---"
print "Q: What is the R-squared adjusted for BIC model?"
print rsq_adj_BIC
print "---"

# (c) For the MSE model

X_train_MSE=X_train_red_center_df[optmodel_MSE].as_matrix()
X_train_MSE_ones=sm.add_constant(X_train_MSE)
X_val_MSE=X_val_adj_sqrt[optmodel_MSE].as_matrix()
X_val_MSE_ones=sm.add_constant(X_val_MSE)

mod_MSE=sm.OLS(y_train_red, X_train_MSE_ones)
res_MSE=mod_MSE.fit()
rsq_adj_MSE=res_MSE.rsquared_adj
y_pred_val_MSE=res_MSE.predict(X_val_MSE_ones)

print "---"
print "Q: What is the R-squared adjusted for MSE?"
print rsq_adj_MSE
print "---"


fig_pred, ax_pred = plt.subplots(2,2)

x_lin=np.linspace(0, 1.6e09, 100)
y_lin=np.linspace(0, 1.6e09, 100)

# (a) Prediction of test set for the AIC model

ax_pred[0,0].scatter(y_pred_val_AIC**2, y_val_adj_sqrt**2, facecolor="royalblue", edgecolor="black")

# (b) Prediction of test set for the BIC model
ax_pred[0,1].scatter(y_pred_val_BIC**2, y_val_adj_sqrt**2, facecolor="royalblue", edgecolor="black")

# (c) Prediction of test set for the MSE model
ax_pred[1,0].scatter(y_pred_val_MSE**2, y_val_adj_sqrt**2, facecolor="royalblue", edgecolor="black")





fig_pred.suptitle("Predicted and True Revenue for the Different Models; Transformation Undone", x=0.1, y=1.3)
#fig_pred.subplots_adjust(left=-1.2, right=1.2, top=1.2, bottom=-1.2)
ax_pred[0,0].set_title("Optimal AIC Model with 10 Attributes (+const.)",y=1.1)
ax_pred[0,1].set_title("Optimal BIC Model with 6 Attributes (+const.)", y=1.1)
ax_pred[1,0].set_title("Optimal MSE Model with 4 Attributes (+const.)", y=1.1)

red_patch=mpatches.Patch(color="red", label="y=x")
fig_pred.legend(handles=[red_patch], bbox_to_anchor=(0.58, 1.2))
for i, ax in enumerate(fig_pred.axes):
  if i<3:
    ax.axis('equal')
    ax.plot(x_lin, y_lin, color="red")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_facecolor("navajowhite")
ax_pred[1,1].axis("off")
fig_pred.subplots_adjust(top=1.1, bottom=-0.2, left=-0.7, right=0.7, hspace=0.45, wspace=0.3)
fig_pred.set_facecolor("floralwhite")


com_AIC_attr="\n".join((r"$\cdot$ (AIC), $R^2_{adjusted}=$%s"%np.round(rsq_adj_AIC, 2),
                       "  " "Budget",
                       "  " "Collection Revenue Mean",
                       "  " "Original Language -",
                       "  " "Production Company -",
                       "  " "Keyword -",
                       "  " "Cast -",
                       "  " "Execute Producer -",
                       "  " "Director -",
                       "  " "Screenplay -",
                       "  " "Writer -"))
com_BIC_attr="\n".join((r"$\cdot$ (BIC), $R^2_{adjusted}=$%s"%np.round(rsq_adj_BIC, 2),
                       "  " "Original Language Revenue Mean",
                       "  " "Production Company -",
                       "  " "Keyword - ",
                       "  " "Cast -",
                       "  " "Producer -",
                       "  " "Director -"))
com_MSE_attr="\n".join((r"$\cdot$ (MSE), $R^2_{adjusted}=$%s"%np.round(rsq_adj_MSE, 2),
                       "  " "Keyword Revenue Mean",
                       "  " "Cast -",
                       "  " "Producer -",
                       "  " "Director -"))
com_models="\n".join((r"$\cdot$ " " All models have a difficulty \n" \
                     "in predicting revenue for high \n" \
                     "revenue movies, as expected from \n" \
                     "the construction of our linear model \n" \
                     "(recall the training points removed due to \n" \
                     "high Cook's distance values)",
                     r"$\cdot$ " "The models seems to have similar \n" \
                     "pattern, in predicted vs. true values",
                     r"$\cdot$ " "The adjusted " r"$R^2_{adjusted}$ " "\n" \
                     "indicates that the model with 10 attributes \n" \
                     "(+const.) seems to perform the best, which \n" \
                     "can be expected as it has the most \n" \
                     "attributes",
                     r"$\cdot$ " "High revenue movies seems to appear \n" \
                     "for true values above 0.5e9"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig_pred.text(-0.05, -0.05, com_AIC_attr, bbox=box)
fig_pred.text(0.28,  0.12, com_BIC_attr, bbox=box)
fig_pred.text(0.28, -0.15, com_MSE_attr, bbox=box)

fig_pred.text(0.72, 0.46, com_models, bbox=box)
