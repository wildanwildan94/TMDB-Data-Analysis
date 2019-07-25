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

### Compute linear model, and consider all diagnostic plots, for reduced data (Cook's Distance)

# (a) Computation of linear model
train_split_sqrt_red_adj_d=train_split_sqrt_adj_d.drop(cook_cutoff_threem_exc_id, axis=0)
X_train_red_df=train_split_sqrt_red_adj_d.drop(["movie_index", "revenue"], axis=1)
y_train_red=train_split_sqrt_red_adj_d["revenue"].as_matrix()
X_train_cols=X_train_df.columns
X_train_red=X_train_red_df.as_matrix()

X_train_red_center_df=X_train_red_df-X_train_red_df.mean()
X_train_red_center=X_train_red_center_df.as_matrix()



X_train_red_center_ones=sm.add_constant(X_train_red_center)

model_red=sm.OLS(y_train_red, X_train_red_center_ones)
model_fit=model_red.fit()
print "---"
print "Q: What is the linear regression summary?"
print model_fit.summary()
print "---"

model_fitted_y = model_fit.fittedvalues

# model residuals
model_residuals = model_fit.resid

# normalized residuals
model_norm_residuals = model_fit.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals
model_leverage = model_fit.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
model_cooks = model_fit.get_influence().cooks_distance[0]


# (b) Diagnostic Pplots

fig_diag, ax_diag=plt.subplots(2,2)
# Res vs. Leverage

sns.residplot(model_fitted_y, y_train_red, ax= ax_diag[0,0],
             scatter_kws={'facecolor':'royalblue',
                         'edgecolor':'black'})

ax_diag[0, 0].set_xlabel("Fitted Values")
ax_diag[0, 0].set_ylabel("Residuals")
ax_diag[0, 0].set_facecolor("navajowhite")
ax_diag[0, 0].set_title("(A) Fitted Values vs. Residuals of \n Linear Model")


# QQ-plot

stats.probplot(model_residuals, dist="norm", plot=ax_diag[0,1])
ax_diag[0, 1].set_facecolor("navajowhite")
ax_diag[0, 1].set_title("(B) QQ-Plot of Residuals of Linear Model")

# Scale-location 


ax_diag[1, 0].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
           ax=ax_diag[1,0])
ax_diag[1,0].set_title("(C) Scale-Location of Linear Model")
ax_diag[1,0].set_xlabel("Fitted Values")
ax_diag[1,0].set_ylabel(r"$\sqrt{standardized residuals}$")
ax_diag[1,0].set_facecolor("navajowhite")

# Standardised Residuals vs. leverage


ax_diag[1,1].scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
           ax=ax_diag[1,1])
ax_diag[1,1].set_xlim((0, max(model_leverage)+0.01))

ax_diag[1,1].set_xlabel("Leverage")
ax_diag[1,1].set_ylabel("Standardized Residuals")
ax_diag[1,1].set_title("(D) Leverage vs. Standardised Residuals of \n Linear Model")
ax_diag[1,1].set_facecolor("navajowhite")

#fig_diag.subplots_adjust(right=1,left=-1, top=1, bottom=-1)
fig_diag.subplots_adjust(top=0.7, bottom=-0.7, left=-0.7, right=0.7, hspace=0.4, wspace=0.3)

com_fr="\n".join((r"$\cdot$ (A) " "Nonlinear residuals is \n" \
                 "a reasonable assumption except, again, for \n" \
                 "fitted values below zero",
                 r"$\cdot$ (A) " "The strange form for fitted \n" \
                 "values around zero is, again, due to revenue \n" \
                 "being non-negative, but the linear model \n" \
                 "may predict negative values"))


com_qq="\n".join((r"$\cdot$ (B) " "An excellent indication that \n" \
                "residuals may be normally distributed,",
                r"$\cdot$ (B)" "The extreme upper tails, from  \n" \
                "the prior model, is remarkably gone - \n" \
                "most likely due to the points removed \n" \
                "from having relative high Cook's Distance",
                r"$\cdot$ (B) " "Note that the model might \n" \
                "now be less efficient in predicting revenue \n" \
                "for movies with high-revenue, e.g. Superhero \n" \
                "Movies, or sequels to famous movies"))

com_scale=r"$\cdot$ (C) " "A fairly equal spread indicates \n" \
                    "that constant variance of residuals is \n" \
                    "reasonable, albeit not perfect"
com_lev="\n".join((r"$\cdot$ (D) " "Most values with \n" \
                  "high standardised residuals are \n"\
                  "not influential",
                  r"$\cdot$ (D) " "There exists a few high leverage \n" \
                  "points, but they have low standardised \n" \
                  "residuals, making them less influential \n" \
                  "on the fit of the linear model"))
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig_diag.text(0.73, 0.4, com_fr, bbox=box)
fig_diag.text(0.73, -0.08, com_qq, bbox=box)
fig_diag.text(0.73, -0.25, com_scale, bbox=box)
fig_diag.text(0.73, -0.59, com_lev, bbox=box)
fig_diag.set_facecolor("floralwhite")
fig_diag.suptitle("Fitted Model; 5% of Data Points above the Cook's Distance \n cutoff is removed; Square root transformation and centering of attributes", y=0.95, x=0.1)


