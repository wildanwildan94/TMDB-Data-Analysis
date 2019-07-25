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

### Construct a Diagnostic Plot; Fitted vs. Residuals


# (a) Construct and Compute Linear Regression Model

X_train_df=train_split_sqrt_adj_d.drop(["movie_index", "revenue"], axis=1)
X_train_cols=X_train_df.columns
X_train=X_train_df.as_matrix()
y_train=train_split_sqrt_adj_d["revenue"].as_matrix()


X_train_center_df=X_train_df-X_train_df.mean()
X_train_center=X_train_center_df.as_matrix()



X_train_center_ones=sm.add_constant(X_train_center)



model_sqrt_center=sm.OLS(y_train, X_train_center_ones)
res_sqrt_center=model_sqrt_center.fit()

# (c) Define Various Model Fit Values
model_fitted_y=res_sqrt_center.fittedvalues
model_residuals=res_sqrt_center.resid
model_norm_residuals = res_sqrt_center.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
model_leverage = res_sqrt_center.get_influence().hat_matrix_diag



# (d) Visualize Diagnostic Plots

fig, ax = plt.subplots(2,2)



sns.residplot(model_fitted_y, y_train, ax= ax[0,0],
             scatter_kws={'facecolor':'royalblue',
                         'edgecolor':'black'})

ax[0,0].set_xlabel("Fitted Values")
ax[0,0].set_ylabel("Residuals")
ax[0,0].set_facecolor("navajowhite")
ax[0,0].set_title("(A) Fitted Values vs. Residuals of \n Linear Model")



stats.probplot(model_residuals, dist="norm", plot=ax[0,1])
ax[0,1].set_facecolor("navajowhite")
ax[0,1].set_title("(B) QQ-Plot of Residuals of Linear Model")


ax[1,0].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
           ax=ax[1,0])
ax[1,0].set_title("(C) Scale-Location of Linear Model")
ax[1,0].set_xlabel("Fitted Values")
ax[1,0].set_ylabel(r"$\sqrt{Standardized \ Residuals}$")
ax[1,0].set_facecolor("navajowhite")


ax[1,1].scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
           ax=ax[1,1])

ax[1,1].set_xlabel("Leverage")
ax[1,1].set_ylabel("Standardized Residuals")
ax[1,1].set_title("(D) Leverage vs. Standardised Residuals \n of Linear Model")
ax[1,1].set_xlim((0, 0.13))
ax[1,1].set_facecolor("navajowhite")


fig.set_facecolor("floralwhite")
fig.subplots_adjust(top=0.7, bottom=-0.7, left=-0.7, right=0.7, hspace=0.4, wspace=0.3)

com_fr="\n".join((r"$\cdot$ (A) " "Nonlinear residuals are \n" \
                 "directly apparent, except for fitted \n" \
                 "values below zero",
                 r"$\cdot$ (A) " "The strange form for fitted \n" \
                 "values around zero is, partly, due to revenue \n" \
                 "being non-negative, but the linear model \n" \
                 "may predict negative values \n" \
                 "A weakness with a linear model"))


com_qq="\n".join((r"$\cdot$ (B) " "A fairly good indication that \n" \
                "residuals may be normally distributed,",
                r"$\cdot$ (B)" "The extreme upper tails may \n" \
                "indicate that data has more extreme \n" \
                "values than what one would expect \n" \
                "from a normal distribution",
                r"$\cdot$ (B)" "Most likely, the extreme values \n" \
                "corresponds to movies with really high \n" \
                "revenue, like Lord of the Rings, which are \n" \
                "infrequent",
                r"$\cdot$ (B) " "A possible remedy would be to \n" \
                "discard these extreme values, but that would \n" \
                "reduce the model's efficiency in predicting \n" \
                "revenue for popular, blockbuster movies"))

com_scale=r"$\cdot$ (C) " "A fairly equal spread indicates \n" \
                    "that constant variance of residuals is reasonable"
com_lev="\n".join((r"$\cdot$ (D) " "Most values with \n" \
                  "high standardised residuals are \n"\
                  "not influential",
                  r"$\cdot$ (D) " "There exists a few high leverage \n" \
                  "points, but they have moderate standardised \n" \
                  "residuals, making they less influential \n" \
                  "on the fit of the linear model"))
box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.73, 0.4, com_fr, bbox=box)
fig.text(0.73, -0.27, com_qq, bbox=box)
fig.text(0.73, -0.4, com_scale, bbox=box)
fig.text(0.73, -0.75, com_lev, bbox=box)
fig.suptitle("Fitted Model; A square root transformation and centering of \n attributes around their means are applied", y=0.9, x=0.1)

