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

### Visualization of Cook's distance and cutoff lines

# (a) Construct and Compute Linear Regression Model

X_train_df=train_split_sqrt_adj_d.drop(["movie_index", "revenue"], axis=1)
y_train=train_split_sqrt_adj_d["revenue"].as_matrix()
X_train_cols=X_train_df.columns
X_train=X_train_df.as_matrix()

X_train_center_df=X_train_df-X_train_df.mean()
X_train_center=X_train_center_df.as_matrix()



X_train_center_ones=sm.add_constant(X_train_center)

model_sqrt_center=sm.OLS(y_train, X_train_center_ones)
res_sqrt_center=model_sqrt_center.fit()

# (b) Define Cook's Distance
model_cooks = np.array(res_sqrt_center.get_influence().cooks_distance[0])


# (c) Define Cook's Distance cutoff value: (1) 3*mean of Cook's Distance

threemean_cutoff=3*np.mean(model_cooks)


print "---"
print "Q: What is the value of 3*mean of Cook's Distance"
print threemean_cutoff
print "---"

# (d) Compute all indices where Cook's distance is above threemean_cutoff
cook_cutoff_threem_exc_id=np.argwhere(model_cooks>threemean_cutoff)[:,0]



print "---"
print "Q: How many points exceeds the 3*Mean of Cook's Distance cutoff?"
print str(len(cook_cutoff_threem_exc_id)) + " out of a total of %s points"%len(model_cooks)
print "---"

perc_above_cutoff=np.round(len(cook_cutoff_threem_exc_id)/float(len(model_cooks)), 2)*100
perc_below_cutoff=np.round((len(model_cooks)-len(cook_cutoff_threem_exc_id))/float(len(model_cooks)), 2)*100

# (e) Visualize the Cook's Distance of Points and the Three Mean Cutoff
# line


fig_cook, ax_cook=plt.subplots()
x_labels=np.array(range(model_cooks.shape[0]))

ax_cook.plot(x_labels, model_cooks, linestyle="None", marker="o",
             markerfacecolor="cornflowerblue", markeredgecolor="black")
ax_cook.plot(x_labels[cook_cutoff_threem_exc_id], model_cooks[cook_cutoff_threem_exc_id],
            linestyle="None", marker="o", markerfacecolor="crimson", markeredgecolor="black")
ax_cook.plot()
ax_cook.plot(x_labels, [threemean_cutoff]*model_cooks.shape[0], color="black", lw=2)
ax_cook.set_facecolor("navajowhite")
ax_cook.set_xlabel("Index")
ax_cook.set_ylabel("Cook's Distance")

cutoff_patch=mpatches.Patch(label="Cutoff Line Cook's Distance",
                           color="black")
above_cutoff_patch=mpatches.Patch(label="Points above cutoff line \n %s %%"%perc_above_cutoff, 
                                 color="crimson")
below_cutoff_patch=mpatches.Patch(label="Points below the cutoff line \n %s %%"%perc_below_cutoff,
                                 color="blue")

ax_cook.legend(handles=[cutoff_patch, above_cutoff_patch, below_cutoff_patch],
         bbox_to_anchor=(0.55, 0.6))
fig_cook.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "The 5 % of points above \n" \
                  "the cutoff line indicates extreme values \n" \
                  "with great influence on the linear model",
                 r"$\cdot$ " "One possible remedy is to remove \n" \
                 "these points from the linear model",
                 r"$\cdot$ " "The influential points are most \n" \
                 "likely big budget movies, which may distort \n" \
                 "the prediction of revenue of low-moderate \n" \
                 "revenue movies - which are probably \n" \
                 "more frequent",
                 r"$\cdot$ " "In case we want to take big budget \n" \
                 "movies into account in our model, we should \n" \
                 "leave the 5 % extreme values in the \n" \
                 "model",
                  r"$\cdot$ " "Because they are only 5 %, it makes \n" \
                  "sense to remove these extreme values"))

fig_cook.text(0.92,0.15, com_res,
             bbox=dict(boxstyle="round", edgecolor="black",
                      facecolor="wheat"))
fig_cook.suptitle("Cook's Distance of Fitted Values")








