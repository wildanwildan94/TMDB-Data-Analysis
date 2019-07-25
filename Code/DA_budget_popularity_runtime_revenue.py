# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import datetime
from collections import Counter
from google.colab import files
import matplotlib.patches as mpatches
plt.rcParams['figure.dpi'] = 100


import unicodedata

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)
    
    
## Load Data

train_d=pd.read_csv('tmdb_train.csv',
                   engine="python")

### Consideration of budget, popularity, runtime vs revenue

## Scatterplot of budget and revenue dependence

budget_rev_d=train_d[["budget", "revenue"]].dropna()
popul_revenue_d=train_d[["popularity", "revenue"]].dropna()
runtime_revenue_d=train_d[["runtime", "revenue"]].dropna()



popul=popul_revenue_d["popularity"].as_matrix()
revenue=popul_revenue_d["revenue"].as_matrix()




# (a) Do scatter plots of budget, popularity, runtime vs. revenue

fig, ax=plt.subplots(2,2)

budget_vals=np.array(budget_rev_d["budget"].values)
revenue_vals=np.array(budget_rev_d["revenue"].values)

profit_index=[i for i in range(len(budget_vals)) if budget_vals[i]<revenue_vals[i]]
loss_index=[i for i in range(len(budget_vals)) if budget_vals[i]>=revenue_vals[i]]
profit_perc=int(100*len(profit_index)/float(len(revenue_vals)))
loss_perc=int(100*len(loss_index)/float(len(revenue_vals)))


ax[0,0].scatter(budget_vals[profit_index], revenue_vals[profit_index],
            facecolor="royalblue", edgecolor="black")
ax[0,0].scatter(budget_vals[loss_index], revenue_vals[loss_index],
            facecolor="crimson", edgecolor="black")
ax[0,0].set_facecolor("navajowhite")
ax[0,0].set_title("Budget vs. Revenue")
ax[0,0].set_xlabel("Budget")
ax[0,0].set_ylabel("Revenue")
x_line=np.linspace(0, 4e08, 100)
y_line=np.linspace(0, 4e08, 100)
ax[0,0].plot(x_line, y_line, color="forestgreen")
ax[0,0].set_xlim((-0.1e08, 0.5e09))
ax[0,0].set_ylim((-0.5e08, 1.7e09))
profit_patch=mpatches.Patch(color="royalblue", label="Profit: %s %%"%profit_perc)
loss_patch=mpatches.Patch(color="crimson", label="Loss: %s %%"%loss_perc)
line_patch=mpatches.Patch(color="forestgreen", label="y=x")
ax[0,0].legend(handles=[profit_patch, loss_patch, line_patch],
              fancybox=True, facecolor="thistle", edgecolor="black",
              fontsize=10)

ax[0,1].scatter(np.sqrt(popul), np.sqrt(revenue), facecolor="cornflowerblue", edgecolor="black")
ax[0,1].set_xlabel("Popularity")
ax[0,1].set_ylabel("Revenue")
ax[0,1].set_title("(Square Root) Popularity vs. Revenue")
ax[0,1].set_facecolor("navajowhite")

ax[1,0].scatter(runtime_revenue_d["runtime"], runtime_revenue_d["revenue"], facecolor="cornflowerblue",
                edgecolor="black")
ax[1,0].set_facecolor("navajowhite")
ax[1,0].set_xlabel("Runtime")
ax[1,0].set_ylabel("Revenue")
ax[1,0].set_title("Runtime vs. Revenue")


fig.subplots_adjust(right=0.8, left=-0.8, top=0.8, bottom=-0.8, hspace=0.3)
fig.set_facecolor("floralwhite")
ax[1,1].axis("off")

com="\n".join((r"$\cdot$ " "More than 3/4 of movies are \n" \
              "profitable - but there exists just \n" \
              "below a quarter of movies that \n" \
              "are at a loss",
              r"$\cdot$ " "Higher revenue seems to be associated \n" \
              "with higher popularity, especially for \n" \
              "moderate and higher values of revenue",
              r"$\cdot$ " "No particular dependence \n" \
              "between runtime and revenue. Just a \n" \
              "more diverse range of revenue values \n" \
              "for movies with just below 2hrs of runtime"))

fig.text(0.1, -0.75, com, bbox=dict(boxstyle="round", edgecolor="black",
                                 facecolor="wheat"),
        fontsize=14, family ="serif")

