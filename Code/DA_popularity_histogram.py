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

### Visualize the distribution of popularity values

## Distribution in a histogram with statistical values

pop_d=train_d["popularity"].dropna().as_matrix()
print len(pop_d)

# (a) Compute Statistical Values; Mean, standard deviation and median

pop_mean=np.mean(pop_d)
pop_std=np.std(pop_d)
pop_median=np.median(pop_d)


# (b) Visualize Histogram for Distribution with Text of Statistical Properties

fig, ax = plt.subplots()

ax.hist(pop_d, facecolor="royalblue", bins="fd")
ax.set_facecolor("navajowhite")
fig.suptitle("Distribution of Popularity Values")
ax.set_xlabel("Popularity")
ax.set_ylabel("Count")

# Add text for statistics

text_stat="\n".join((r"$\cdot$" "Mean: %s"%np.round(pop_mean, 3),
                    r"$\cdot$" "Median: %s"%np.round(pop_median, 3),
                    r"$\cdot$" "Standard Deviation: %s"%np.round(pop_std, 3)))
textstr="\n".join((r"$\cdot$" "Most movies are moderately popular",
                  r"$\cdot$" "A few movies with huge popularity"))
fig.text(0.3,0.5, text_stat, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=12, family="serif")
fig.text(0.92, 0.5, textstr, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=12, family="serif")
fig.set_facecolor("floralwhite")

