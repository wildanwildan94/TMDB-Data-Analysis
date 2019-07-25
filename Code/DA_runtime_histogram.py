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
### Visualize the distribution of the runtime attribute

## Presentation of the form of runtime and a distribution plot

runtime_d=train_d["runtime"].dropna().as_matrix()

# (a) Presentation of the form of runtime
print runtime_d[0]


# (b) Define runtime statistics

runtime_mean=np.round(np.mean(runtime_d), 3)
runtime_std=np.round(np.std(runtime_d), 3)
runtime_median=np.round(np.median(runtime_d), 3)


# (c) Visualize the distribution of the runtime attribute

fig, ax = plt.subplots()


ax.hist(runtime_d, facecolor="cornflowerblue", bins="fd", alpha=0.8, edgecolor="black")
ax.set_facecolor("navajowhite")
ax.set_xlabel("Runtime (Mins)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Runtime")


stat_text="\n".join((r"$\cdot$ " "Mean: %s"%runtime_mean,
                    r"$\cdot$ " "Standard Deviation: %s"%runtime_std,
                    r"$\cdot$ " "Median: %s"%runtime_median))

fig.text(0.45,0.6, stat_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), family="serif", fontsize=12)
fig.set_facecolor("floralwhite")

com_text="\n".join((r"$\cdot$ " "Most movies have a runtime just below 2 hours",
                   r"$\cdot$ " "Note that there is a moderate proportion of \n" \
                   "movies longer than 2 hours. "))

fig.text(0.93, 0.6, com_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), family="serif", fontsize=12)
