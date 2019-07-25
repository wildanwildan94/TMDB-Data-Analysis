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
                   
 ### Visualization of the distribution of budget values

## (1) Histogram of Budget Values; To Visualize the Distribution of Values

fig, ax = plt.subplots()


ax.hist(train_d["budget"], bins="fd", edgecolor="black", facecolor="royalblue")
ax.set_xlabel("Budget")
ax.set_ylabel("Count")
ax.set_facecolor("navajowhite")
fig.suptitle("Histogram of Budget Values")
fig.set_facecolor("floralwhite")
textstr="\n".join((r"$\cdot$ Most movies are low-budget",
                  r"$\cdot$ Budget decreases exponentially",
                  r"$\cdot$ A few high-budget movies -" "\n" "mosty likely blockbuster movies"))

props=dict(boxstyle="round", facecolor="wheat", alpha=0.5)

fig.text(0.93, 0.5, textstr, bbox=props, fontsize=14, family="serif")
