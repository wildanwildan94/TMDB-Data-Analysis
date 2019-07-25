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
### Quick analyze of the status attribute

## Present the form of status; And a distribution of count

status_d=train_d["status"].dropna().as_matrix()

# (a) Form of status
print status_d[10]
print list(set(status_d))

# (b) Consider the count of the possible values

count_status_d=Counter(status_d).items()

status_labels, status_count=zip(*count_status_d)
status_labels=list(status_labels)
status_count=list(status_count)

# (c) Comment on the status attribute
fig, ax = plt.subplots()

com_str="\n".join((r"$\cdot$ " "The only two possible status \n" \
                  "values are 'Released' and 'Rumored'",
                  r"$\cdot$ " "2996 movies have the status 'Released' \n" \
                  "and 4 movies have the status 'Rumored'", 
                  r"$\cdot$ " "Hence, it is an easily discared \n" \
                  "attribute"))
y_labels=range(2)

ax.barh(y_labels, status_count, edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(status_labels, fontsize=13)
ax.set_xlim((0, max(status_count)+400))
ax.set_facecolor("navajowhite")
ax.set_title("Distribution of Status Values")
fig.set_facecolor("floralwhite")

for x, y, val in zip(status_count, y_labels, status_count):
  ax.text(x+20, y, str(val), fontsize=12, weight="bold")

fig.text(0.92,0.3, com_str, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
       fontsize=14, family="serif")



