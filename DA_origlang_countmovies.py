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
### Continuation of Consideration of original_language

## Want to sum all original_language with one count into an 'etc' member

count_orig_lang=train_d.groupby(["original_language"]).size().reset_index(name="count")


amt_count_one_orig_lang=len(count_orig_lang.query("count==1"))

count_orig_lang_m=count_orig_lang.query("count>1").as_matrix()


count_orig_lang_m=np.vstack([count_orig_lang_m, ["other", amt_count_one_orig_lang]])

orig_lang_labs=count_orig_lang_m[:,0]
orig_lang_count_labs=[int(x) for x in count_orig_lang_m[:,1]]



# Sort List Depending on count_genre
orig_lang_labs=[x for _,x in sorted(zip(orig_lang_count_labs, orig_lang_labs))]
orig_lang_count_labs=[x for x,_ in sorted(zip(orig_lang_count_labs, orig_lang_labs))]


print orig_lang_labs
print orig_lang_count_labs


# (b) Visualizat in a Barplot the Distribution of Original Languages

fig, ax = plt.subplots()

# Define cutoff for 1x2 plot
cutoff=int(len(orig_lang_labs))-1

y_labels=range(cutoff)
ax.barh(y_labels, orig_lang_count_labs[0:cutoff], facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels[0:cutoff])
ax.set_yticklabels(orig_lang_labs[0:cutoff])
ax.set_xlabel("Count")
ax.set_facecolor("navajowhite")
fig.subplots_adjust(wspace=0.7, hspace=0.7, left=-0.5, right=0.5, top=1, bottom=0)
#fig.suptitle("Count of Genres; \n'en': %s"%orig_lang_count_labs[-1])

eng_str="Count of en: %s"%orig_lang_count_labs[-1]

fig.text(0, 0.5, eng_str, bbox=dict(boxstyle="round", facecolor="ivory", alpha=0.5), fontsize=15, family="serif")
fig.suptitle("Amount of Movies With Original Language", x=0.2, y=1.05)

textstr="\n".join((r"$\cdot$" "Other: Amount of spoken languagues \n with one related movie ",
                  r"$\cdot$" "Most movies are in english",
                  r"$\cdot$" "French movies are second by a wide margin,\n probably related to France having a \n rich, historic art culture" ))
fig.text(0.52, 0.5, textstr, bbox=dict(boxstyle="round", facecolor="ivory", alpha=0.5), fontsize=15, family="serif")
fig.set_facecolor("floralwhite")
