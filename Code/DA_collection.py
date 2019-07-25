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

### Particular Consideration of the belongs_to_collection attribute

## Print out the Counts of Different 'name' in dict

# (a) Insert all non-nan name into a list

list_of_btc=[ast.literal_eval(item)[0]["name"] for item in train_d["belongs_to_collection"].dropna()]

# (b) Count the Occurence of Each Unique String

unq_name_btc=Counter(list_of_btc)

# (c) Print out the Number of Collections and the top n collections

n=10
nmbr_unq_name_btc=len(unq_name_btc)
print "Amount of Unique Collections: %s"%nmbr_unq_name_btc
top_n_btc=unq_name_btc.most_common(n)
print "Top %s Collections in Amount "%n
for item in top_n_btc:
  print "Collection: "
  print item[0]
  print "Count of in Collection: "
  print item[1]

### Visualize how many movies belongs to a certain collection


amt_mbs_collection=[item[1] for item in unq_name_btc.items()]
amt_mbs_collection_gone=[item[1] for item in unq_name_btc.items() if item[1]>1]


amt_members, count_amt_members=zip(*Counter(amt_mbs_collection).items())
amt_members_gone, count_amt_members_gone=zip(*Counter(amt_mbs_collection_gone).items())



fig, ax = plt.subplots(2,1)

ax[0].bar(amt_members, count_amt_members, facecolor="royalblue", edgecolor="black")
ax[1].bar(amt_members_gone, count_amt_members_gone, facecolor="royalblue", edgecolor="black")

ax[0].set_facecolor("navajowhite")
ax[1].set_facecolor("navajowhite")


ax[1].set_xlabel("Members in Collection")
ax[0].set_ylabel("Count")
ax[1].set_ylabel("Count")
ax[0].set_title("At Least One Movie in Collection")
ax[1].set_title("At least Two Movies in Collection")

fig.suptitle("Amount of Movies Belonging to a Collection")

props=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
textstr="\n".join((r"$\cdot$ Most collections have one movie",
                  r"$\cdot$" "Moderate amount of collections with \n" \
                   " 2-4 movies",
                  r"$\cdot$ One collection with 16 movies (James Bond)",
                  r"$\cdot$ Makes sense - rarely more than" "\n three movies in a collection" "\n" "e.g. sequel, trilogy"))

fig.text(0.95, 0.5, textstr, fontsize=13, bbox=props, family="serif")
fig.subplots_adjust(hspace=0.5)
fig.set_facecolor("floralwhite")


