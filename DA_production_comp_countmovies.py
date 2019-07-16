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

### Visualize the most popular production companies in the dataset

## Want to consider the count of movies for the 20 production companies with 
## the most movies produced

prod_d=train_d["production_companies"].dropna().as_matrix()

# (a) Print the form
print prod_d[0]

# (b) Transform production into a list of dicts

prod_movies_dicts=[]
for item in train_d["production_companies"].dropna().as_matrix():
  #print type(item)
  #print "--"
  new_item=ast.literal_eval(item)
  prod_movies_dicts.extend(new_item)
  
  
# (c) Define and fill a list of all production companies for all movies

prod_movies_list=[item["name"] for item in prod_movies_dicts]
print prod_movies_list


# (d) Count the number of movies each production company is behind

count_prod_movies=Counter(prod_movies_list).most_common(20)
print count_prod_movies


prod_movies_comp, prod_movies_count=zip(*count_prod_movies)
prod_movies_comp=list(prod_movies_comp)
prod_movies_count=list(prod_movies_count)

# Sort list depending on count_genre
prod_movies_comp=[x for _,x in sorted(zip(prod_movies_count, prod_movies_comp))]
prod_movies_count=[x for x,_ in sorted(zip(prod_movies_count, prod_movies_comp))]


# (e) Visualize the number of movies the top 20 production companies have produced

fig, ax = plt.subplots()


cutoff=int(len(prod_movies_comp))
y_labels=range(cutoff)

ax.barh(y_labels, prod_movies_count, facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(prod_movies_comp, fontsize=11)
ax.set_facecolor("navajowhite")
fig.subplots_adjust(wspace=0.7, hspace=0.7, left=-0.45, right=0.35, top=0.4, bottom=-0.6)
title_str="Amount of Movies Produced by Most Popular \n Production Companies"
fig.suptitle(title_str, x=0, y=0.5, fontsize=13)
fig.set_facecolor("floralwhite")
ax.set_xlabel("Count")

com_str="\n".join((r"$\cdot$ " "Most popular production companies are American \n" \
"production companies, such as Warner Bros, \n Paramount Pictures, etc.",
                  r"$\cdot$ " "A couple of European production companies,\n like BBC Films, Canal+, etc."))
fig.text(0.38, 0.1, com_str, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), fontsize=13, family="serif")

