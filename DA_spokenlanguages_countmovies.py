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

### Want to consider the spoken_language attribute


spok_lang_d=train_d["spoken_languages"].dropna().as_matrix()

# (a) Form of spoken_language

print spok_lang_d[3]


# (b) Define and fill a list of the dicts


spok_lang_dicts=[]
for item in train_d["spoken_languages"].dropna().as_matrix():
  new_item=ast.literal_eval(item)
  spok_lang_dicts.extend(new_item)
  
print spok_lang_dicts

# (c) Define a list of all short names for spoken languages
shortn_sl_list=[item["iso_639_1"] for item in spok_lang_dicts]


# (d) Count the occurence of each shortname and long name of spoken languages

shortn_sl_count=Counter(shortn_sl_list).items()
shortn_sl_name, shortn_sl_count=zip(*shortn_sl_count)

# Sort List Depending on count 
shortn_sl_name=[x for _,x in sorted(zip(shortn_sl_count, shortn_sl_name))]
shortn_sl_count=[x for x,_ in sorted(zip(shortn_sl_count, shortn_sl_name))]


# (e) Create New shortn and counts, with all spoken_languages with 1 aggregated together

index_one=[i for i in range(len(shortn_sl_count)) if shortn_sl_count[i]==1]
last_elem_index_one=index_one[-1]
count_index_one=len(index_one)

shortn_sl_name=shortn_sl_name[last_elem_index_one+1:]
shortn_sl_count=shortn_sl_count[last_elem_index_one+1:]
shortn_sl_name.append("other")
shortn_sl_count.append(count_index_one)

shortn_sl_name=[x for _,x in sorted(zip(shortn_sl_count, shortn_sl_name))]
shortn_sl_count=[x for x,_ in sorted(zip(shortn_sl_count, shortn_sl_name))]



# (f) Consider the ten most common spoken languages

shortn_sl_count=Counter(shortn_sl_list).most_common(10)
shortn_sl_name, shortn_sl_count=zip(*shortn_sl_count)

# Sort List Depending on count 
shortn_sl_name=[x for _,x in sorted(zip(shortn_sl_count, shortn_sl_name))]
shortn_sl_count=[x for x,_ in sorted(zip(shortn_sl_count, shortn_sl_name))]


# (g) Visualize the count of each spoken language

fig, ax = plt.subplots()

y_labels=range(len(shortn_sl_name))
ax.barh(y_labels, shortn_sl_count, facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(shortn_sl_name, fontsize=12)
ax.set_xlim((0, 3000))

ax.set_facecolor("navajowhite")
ax.set_title("Amount of Movies with a Certain Spoken Language")
ax.set_xlabel("Count")

for i, v in enumerate(shortn_sl_count):
  ax.text(v+50, i-0.1, str(v), color="black", fontweight="bold")
  
  
fig.set_facecolor("floralwhite")
com_str="\n".join((r"$\cdot$ " "As with original language, \n" \
                  "most movies have English as a spoken language",
                  r"$\cdot$ " "French, Spanish, German as top \n" \
                  "European languages makes sense, as they are \n" \
                  "major European nations"))

fig.text(0.92,0.5, com_str, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), fontsize=13, family="serif")
