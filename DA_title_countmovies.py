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
### Consideration of the title attribute

## Distribution of the top ten words and then customly chosen words

title_d=train_d["title"].dropna().as_matrix()

# (a) Present the form
res=re.findall(r'\w+',title_d[10])
res=[x.lower() for x in res]
print res
# (b) Extract all words and lowercase the words

words_in_titles=[]

for item in title_d:
  new_item=re.findall(r'\w+', item)
  new_item=[x.lower() for x in new_item]
  words_in_titles.extend(new_item)
  
# (c) Count the occurence of each unique word

count_words_titles=Counter(words_in_titles)
print count_words_titles
print len(count_words_titles)
                      
# (d) Define important words and compute list of counts for these important words

import_words=["2", "man", "last", "love", "life", "night", "dead", "day", "house", "american"]

dict_count_words=dict(count_words_titles)

count_import_words=[]

for item in import_words:
  temp_count=dict_count_words[item]
  count_import_words.append(temp_count)
  
print import_words
print count_import_words

# Sort List Depending on count 
import_words=[x for _,x in sorted(zip(count_import_words, import_words))]
count_import_words=[x for x,_ in sorted(zip(count_import_words, import_words))]


# (e) Define and fill the ten most common words

top_words_name, top_words_count=zip(*count_words_titles.most_common(10))

top_words_name=[x for _,x in sorted(zip(top_words_count, top_words_name))]
top_words_count=[x for x,_ in sorted(zip(top_words_count, top_words_name))]


# (f) Visualize count of the ten most common words and ten important words

fig, ax = plt.subplots(2,1, sharex=True)

y_labels=range(len(top_words_name))

ax[0].barh(y_labels, top_words_count, facecolor="royalblue", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(top_words_name)
ax[0].set_facecolor("navajowhite")
ax[0].set_title("Ten most common words in titles")

ax[1].barh(y_labels, count_import_words, facecolor="royalblue", edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels(import_words)
ax[1].set_facecolor("navajowhite")
ax[1].set_xlabel("Count")
ax[1].set_title("Ten particular words in titles")

fig.subplots_adjust(top=0.5, bottom=-0.7, left=-0.5, right=0.5)
fig.set_facecolor("floralwhite")

com_com_text="\n".join((r"$\cdot$ " "A mix of common and \n" \
                       "interesting words in title",
                       r"$\cdot$ " "'The' is most common, by a wide margin - \n" \
                       "shows a popularity in a 'The'-structure of titles"))
com_man_text="\n".join((r"$\cdot$ " "Manually chosen words have \n" \
                       "moderately high counts - indicates that it \n" \
                       "may be possible, again, to utilize title words to \n" \
                       "to aggregate main characterstics of \n" \
                       "movies.",
                       r"$\cdot$ " "The usage of '2' indicates a \n" \
                       "lot of sequel movies",
                       r"$\cdot$ " "Love, life, day indicates movies  \n" \
                       "with positive, uplifting nature",
                       r"$\cdot$ " "Night, dead indicates a prevalence \n" \
                       "of movies with a sense of danger, scary \n" \
                       "in them" \
                       r"$\cdot$ " "A wide reach in different possible \n" \
                       "categories of movies in title"))

fig.text(0.53, 0.3, com_com_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), 
        family="serif", fontsize=13)
fig.text(0.53, -0.8, com_man_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        family="serif", fontsize=13)
