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

### Analyze of the tagline attribute


## Present the form; Extract all words in all taglines; And show the ten most common words

tagline_d=train_d["tagline"].dropna().as_matrix()

# (a) Present the form
res=re.findall(r'\w+',tagline_d[10])
res=[x.lower() for x in res]
print res

# (b) Extract all words and lowercase the words

words_in_taglines=[]

for item in tagline_d:
  new_item=re.findall(r'\w+', item)
  new_item=[x.lower() for x in new_item]
  words_in_taglines.extend(new_item)
  
# (c) Count the occurence of each unique word

count_words_taglines=Counter(words_in_taglines)
print count_words_taglines
print len(count_words_taglines)
                      
  
# (d) Define ten popular, manually chosen words to analyze

import_words=["you", "in", "one", "he", "no", "can", "love", "life", "world", "story"]

dict_count_words=dict(count_words_taglines)

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

top_words_name, top_words_count=zip(*count_words_taglines.most_common(10))

top_words_name=[x for _,x in sorted(zip(top_words_count, top_words_name))]
top_words_count=[x for x,_ in sorted(zip(top_words_count, top_words_name))]


# (f) Visualize the ten most common words and ten uncommon, common words


fig, ax = plt.subplots(2,1, sharex=True)

y_labels=range(len(top_words_name))

ax[0].barh(y_labels, top_words_count, facecolor="royalblue", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(top_words_name, fontsize=13)
ax[0].set_facecolor("navajowhite")
ax[0].set_title("Ten most common words in taglines")

ax[1].barh(y_labels, count_import_words, facecolor="royalblue", edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels(import_words, fontsize=13)
ax[1].set_facecolor("navajowhite")
ax[1].set_xlabel("Count")
ax[1].set_title("Ten manually chosen words in taglines")


fig.subplots_adjust(top=0.5, bottom=-0.7, left=-0.5, right=0.5)
fig.set_facecolor("floralwhite")

com_com_text="\n".join((r"$\cdot$ " "Most common words are \n" \
                       "uninteresting, sentence-building words",
                       r"$\cdot$ " "Not indicative of any prevalent \n" \
                       "characteristics of movies"))
com_man_text="\n".join((r"$\cdot$ " "Manually chosen words have \n" \
                       "moderately high counts - indicates that it \n " \
                       "may be possible to utilize tagline words to \n " \
                       "to aggregate main characterstics of \n" \
                       "movies.",
                       r"$\cdot$ " "Love, life indicates a lot of movies \n " \
                       "associated with a positive message",
                       r"$\cdot$ " "Word, story shows the adventurous \n" \
                       "associated movies are prevalent"))

fig.text(0.53, 0.2, com_com_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), 
        family="serif", fontsize=13)
fig.text(0.53, -0.65, com_man_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        family="serif", fontsize=13)

