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
### Consideration of the keywords attribute

## Want to consider the distribution of counts of top ten common words and ten customly chosen words

keywords_d=train_d["Keywords"].dropna().as_matrix()
print title_d[0]

# (a) Present the form
res=re.findall(r'\w+',keywords_d[10])
res=[x.lower() for x in res]
print res

# (b) Define and fill a list of the dicts


keywords_dicts=[]
for item in keywords_d:
  new_item=ast.literal_eval(item)
  keywords_dicts.extend(new_item)
  
  

# (b) Extract all words and lowercase the words

words_in_keywords=[]

for item in keywords_dicts:
  new_item=re.findall(r'\w+', item["name"])
  new_item=[x.lower() for x in new_item]
  words_in_keywords.extend(new_item)
  
# (c) Count the occurence of each unique word

count_words_keywords=Counter(words_in_keywords)
print count_words_keywords
print len(count_words_titles)


import_words=["relationship", "woman", "love", "murder", "war", "police", "death", "drug", "novel", "family"]

dict_count_words=dict(count_words_keywords)

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

top_words_name, top_words_count=zip(*count_words_keywords.most_common(10))

top_words_name=[x for _,x in sorted(zip(top_words_count, top_words_name))]
top_words_count=[x for x,_ in sorted(zip(top_words_count, top_words_name))]



# (f) Visulaize count of the ten most common words and ten important words

fig, ax = plt.subplots(2,1, sharex=True)

y_labels=range(len(top_words_name))

ax[0].barh(y_labels, top_words_count, facecolor="royalblue", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(top_words_name)
ax[0].set_facecolor("navajowhite")
ax[0].set_ylabel("Words")
ax[0].set_title("Ten most common words in keywords")

ax[1].barh(y_labels, count_import_words, facecolor="royalblue", edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels(import_words)
ax[1].set_facecolor("navajowhite")
ax[1].set_xlabel("Count")
ax[1].set_title("Ten particular words in keywords")


fig.subplots_adjust(top=0.5, bottom=-0.8, left=-0.5, right=0.5)
fig.set_facecolor("floralwhite")

com_com_text="\n".join((r"$\cdot$ " "The keywords are dominated \n" \
                       "by words that are very descriptive \n"
                        "of movies",
                       r"$\cdot$ " "Most common words are interesting words, \n" \
                       "with very low amount of sentence-building \n"
                       "words",
                       r"$\cdot$ " "Major takeaway is that keywords \n"
                       "are suitable to aggregate main characterstics \n"
                       "of movies"))
com_man_text="\n".join((r"$\cdot$ " "Manually chosen words have \n" \
                       "high counts, relative the common words",
                       r"$\cdot$ " "The usage of 'novel' indicates a \n" \
                       "lot of movies are associated with novel(s)",
                       r"$\cdot$ " "Words like relationship, love \n" \
                       "murder, war, death, drug, family indicates \n"
                        "a wide range of categories of movies exists"))

fig.text(0.53, 0, com_com_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"), 
        family="serif", fontsize=13)
fig.text(0.53, -0.7, com_man_text, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        family="serif", fontsize=13)
