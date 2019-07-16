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
### Want to consider the relationship between keywords and 
### revenue for the movies the keywords are associated with

keyw_revenue_d=train_d[["Keywords", "revenue"]].dropna()

# (a) Present the form of keywords

print keyw_revenue_d.iloc[0]


# (b) Define and fill a list of the dicts


word_in_keywords=[]
word_revenue=[]
for index, row in keyw_revenue_d.iterrows():
  temp_keywords=ast.literal_eval(row["Keywords"])
  
  for keyword in temp_keywords:
    word_in_keywords.append(keyword["name"].lower())
    word_revenue.append(row["revenue"])
  

# (c) Define a dataframe of word and associated revenue

word_revenue_d=pd.DataFrame({'word':word_in_keywords,
                            'revenue':word_revenue})
print word_revenue_d.head(5)

# (b) Group all words by average revenue and count

word_avgrev_count_d=word_revenue_d.groupby("word").agg({'word':'size',
                                                       'revenue':'mean'}).rename(columns={'word':'word_count',
                                                                                         'revenue':'revenue_mean'}).reset_index()

print word_avgrev_count_d.head(5)


# (c) Consider top twenty words by average revenue, conditioned that they appear at least 10 times in total

top_words_avgrev_d=word_avgrev_count_d.query("word_count>=10").sort_values(by="revenue_mean", ascending=False).head(20).sort_values(by="revenue_mean")
top_words_avgrev_m=top_words_avgrev_d.as_matrix()

# (d) Visualize top twenty words in two plots


fig, ax = plt.subplots()

cutoff=int(top_words_avgrev_m.shape[0]/float(2))
cutoff=int(top_words_avgrev_m.shape[0])
y_labels=range(cutoff)

ax.barh(y_labels, top_words_avgrev_m[0:cutoff, 2], facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels([x.title() for x in top_words_avgrev_m[0:cutoff, 0]], fontsize=13)
ax.set_xlabel("Revenue")
ax.set_facecolor("navajowhite")

fig.suptitle("Average Revenue of Keywords")
fig.subplots_adjust(hspace=0.2, bottom=-0.23)
fig.set_facecolor("floralwhite")

com="\n".join((r"$\cdot$" "Many of the words \n" \
              "indicate that high-revenue \n" \
              "associated keywords characterizes \n" \
              "movies well.",
              r"$\cdot$ " "For example, keywords \n" \
              "associated with superhero movies \n" \
              "are frequent in top keywords", 
              r"$\cdot$ " "Also notice words like Dinosaur, \n" \
              "Secret Identity, Mission, Mutant, etc. \n" \
              "which all clearly describe an \n" \
              "important aspect of their \n" \
              "associated movies"))
com_keyword=r"$\cdot$ "" For each keyword, collect all movies \n" \
                      "with that keyword. Then, take the \n" \
                      "average  of the revenue associated \n" \
                      "with these movies"

fig.text(0.92, 0.15, com, bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"),
        fontsize=14, family="serif")
fig.text(0.92, -0.2, com_keyword, bbox=dict(boxstyle="round", edgecolor="black",
                                           facecolor="wheat"),
        fontsize=14, family="serif")
  
