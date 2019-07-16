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

### Top collections and its associated average revenue (over movies in collection)



btcr_d=train_d[["belongs_to_collection", "revenue"]].dropna()

# (a) Create an array of names of collection for each movie, to be inserted into btcr_d

collection_array=[]
for item in btcr_d["belongs_to_collection"]:
  new_item=ast.literal_eval(item)
  new_item_name=new_item[0]["name"]
  collection_array.append(new_item_name)
  
  
btcr_d["collection_name"]=collection_array

# (b) Aggregate with respect to amount in collection and average revenue and standard dev

collection_count_meanrev=btcr_d[["collection_name", "revenue"]].groupby("collection_name").agg({'collection_name':'size',
                                                                             'revenue':'mean'}).rename(columns={'collection_name':'collection_count',
                                                                                                               'revenue':'revenue_mean'}).reset_index()

# (c) Store the collections with the most amount of 
# associated movies (top ten)

top_collection=collection_count_meanrev.sort_values(by="collection_count", ascending=False).head(10)

# (d) Visualize top ten collections with revenue

y_labels_collection=range(top_collection.shape[0])
y_labels_name=top_collection.iloc[:,0].as_matrix()

y_labels_name[4]="Pokemon Collection"

fig, ax = plt.subplots()

ax.barh(y_labels_collection, top_collection.iloc[:,2], facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels_collection)
ax.set_yticklabels(y_labels_name, fontsize=13)
ax.set_facecolor("navajowhite")
ax.set_xlabel("Average Revenue")
ax.set_title("Average Revenue of Movies in Collections; \n With Amount of Movies in Collection in Black Text")


for x,y, val in zip(top_collection.iloc[:,2], y_labels_collection, top_collection.iloc[:,1]):
  ax.text(x+0.05e8, y, str(val), color="black", fontweight="bold")
  
  
fig.subplots_adjust(wspace=-0.1, bottom=-0.05)

com="\n".join((r"$\cdot$ " "The collections with the \n" \
              "most movies are mainstream collections, \n" \
              "- like Transformers and James Bond",
              r"$\cdot$ " "Most popular collections have \n" \
              "between 4-5 movies", 
              r"$\cdot$ " "The collections with the most, average \n" \
              "revenue are Transformers, Ice Age and \n" \
              "The Fast and the Furious, which \n" \
              "is not a surprise"))
fig.text(0.93, 0.2, com, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        fontsize=13, family="serif")
fig.set_facecolor("floralwhite")

count_patch=mpatches.Patch(color="black", label="Amount of Movies \n in Collection")
fig.legend(handles=[count_patch], bbox_to_anchor=(1.2,0.5))
#for i, v in enumerate(top_collection.iloc[:,1].as_matrix()):
#  ax.text(v+3, i-0.1, str(v), color="black", fontweight="bold")
