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
                   
                   
### Initial consideration of the genres attribute; With visualization of amount 
### movies related to each possible genre

## (a) Present the format of the genres

print train_d["genres"].iloc[3]

## (b) Construct a List of Entries in Genres; 

# Define and fill a list of dicts for the genres for all movies
genres_movies_dicts=[]
for item in train_d["genres"].dropna().as_matrix():
  new_item=ast.literal_eval(item)
  genres_movies_dicts.extend(new_item)
  
  
  
# Define and fill a list of all the genre strings
genres_movies=[]

for item in genres_movies_dicts:
  genres_movies.append(item["name"])

## (c) Count the Occurence of Each Genre

count_genres_movies=Counter(genres_movies).items()



## (d) Create arrays for each genre, and its name, and amount of 
## movies in that genre

genre, count_genre=zip(*count_genres_movies)
genre=list(genre)
count_genre=list(count_genre)

# Sort List Depending on count_genre
genre=[x for _,x in sorted(zip(count_genre, genre))]
count_genre=[x for x,_ in sorted(zip(count_genre, genre))]


## (e) Visualizat amount of movies in each genre

fig, ax = plt.subplots()

y_labels=range(len(genre))
ax.barh(y_labels, count_genre, facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(genre)
ax.set_xlabel("Count")
ax.set_facecolor("navajowhite")
fig.suptitle("Amount of Movies Related to Each Genre")

textstr="\n".join((r"$\cdot$" "The most popular genres are drama, \n comedy, thriller and action",
                  r"$\cdot$ Relatively few music and" "\n documentary type movies"))

props=dict(boxstyle="round", facecolor="wheat", alpha=0.5)

fig.text(0.93, 0.5, textstr, fontsize=14, bbox=props, family="serif")
fig.set_facecolor("floralwhite")
