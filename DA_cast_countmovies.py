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
### Analyze of the cast attribute, with an emphasis on
### the most popular cast members and gender balance of 
### cast members in movies


cast_d=train_d["cast"].dropna().as_matrix()


# (a) Print a general form
print cast_d[0]

# (b) Construct a list of the dicts for cast

cast_dicts=[]
for item in cast_d:
  new_item=ast.literal_eval(item)
  cast_dicts.extend(new_item)


# (b) Extract a list of all cast occuring in movies, and extract all genders occuring in movies
actors_in_cast=[]
genders_in_cast=[]

for item in cast_dicts:
  new_item_name=item["name"]
  new_item_gender=item["gender"]
  actors_in_cast.append(new_item_name)
  genders_in_cast.append(new_item_gender)


 
# (c) Count the occurence of each actor and gender

count_actors_cast=Counter(actors_in_cast)
count_genders_cast=Counter(genders_in_cast)

# (d) Define the most common actors and the two genders

actor_names, actor_count=zip(*count_actors_cast.most_common(10))
gender_names, gender_count=zip(*count_genders_cast.items())


# Sort List Depending on count 
actor_names=[x for _,x in sorted(zip(actor_count, actor_names))]
actor_count=[x for x,_ in sorted(zip(actor_count, actor_names))]


# (f) Visualize the occurence of actors and gender

fig, ax = plt.subplots(1,2)

y_labels_actor=range(len(actor_names))
y_labels_gender=range(len(gender_names))

ax[0].barh(y_labels_actor, actor_count, facecolor="crimson", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(actor_names)
ax[0].set_facecolor("navajowhite")
ax[0].set_xlabel("Count")
ax[0].set_title("Amount of Movies with Cast Member")

new_gender_names=["Unset", "Female", "Male"]
ax[1].bar(y_labels_gender, gender_count, facecolor="crimson", edgecolor="black")
ax[1].set_xticks(y_labels_gender)
ax[1].set_xticklabels(new_gender_names)
ax[1].set_facecolor("navajowhite")
ax[1].set_ylabel("Count")
ax[1].set_title("Gender of Cast Members")


fig.subplots_adjust(left=-0.3, right=0.5, wspace=0.6)
fig.set_facecolor("floralwhite")

com="\n".join((r"$\cdot$ " "Most popular cast members are \n" \
              "male, with just a single female",
              r"$\cdot$ " "There is a large difference in amount of \n" \
              "male and female cast in movies, indicating \n" \
              "a large gender imbalance in the cast of \n" \
              "movies"))

fig.text(0.53, 0.3, com, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        family="serif", fontsize=13)

