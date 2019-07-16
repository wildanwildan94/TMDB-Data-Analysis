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
### Visualize top performing movies in four genres:
### adventure, family, animation and fantasy


genre_revenue_d=train_d[["title", "genres", "revenue", "budget"]].dropna(how="any")

print genre_revenue_d.iloc[0]

# (a) Create an dataframe with every genre and revenue of each movie

genre_name_array=[]
title_name_array=[]
revenue_name_array=[]
budget_name_array=[]
income_array=[]
for item in genre_revenue_d.as_matrix():
  temp_title=item[0]
  temp_genre=item[1]
  temp_revenue=item[2]
  temp_budget=item[3]
  temp_income=temp_revenue-temp_budget
  genre_literal=ast.literal_eval(temp_genre)
  for temp_genre in genre_literal:
    genre_name_array.append(temp_genre["name"])
    title_name_array.append(temp_title)
    revenue_name_array.append(temp_revenue)
    budget_name_array.append(temp_budget)
    income_array.append(temp_income)
    
    


movie_genres_revenue_d=pd.DataFrame({'title':title_name_array,
                                    'genre':genre_name_array,
                                    'revenue':revenue_name_array,
                                    'budget':budget_name_array,
                                    'income':income_array})

topgenre_inc_name_d=movie_genres_revenue_d.query("genre=='Adventure' or genre=='Fantasy' or genre=='Animation' or genre=='Family'")

# (b) Create matrixs of most profitable and 


# (c) Visualize top 5 profitable and unprofitable

fig, ax = plt.subplots(2,1)

genre_names=["Adventure", "Fantasy", "Animation", "Family"]
color_rev=["royalblue", "lightsteelblue", "navy", "slateblue"]
color_inc=["maroon", "firebrick","red", "lightcoral"]

cutoff=5

prof_titles=[]
unprof_titles=[]
N=len(genre_names)*cutoff
for i in range(len(genre_names)):
  y_labels=np.arange((N-(i+1)*cutoff), N-(i)*cutoff)
  print y_labels
  top_prof=topgenre_inc_name_d.query("genre=='%s'"%genre_names[i]).sort_values(by="income", ascending=False).head(5).sort_values(by="revenue")
  min_prof=topgenre_inc_name_d.query("genre=='%s'"%genre_names[i]).sort_values(by="income", ascending=True).head(5).sort_values(by="revenue")
  ax[0].barh(y_labels, top_prof["revenue"], facecolor=color_rev[i], edgecolor="black")
  ax[0].barh(y_labels, top_prof["income"], facecolor=color_inc[i], edgecolor="black")
  ax[1].barh(y_labels, min_prof["revenue"], facecolor=color_rev[i], edgecolor="black")
  ax[1].barh(y_labels, min_prof["income"], facecolor=color_inc[i], edgecolor="black")
  prof_titles.extend(top_prof["title"])
  unprof_titles.extend(min_prof["title"])
y_labels=range(len(prof_titles))
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(prof_titles)
ax[0].set_xlabel("Revenue/Income")
ax[0].set_title("Top 5 Performing Movies in Genres")
ax[0].set_facecolor("navajowhite")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels(unprof_titles)
ax[1].set_xlabel("Revenue/Income")
ax[1].set_title("Worst 5 Performing Movies in Genres")
ax[1].set_facecolor("navajowhite")

fig.subplots_adjust(hspace=0.25, bottom=0, top=1.6)

adv_rev_patch=mpatches.Patch(color=color_rev[0], label="Adventure Revenue")
fan_rev_patch=mpatches.Patch(color=color_rev[1], label="Fantasy Revenue")
ani_rev_patch=mpatches.Patch(color=color_rev[2], label="Animation Revenue")
fam_rev_patch=mpatches.Patch(color=color_rev[3], label="Family Revene")

adv_inc_patch=mpatches.Patch(color=color_inc[0], label="Adventure Income")
fan_inc_patch=mpatches.Patch(color=color_inc[1], label="Fantasy Income")
ani_inc_patch=mpatches.Patch(color=color_inc[2], label="Animation Income")
fam_inc_patch=mpatches.Patch(color=color_inc[3], label="Family Income")

fig.legend(handles=[adv_rev_patch, fan_rev_patch, ani_rev_patch, fam_rev_patch, adv_inc_patch, fan_inc_patch, ani_inc_patch, fam_inc_patch],
          ncol=2, bbox_to_anchor=(2.05, 1.7))
fig.set_facecolor("floralwhite")


com="\n".join((r"$\cdot$ " "Top performing movies \n" \
              "are well-known blockbuster \n" \
              "movies",
              r"$\cdot$ " "Movies from the same \n" \
              "collection are frequent, like \n" \
              "Lord of the Rings, and Ice Age \n" \
              "movies",
              r"$\cdot$ " "A couple of movies \n" \
              "appear in multiple genres, like \n" \
              "Zootopia",
              r"$\cdot$ " "The only well-known movie \n" \
              "among the worst performing is \n" \
              "Titan A.E. - otherwise there are \n" \
              "only unknown movies - as expected \n"\
               "perhaps"))

fig.text(1, 0, com, bbox=dict(boxstyle="round", facecolor="wheat",
                                 edgecolor="black"),
        fontsize=12, family="serif")
