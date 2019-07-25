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

### Want to visualize the top five movies by revenue for top three genres


genre_revenue_d=train_d[["title", "genres", "revenue"]].dropna(how="any")

print genre_revenue_d.iloc[0]
# (a) Create an dataframe with every genre and revenue of each movie

genre_name_array=[]
title_name_array=[]
revenue_name_array=[]
for item in genre_revenue_d.as_matrix():
  temp_title=item[0]
  temp_genre=item[1]
  temp_revenue=item[2]
  genre_literal=ast.literal_eval(temp_genre)
  for temp_genre in genre_literal:
    genre_name_array.append(temp_genre["name"])
    title_name_array.append(temp_title)
    revenue_name_array.append(temp_revenue)
    
    


movie_genres_revenue_d=pd.DataFrame({'title':title_name_array,
                                    'genre':genre_name_array,
                                    'revenue':revenue_name_array})

# (a) Create matrix of data for the top five movies of the top three genres


topmovies_adventure_m=movie_genres_revenue_d.query("genre=='Adventure'").sort_values(by="revenue", ascending=False).head(5).as_matrix()
topmovies_fantasy_m=movie_genres_revenue_d.query("genre=='Fantasy'").sort_values(by="revenue", ascending=False).head(5).as_matrix()
topmovies_animation_m=movie_genres_revenue_d.query("genre=='Animation'").sort_values(by="revenue", ascending=False).head(5).as_matrix()


# (b) Visualize top performing movies for top genres

fig, ax=plt.subplots(3,1, sharex=True)


y_labels_adv=range(topmovies_adventure_m.shape[0])


ax[0].barh(y_labels_adv, topmovies_adventure_m[:,1], facecolor="royalblue", edgecolor="black")
ax[0].set_yticks(y_labels_adv)
ax[0].set_yticklabels(topmovies_adventure_m[:,2], fontsize=13)
ax[0].set_facecolor("navajowhite")
ax[0].set_title("Top Movies by Revenue for Adventure Movies")



y_labels_fan=range(topmovies_fantasy_m.shape[0])

ax[1].barh(y_labels_fan, topmovies_fantasy_m[:,1], facecolor="royalblue", edgecolor="black")
ax[1].set_yticks(y_labels_fan)
ax[1].set_yticklabels(topmovies_fantasy_m[:,2], fontsize=13)
ax[1].set_facecolor("navajowhite")
ax[1].set_title("Top Movies by Revenue for Fantasy Movies")



y_labels_ani=range(topmovies_animation_m.shape[0])

ax[2].barh(y_labels_ani, topmovies_animation_m[:,1], facecolor="royalblue", edgecolor="black")
ax[2].set_yticks(y_labels_ani)
ax[2].set_yticklabels(topmovies_animation_m[:,2], fontsize=13)
ax[2].set_facecolor("navajowhite")
ax[2].set_title("Top Movies by Revenue for Animation Movies")
ax[2].set_xlabel("Revenue")


fig.subplots_adjust(hspace=0.3, bottom=-0.1)
fig.set_facecolor("floralwhite")

com="\n".join((r"$\cdot$ " "Top adventure movies are \n" \
              "the most profitable",
              r"$\cdot$ " "The most profitable movies \n" \
              "from all genres are popular \n" \
               "mainstream movies"))

fig.text(0.93, 0.2, com, bbox=dict(boxstyle="round", edgecolor="black",
                               facecolor="wheat"),
        fontsize=14, family="serif")
