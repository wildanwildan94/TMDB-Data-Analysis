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
### Consideration of revenue, income on genre

## Want to consider the average revenue of each genre, and the count of movies in each genre

genre_revenue_d=train_d[["title", "genres", "revenue", "budget"]].dropna(how="any")

print genre_revenue_d.iloc[0]
# (a) Create an dataframe with every genre and revenue of each movie

genre_name_array=[]
title_name_array=[]
revenue_name_array=[]
budget_name_array=[]
for item in genre_revenue_d.as_matrix():
  temp_title=item[0]
  temp_genre=item[1]
  temp_revenue=item[2]
  temp_budget=item[3]
  genre_literal=ast.literal_eval(temp_genre)
  for temp_genre in genre_literal:
    genre_name_array.append(temp_genre["name"])
    title_name_array.append(temp_title)
    revenue_name_array.append(temp_revenue)
    budget_name_array.append(temp_budget)
    
    


movie_genres_revenue_d=pd.DataFrame({'title':title_name_array,
                                    'genre':genre_name_array,
                                    'revenue':revenue_name_array,
                                    'budget':budget_name_array})

# (b) Calculate the average revenue and count of genre

genre_avgrev_count_d=movie_genres_revenue_d[["genre", "revenue", "budget"]].groupby("genre").agg({'genre':'size',
                                                                             'revenue':'mean',
                                                                                                 "budget":'mean'}).rename(columns={'genre':'genre_count',
                                                                                                               'revenue':'revenue_mean',
                                                                                                                                  "budget":"budget_mean"}).reset_index()


# (c) Visualize the average revenue and count of each genre

genre_avgrev_count_sorted_m=genre_avgrev_count_d.sort_values(by="revenue_mean", ascending=True).as_matrix()


y_labels=range(genre_avgrev_count_sorted_m.shape[0])

fig, ax = plt.subplots()


ax.barh(y_labels, genre_avgrev_count_sorted_m[:,3], edgecolor="black", facecolor="slateblue", alpha=0.8)
ax.barh(y_labels, genre_avgrev_count_sorted_m[:,2], edgecolor="black", facecolor="royalblue", alpha=0.5)
ax.set_yticks(y_labels)
ax.set_yticklabels(genre_avgrev_count_sorted_m[:,0], fontsize=12)
ax.set_facecolor("navajowhite")
ax.set_xlim((0, 2e8))
ax.set_title("Average Revenue & Budget for Different Genres", fontsize=13)
ax.set_xlabel("Avg. Revenue/Budget")
ax.set_ylabel("Genre Name")
ax.set_xlim((0, 2.7e08))
fig.subplots_adjust(right=0.5, left=-0.5, top=0.5, bottom=-0.5)

# Add custom legends
rev_patch=mpatches.Patch(color="slateblue", label="Average Revenue", alpha=0.8)
bug_patch=mpatches.Patch(color="royalblue", label="Average Budget", alpha=0.5)
inc_patch=mpatches.Patch(color="black", label="Income")
fig.legend(handles=[rev_patch, bug_patch, inc_patch], bbox_to_anchor=(1.2, 0.6),
          facecolor="beige", fontsize=13, edgecolor="black")

fig.set_facecolor("floralwhite")

com="\n".join((r"$\cdot$ " "Most profitable genres are \n" \
              "Adventure, Fantasy, Animation, Family. \n" \
              "Possibly because those genres attract \n" \
              "parents & children, but also adults",
              r"$\cdot$ " "Most other genres generate much \n" \
              "less in revenue and profit",
              r"$\cdot$ " "History-centered movies seems \n" \
              "to perform the worst"))
fig.text(0.53, -0.25, com, bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"),
        fontsize=13, family="serif")

for i,j,k,l in zip(y_labels, genre_avgrev_count_sorted_m[:,1], genre_avgrev_count_sorted_m[:,3], genre_avgrev_count_sorted_m[:,2]):
  inc=k-l
  inc_mill=inc
  inc_mill=np.round(inc/float(1000000), 1)
  inc_mill_str=str(inc_mill)+" Million USD"
  ax.text(k+0.02e08, i-0.3, inc_mill_str, color="black", fontweight="bold")
  

