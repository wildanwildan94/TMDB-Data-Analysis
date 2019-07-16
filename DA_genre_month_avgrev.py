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
### Consideration of revenue over months for top genres

## Want to consider the revenue for top genres over different months
## (Adventure, Animation, Fantasy, Family)

### Continution date, genres, income and revenue

## Want to analyze how income, revenue vary over dates, and with respecto genres

date_gen_rev_d=train_d[["release_date", "genres", "revenue"]].dropna()



# (a) Analyze the form of release_date

date_str=date_gen_rev_d["release_date"].iloc[0]
date_time_obj=datetime.datetime.strptime(date_str, '%m/%d/%y')
print date_time_obj

# (b) Apply strptime to release_date for datetime format

date_gen_rev_d["release_date"]=[datetime.datetime.strptime(x, '%m/%d/%y') for x in date_gen_rev_d["release_date"]]


new_release_date=[]

for index, row in date_gen_rev_d.iterrows():
  temp_date=row["release_date"]
  if temp_date.year>=2020:
    new_release_date.append(temp_date.replace(year=temp_date.year-100))
  else:
    new_release_date.append(temp_date)
    
    
date_gen_rev_d["release_date"]=new_release_date


# (c) Create a month attribute in the data

date_gen_rev_d["month"]=[x.strftime("%B") for x in date_gen_rev_d["release_date"]]
date_gen_rev_d["month_index"]=[int(x.strftime("%-m")) for x in date_gen_rev_d["release_date"]]



# (d) Add genre name to each movie

genre_name_movies=[]
month_movies=[]
month_index_movies=[]
revenue_movies=[]
for index, row in date_gen_rev_d.iterrows():
  temp_genres=ast.literal_eval(row["genres"])
  temp_month=row["month"]
  temp_month_index=row["month_index"]
  temp_revenue=row["revenue"]
  for genre in temp_genres:
    genre_name_movies.append(genre["name"])
    month_movies.append(temp_month)
    month_index_movies.append(temp_month_index)
    revenue_movies.append(temp_revenue)
    


genre_mov_d=pd.DataFrame({'genre_name':genre_name_movies,
                         'month':month_movies,
                         'month_index':month_index_movies,
                         'revenue':revenue_movies})



genre_mov_agg_d= genre_mov_d.groupby(["genre_name", "month"]).agg({'revenue':['mean', 'median'],
                                                       'month_index':'first'})

genre_mov_agg_d.columns=genre_mov_agg_d.columns.map('_'.join)
genre_mov_agg_d.reset_index(inplace=True)


# Only consider certain genres

genre_mov_agg_top_d=genre_mov_agg_d.query("genre_name=='Adventure' or genre_name=='Animation' or genre_name=='Fantasy' or genre_name=='Family'")
genre_mov_agg_cust_d=genre_mov_agg_d.query("genre_name=='Comedy' or genre_name=='Crime' or genre_name=='Horror' or genre_name=='Romance' or genre_name=='Science Fiction' or genre_name=='Thriller'")
amount_genres=4


y_labels=np.arange(1,13)
y_month=["January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"]

cm_cust=plt.get_cmap("inferno")
size=4

fig, ax = plt.subplots(2,1)

i=0
for a, b in genre_mov_agg_top_d.groupby("genre_name"):
  b=b.sort_values(by="month_index_first")
  ax[0].plot(b["month_index_first"],
          b["revenue_mean"],
          alpha=0.8,
          marker="o",
          linestyle="-",
          color=cm_cust(i/float(size)),
         label="%s"%a,
         markeredgecolor="black")
  i+=1
  
  
  
ax[0].set_xticks(y_labels)
ax[0].set_xticklabels(y_month)
ax[0].set_ylabel("Average Revenue")
ax[0].set_title("Average Revenue for Different Months for Different Genres \n Last 18 Years of Data")
ax[0].set_facecolor("navajowhite")

ax[0].legend(bbox_to_anchor=(1,1))

size=6
i=0
for a, b in genre_mov_agg_cust_d.groupby("genre_name"):
  b=b.sort_values(by="month_index_first")
  ax[1].plot(b["month_index_first"],
          b["revenue_mean"],
          alpha=0.8,
          marker="o",
          linestyle="-",
          color=cm_cust(i/float(size)),
         label="%s"%a,
         markeredgecolor="black")
  i+=1
  
  
  
ax[1].set_xticks(y_labels)
ax[1].set_xticklabels(y_month)
ax[1].set_ylabel("Average Revenue")
ax[1].set_xlabel("Month")
ax[1].set_facecolor("navajowhite")
ax[1].legend(bbox_to_anchor=(1,1))

fig.subplots_adjust(left=0.3, right=2, hspace=0.3)
fig.set_facecolor("floralwhite")

com="\n".join((r"$\cdot$ " "Most genres have average revenue peaks during the semester seasons (summer & winter)",
              r"$\cdot$ " "The average revenue over months vary between different genres"))

fig.text(0.4, -0.15, com, bbox=dict(boxstyle="round", edgecolor="black",
                                facecolor="wheat"), 
        fontsize=14, family="serif")
