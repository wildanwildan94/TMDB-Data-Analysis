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
### Want to visualize the revenue associated with some cast members


cast_revenue_d=train_d[["cast", "revenue"]].dropna()


# (a) Print a general form
print cast_revenue_d.iloc[0]

# (b) Construct a dataframe of cast members and associated revenue

cast_array=[]
revenue_array=[]
for index, row  in cast_revenue_d.iterrows():
  
  list_actors=ast.literal_eval(row["cast"])
  for act in list_actors:
    cast_array.append(act["name"].lower())
    revenue_array.append(row["revenue"])



cast_revenue_d=pd.DataFrame({'cast':cast_array,
                            'revenue':revenue_array})



# (c) Group cast and calculate total revenue


cast_sumrev_count_d=cast_revenue_d.groupby("cast").agg({'cast':'size',
                                                       'revenue':'sum'}).rename(columns={'cast':'count',
                                                                                        'revenue':'revenue_sum'}).reset_index()


# (d) Consider the top ten by average and sum of revenue


top_sumrev_count_d=cast_sumrev_count_d.sort_values(by="revenue_sum", ascending=False).head(20).sort_values(by="revenue_sum")


top_sumrev_count_m=top_sumrev_count_d.as_matrix()


# (f) Visualize the top ten average and sum of revenue

fig, ax = plt.subplots()

y_labels=range(top_sumrev_count_m.shape[0])




# Sum of revenue
ax.barh(y_labels, top_sumrev_count_m[:,2], facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels([x.title() for x in top_sumrev_count_m[:,0]])
ax.set_title("Sum of Revenue for Cast Members; Top 20")
ax.set_xlabel("Sum Revenue")
ax.set_facecolor("navajowhite")

ax.set_xlim((0, 1.2e10))
fig.subplots_adjust(hspace=0.6, bottom=-0.1)
fig.set_facecolor("floralwhite")
rev_patch=mpatches.Patch(color="black", label="Sum Revenue")
fig.legend(handles=[rev_patch], bbox_to_anchor=(1, 0.5),
          )

com="\n".join((r"$\cdot$ " "Mostly male in the top",
              r"$\cdot$ " "A few females, like \n" \
              "Cate Blanchett, Judi Dench"))
com_rev=r"$\cdot$ " "Based on, for a given Cast Member, \n" \
"collect all movies associated with that \n" \
"cast member, and sum the revenue \n" \
"of those movies"
fig.text(0.92,0.4, com, bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"),
        fontsize=14, family="serif")
fig.text(0.92, 0, com_rev, bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"),
        fontsize=14, family="serif")

for x, y, val in zip(top_sumrev_count_m[:,2], y_labels, top_sumrev_count_m[:,2]):
  cast_rev=str(int(val/float(1000000)))
  cast_rev_str=cast_rev + " Million"
  ax.text(x+0.01e10, y-0.35, cast_rev_str, fontweight="bold")
