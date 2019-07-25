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
## Consideration of director and writer and revenue

crew_revenue_d=train_d[["crew", "revenue"]].dropna()

# (a) Print a general form and construct a list of dicts
print crew_revenue_d.iloc[0]
#def strip_accents(text):

# (b) Construct dataframe for revenue for each writer and director 
dir_array=[]
writer_array=[]
dir_rev_array=[]
writer_rev_array=[]

for index, row in crew_revenue_d.iterrows():
  new_crew=ast.literal_eval(row["crew"])
  
  for cm in new_crew:
    if cm["job"]=='Director':
      dir_array.append(strip_accents(cm["name"].lower()))
      dir_rev_array.append(row["revenue"])
      
    elif cm["job"]=='Writer':
      writer_array.append(strip_accents(cm["name"].lower()))
      writer_rev_array.append(row["revenue"])
      
      
dir_rev_d=pd.DataFrame({'dir_name':dir_array,
                       'dir_revenue':dir_rev_array})
writ_rev_d=pd.DataFrame({'writ_name':writer_array,
                        'writ_rev':writer_rev_array})

      
# (c) Construct top writer and director, for both average and sum of revenue

avgrev_dir_d=dir_rev_d.groupby("dir_name").agg({'dir_name':'size',
                                                   'dir_revenue':'mean'}).rename(columns={'dir_name':'dir_count',
                                                                                         'dir_revenue':'revenue_mean'}).reset_index()
sumrev_dir_d=dir_rev_d.groupby("dir_name").agg({'dir_name':'size',
                                                   'dir_revenue':'sum'}).rename(columns={'dir_name':'dir_count',
                                                                                        'dir_revenue':'revenue_sum'}).reset_index()
avgrev_writ_d=writ_rev_d.groupby("writ_name").agg({'writ_name':'size',
                                                      'writ_rev':'mean'}).rename(columns={'writ_name':'writ_count',
                                                                                          'writ_rev':'revenue_mean'}).reset_index()
sumrev_writ_d=writ_rev_d.groupby("writ_name").agg({'writ_name':'size',
                                                       'writ_rev':'sum'}).rename(columns={'writ_name':'writ_count',
                                                                                         'writ_rev':'revenue_sum'}).reset_index()


# (d) Consider the top 5 of each category considered


top_sumrev_dir_m=sumrev_dir_d.sort_values(by="revenue_sum", ascending=False).head(10).sort_values(by="revenue_sum").as_matrix()
top_sumrev_writ_m=sumrev_writ_d.sort_values(by="revenue_sum", ascending=False).head(10).sort_values(by="revenue_sum").as_matrix()


# (e) Visualize the top 5 of each category considered

fig, ax = plt.subplots(1,2)

y_labels=range(top_sumrev_dir_m.shape[0])



# Director, sum of revenue
ax[0].barh(y_labels, top_sumrev_dir_m[:,2], facecolor="royalblue", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels([x.title() for x in top_sumrev_dir_m[:,0]])
ax[0].set_facecolor("navajowhite")
ax[0].set_title("Sum of Revenue for Top 10 Directors")
ax[0].set_xlabel("Revenue")


# Writer, sum of revenue
ax[1].barh(y_labels, top_sumrev_writ_m[:,2], facecolor="royalblue", edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels([x.title() for x in top_sumrev_writ_m[:,0]])
ax[1].set_facecolor("navajowhite")
ax[1].set_title("Sum of Revenue for Top 10 Writers")
ax[1].set_xlabel("Revenue")

fig.subplots_adjust(hspace=0.6, right=1.65, wspace=0.45)
fig.set_facecolor("floralwhite")
com_dir="\n".join((r"$\cdot$ " "Only male directors present",
              r"$\cdot$ " "Many well-known directors \n" \
              "present, like Peter Jackson and \n" \
              "Michael Bay"))
com_writ="\n".join((r"$\cdot$ " "As for directors, \n" \
                   "only male writers present",
                  r"$\cdot$ " "Chris Morgan (The Fast \n" \
                  "and Furious) and Joss Whedon \n" \
                   "(Avengers) in the top"))

fig.text(0.2, -0.25, com_dir, bbox=dict(boxstyle="round", facecolor="wheat",
                                      edgecolor="black"),
        fontsize=13, family="serif")
fig.text(1.15, -0.3, com_writ, bbox=dict(boxstyle="round", facecolor="wheat",
                                       edgecolor="black"),
        fontsize=13, family="serif")
