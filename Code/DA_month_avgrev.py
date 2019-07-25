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
### Visualize how average revenue changes over months in the year


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


# (d) Group by month and do mean and median operation

# act_profm_d.columns=act_profm_d.columns.map('_'.join)
date_gen_rev_agg_d=date_gen_rev_d[["month", "revenue", "month_index"]].groupby("month").agg({'revenue':['mean', 'median'],
                                                                             'month_index':'first'})
date_gen_rev_agg_d.columns=date_gen_rev_agg_d.columns.map('_'.join)
date_gen_rev_agg_d.reset_index(inplace=True)
date_gen_rev_agg_d.sort_values(by="month_index_first", inplace=True)

# (e) Visualize revenue mean and median 


fig, ax =plt.subplots()

y_labels=date_gen_rev_agg_d["month_index_first"]
month_labels=date_gen_rev_agg_d["month"]

ax.barh(y_labels, date_gen_rev_agg_d["revenue_mean"], facecolor="royalblue",
        edgecolor="black")
ax.set_title("Average Revenue of Each Month \n Last 18 Years")
ax.set_facecolor("navajowhite")
ax.set_xlabel("Average Revenue")
ax.set_yticks(y_labels)
ax.set_yticklabels(month_labels, fontsize=13)


fig.subplots_adjust(wspace=0.8)
fig.set_facecolor("floralwhite")
com="\n".join((r"$\cdot$ " "Average revenue peaks \n"\
              "during the summer and \n" \
               "December",
              r"$\cdot$ " "Indicates that semester \n" \
              "seasons are attractive for \n" \
              "high revenue movies"))
com_rev=r"$\cdot$ " "Based on, for a given month, \n"\
"collect all movies released in that \n"\
"month, and take the average of \n" \
"the revenue of the movies"

fig.text(0.92, 0.5, com, bbox=dict(boxstyle="round", facecolor="wheat",
                                edgecolor="black"),
        fontsize=13, family="serif")
fig.text(0.92, 0.2, com_rev, bbox=dict(boxstyle="round", facecolor="wheat",
                                   edgecolor="black"),
        fontsize=13, family="serif")
