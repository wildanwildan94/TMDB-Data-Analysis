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
                   
### Visualize how a smoothed average revenue vary over dates


date_gen_rev_d=train_d[["release_date", "genres", "revenue"]].dropna()

print date_gen_rev_d["release_date"].head(5)
print len(date_gen_rev_d)

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
      


# (c) Compute rolling mean of revenue with respect to datetime; 30 Days intervals

date_rev_d=date_gen_rev_d[["release_date", "revenue"]].set_index("release_date", drop=True).sort_index()
date_rev_rm_d=date_rev_d.rolling('30D').mean()



# (d) Visualize average revenue of time (all times)

fig, ax = plt.subplots(2,1)


fig.suptitle("Smoothed, Average Revenue in Intervals \n of 30 Days", y=1.05)
ax[0].plot(date_rev_rm_d.index, date_rev_rm_d["revenue"], 'r-')
ax[0].set_title("Average Revenue over Time")
ax[0].set_facecolor("navajowhite")
ax[0].set_ylabel("Average Revenue")

# (e) Visualize average revenue over time, recent 10 years

date_rev_rm_rec_d=date_rev_rm_d.query("release_date>'2000-01-01'")


ax[1].plot(date_rev_rm_rec_d.index, date_rev_rm_rec_d["revenue"], 'r-')
ax[1].set_title("Average Revenue over Last 18 Years")
ax[1].set_facecolor("navajowhite")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Average Revenue")
fig.set_facecolor("floralwhite")

fig.subplots_adjust(hspace=0.4)

com="\n".join((r"$\cdot$ " "Smoothed, average revenue \n" \
              "increases over time - \n"\
              "most likely due to inflation \n" \
              "and/or better performing \n" \
              "movies over time, and \n" \
               "an increase in movies done \n" \
               "over time",
              r"$\cdot$ " "For the last 18 years \n" \
              "of average revenue, there is \n" \
              "a clear seasonality and repetition \n" \
              "in the average revenue over \n" \
              "the years"))
fig.text(0.93, 0.2, com, bbox=dict(boxstyle="round", edgecolor="black",
                                 facecolor="wheat"),
        fontsize=13, family="serif")







