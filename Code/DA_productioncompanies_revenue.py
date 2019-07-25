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
### Consideration of production_companies and revenue


# (a) Construct a dataframe with all movies, associated production company and revenue

prod_revenue_d=train_d[["production_companies", "revenue"]].dropna()



# (b) Define and fill a dataframe with all production companies and associated revenues

prods_array=[]
revenue_array=[]
for item in prod_revenue_d.as_matrix():
  prod_list=ast.literal_eval(item[0])
  for prod in prod_list:
    prods_array.append(prod["name"])
    revenue_array.append(item[1])
    

prod_revenue_tot_d=pd.DataFrame({'prod_comp':prods_array,
                                'revenue':revenue_array})


# (c) Construct a dataframe with the count, and sum of revenue for each production company


prod_sumrev_count_d=prod_revenue_tot_d.groupby("prod_comp").agg({'revenue':np.sum,
                                                                'prod_comp':'size'}).rename(columns={'revenue':'revenue_sum',
                                                                                                    'prod_comp':'count'}).reset_index()

# (d) Construct data of top 10 production companies for sum of revenue

top_prod_sumrev_m=prod_sumrev_count_d.sort_values(by="revenue_sum", ascending=False).head(10).sort_values(by="revenue_sum").as_matrix()

  
# (e) Visualize the top ten production companies by sum revenue

y_ticklabels_sum=top_prod_sumrev_m[:,0]



fig, ax = plt.subplots()

y_labels=range(top_prod_sumrev_m.shape[0])


ax.barh(y_labels, top_prod_sumrev_m[:,2], facecolor="royalblue", edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(top_prod_sumrev_m[:,0], fontsize=13)
ax.set_facecolor("navajowhite")
ax.set_title("Top Production Companies by Total Revenue")
ax.set_xlabel("Revenue", fontsize=13)


fig.set_facecolor("floralwhite")
com=r"$\cdot$ Top production " "\n" \
"companies are the \n"\
"well-known American giants"

com_rev=r"$\cdot$ " "Based on, for a given production \n" \
"company, collect all movies associated \n" \
"with that production company, \n" \
"and sum the revenue of those movies"


fig.text(0.92, 0.5, com, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        fontsize=13, family="serif")
fig.text(0.92, 0.2, com_rev, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        fontsize=13, family="serif")
