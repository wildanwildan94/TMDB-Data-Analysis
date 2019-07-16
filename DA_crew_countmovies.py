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
### Consideration of Crew Attribute

## First a analyze of the form

crew_d=train_d["crew"].dropna().as_matrix()

# (a) Print a general form and construct a list of dicts
print crew_d[0]

crew_dicts=[]
for item in crew_d:
  new_item=ast.literal_eval(item)
  crew_dicts.extend(new_item)

# (b) Create arrays with director, writer names and genders

dir_names=[]
writ_names=[]
dir_gender=[]
writ_gender=[]

for item in crew_dicts:
  
  if item["job"]=="Director":
    dir_names.append(item["name"])
    dir_gender.append(item["gender"])
  elif item["job"]=="Writer":
    writ_names.append(item["name"])
    writ_gender.append(item["gender"])
    
print dir_names[0]
print dir_gender[0]
print writ_names[0]
print writ_gender[0]


# (c)  Count the occurence of each director, writer, director gender and writer gender

dir_names_count=Counter(dir_names)
dir_gender_count=Counter(dir_gender)
writ_names_count=Counter(writ_names)
writ_gender_count=Counter(writ_gender)

print writ_names_count

# (d) Extract the top ten of director and writers, and extract counts for genders

dir_names_topten=dir_names_count.most_common(10)
writ_names_topten=writ_names_count.most_common(10)
dir_gender_count_array=dir_gender_count.items()
writ_gender_count_array=writ_gender_count.items()


# (e) Create lists of labels and counts and sort

dir_names_topten_names, dir_names_topten_count=zip(*dir_names_topten)
writ_names_topten_names, writ_names_topten_count=zip(*writ_names_topten)
dir_gender_names, dir_gender_count=zip(*dir_gender_count_array)
writ_gender_names, writ_gender_count=zip(*writ_gender_count_array)

dir_names_topten_names=[x for _,x in sorted(zip(dir_names_topten_count, dir_names_topten_names))]
dir_names_topten_count=[x for x,_ in sorted(zip(dir_names_topten_count, dir_names_topten_names))]

writ_names_topten_names=[x for _,x in sorted(zip(writ_names_topten_count, writ_names_topten_names))]
writ_names_topten_count=[x for x,_ in sorted(zip(writ_names_topten_count, writ_names_topten_names))]

# (f) Visualize the Distribution of top director writers and the gender balance for director and writers

fig, ax=plt.subplots(2,2)

y_labels_dirnames=range(len(dir_names_topten_names))
ax[0,0].barh(y_labels_dirnames, dir_names_topten_count, facecolor="royalblue",
          edgecolor="black")
ax[0,0].set_yticks(y_labels_dirnames)
ax[0,0].set_yticklabels(dir_names_topten_names, fontsize=13)
ax[0,0].set_facecolor("navajowhite")
ax[0,0].set_title("Top Ten Directors")

x_labels_dirgender=range(len(dir_gender_names))
dirgender_new=["Unset", "Female", "Male"]
ax[0,1].bar(x_labels_dirgender, dir_gender_count, facecolor="royalblue",
         edgecolor="black")
ax[0,1].set_xticks(x_labels_dirgender)
ax[0,1].set_xticklabels(dirgender_new)
ax[0,1].set_facecolor("navajowhite")
ax[0,1].set_title("Gender of Directors")


y_labels_writnames=range(len(writ_names_topten_names))
ax[1,0].barh(y_labels_writnames, writ_names_topten_count, facecolor="slateblue",
          edgecolor="black")
ax[1,0].set_yticks(y_labels_writnames)
ax[1,0].set_yticklabels(writ_names_topten_names, fontsize=13)
ax[1,0].set_facecolor("navajowhite")
ax[1,0].set_title("Top Ten Writers")

x_labels_writgender=range(len(writ_gender_names))
writgender_new=["Unset", "Female", "Male"]
ax[1,1].bar(x_labels_writgender, writ_gender_count, facecolor="royalblue",
         edgecolor="black")
ax[1,1].set_xticks(x_labels_writgender)
ax[1,1].set_xticklabels(writgender_new)
ax[1,1].set_facecolor("navajowhite")
ax[1,1].set_title("Gender of Writers")


fig.subplots_adjust(hspace=0.4, top=0.7, bottom=-0.7, wspace=0.3)

com_dir="\n".join((r"$\cdot$ " "Most popular directors are male",
                  r"$\cdot$ " "A wide difference in amount of \n" \
                  "male and female directors"))

com_writ="\n".join((r"$\cdot$ " "Similar to directors, most popular \n" \
                   "writers are male",
                   r"$\cdot$ " "A huge gender imbalance in \n" \
                   " amount of of male and female writers"))

fig.text(0.93, 0.4, com_dir, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
         fontsize=13, family="serif")
fig.text(0.93, -0.5, com_writ, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black"),
        fontsize=13, family="serif")
fig.set_facecolor("floralwhite")

