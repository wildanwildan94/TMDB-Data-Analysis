# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
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
    
## Load Training and Test Data and do some initial
## transformations


train_d=pd.read_csv('tmdb_train.csv',
                   engine="python")
test_d=pd.read_csv('tmdb_test.csv',
                  engine="python")


# (a) Fill NaN values of release_date with 1/10/00

train_d["release_date"]=train_d["release_date"].fillna("1/10/00")
test_d["release_date"]=test_d["release_date"].fillna("1/10/00")

# (b) Transform the release_date from a string into datetime object
train_d["release_date"]=[datetime.datetime.strptime(x, '%m/%d/%y') for x in train_d["release_date"]]
test_d["release_date"]=[datetime.datetime.strptime(x, '%m/%d/%y') for x in test_d["release_date"]]


# (c) Because of how datetime reads data, have to decrease some 
# datetime values by 100 years (e.g. turn 2024 into 1924)
new_release_date_train=[]

for index, row in train_d.iterrows():
  temp_date=row["release_date"]
  if temp_date.year>=2020:
    new_release_date_train.append(temp_date.replace(year=temp_date.year-100))
  else:
    new_release_date_train.append(temp_date)
    
    
train_d["release_date"]=new_release_date_train
  
  
new_release_date_test=[]
for index, row in test_d.iterrows():
  temp_date=row["release_date"]
  
  if temp_date.year>=2020:
    new_release_date_test.append(temp_date.replace(year=temp_date.year-100))
  else:
    new_release_date_test.append(temp_date)
    
test_d["release_date"]=new_release_date_test
      





# (d) Create a movie index for each movie, for convenience

train_length=train_d.shape[0]
test_length=test_d.shape[0]

train_index=np.arange(1, train_length+1)
test_index=np.arange(train_length+1, train_length+1+test_length)

train_d["movie_index"]=train_index
test_d["movie_index"]=test_index


print "---"
print "Q: How does train_d look like?"
print train_d.head(3)
print train_d.iloc[0]
print "---"

### Two functions, for the belongs_to_collection attribute

## btc_create(data): Creates a dataframe of each collection and 
## its associated average revenue, as determined from dataframe data
## btc_apply(data, data_btc): Applies the average revenue of collections, based
## on dataframe data_btc, on all movies present in data


def btc_create(data):
  coll_array=[]
  title_array=[]
  rev_array=[]

  for index, row in data.iterrows():
    temp_rev=row["revenue"]
    temp_title=row["movie_index"]

    try:
      temp_colls=ast.literal_eval(row["belongs_to_collection"])
    
      if len(temp_colls)==0:
        coll_array.append("unknown")
        rev_array.append(temp_rev)
        title_array.append(temp_title)
      for item in temp_colls:
        temp_name=item["name"]
        temp_name=strip_accents(temp_name.replace('\xa0', ''))
        temp_name=strip_accents(temp_name)
        coll_array.append(temp_name)
        title_array.append(temp_title)
        rev_array.append(temp_rev)
    except:
      coll_array.append("unknown")
      rev_array.append(temp_rev)
      title_array.append(temp_title)
    
    
  title_collname_d=pd.DataFrame({'movie_index':title_array,
                              'collection_name':coll_array})
  collname_rev_d=pd.DataFrame({'collection_name':coll_array,
                            'revenue':rev_array})

  collname_avgrev_d=collname_rev_d.groupby("collection_name").agg({'revenue':'mean'}).rename(columns={'revenue':'collection_revenue_mean'}).reset_index()

  
  return collname_avgrev_d


def btc_apply(data, data_btc):
  
  
  
  coll_array=[]
  titleindex_array=[]
  title_array=[]
  
  for index, row in data.iterrows():
    temp_titleindex=row["movie_index"]
    temp_title=row["original_title"]

    try:
      temp_colls=ast.literal_eval(row["belongs_to_collection"])
      if len(temp_colls)==0:
        coll_array.append("unknown")
        title_array.append(temp_title)
        titleindex_array.append(temp_titlelindex)
    
      for item in temp_colls:
        temp_name=item["name"]
        temp_name=strip_accents(temp_name.replace('\xa0', ''))
        temp_name=strip_accents(temp_name)
        coll_array.append(temp_name)
        title_array.append(temp_title)
        titleindex_array.append(temp_titleindex)
    except:
      coll_array.append("unknown")
      title_array.append(temp_title)
      titleindex_array.append(temp_titleindex)
    
    
  title_collname_d=pd.DataFrame({'movie_index':titleindex_array,
                                 'original_title':title_array,
                              'collection_name':coll_array})
  
  

  coll_avgrev=np.mean(data_btc["collection_revenue_mean"])
  title_collname_avgrev_d=title_collname_d.merge(data_btc, on="collection_name", how="left")
  
  # Fill NaN with mean values
  title_collname_avgrev_d["collection_revenue_mean"]=title_collname_avgrev_d["collection_revenue_mean"].fillna(coll_avgrev)

  return title_collname_avgrev_d
  


### Two functions, for the genres attribute

## genres_create(data): Creates a dataframe of each genre and 
## its associated average revenue, as determined from dataframe data
## genres_apply(data, data_genres): Applies the average revenue of genre, based
## on dataframe data_genres, on all movies present in data




def genres_create(data):
  

  date_gen_rev_d=data[["release_date", "genres", "revenue", "movie_index"]]


  # Creates two attributes; month which is the month, and month_index,
  # which is an index for the month
  date_gen_rev_d["month"]=[x.strftime("%B") for x in date_gen_rev_d["release_date"]]
  date_gen_rev_d["month_index"]=[int(x.strftime("%-m")) for x in date_gen_rev_d["release_date"]]



  genre_name_movies=[]
  month_movies=[]
  month_index_movies=[]
  revenue_movies=[]
  titleindex_movies=[]
  for index, row in date_gen_rev_d.iterrows():
  
    temp_month=row["month"]
    temp_month_index=row["month_index"]
    temp_revenue=row["revenue"]
    temp_titleindex=row["movie_index"]
    try:
      temp_genres=ast.literal_eval(row["genres"])
      if len(temp_genres)==0:
        genre_name_movies.append("unknown")
        month_movies.append(temp_month)
        month_index_movies.append(temp_month_index)
        revenue_movies.append(temp_revenue)
        titleindex_movies.append(temp_titleindex)
      for genre in temp_genres:
        genre_name_movies.append(genre["name"])
        month_movies.append(temp_month)
        month_index_movies.append(temp_month_index)
        revenue_movies.append(temp_revenue)
        titleindex_movies.append(temp_titleindex)
    except:
      genre_name_movies.append("unknown")
      month_movies.append(temp_month)
      month_index_movies.append(temp_month_index)
      revenue_movies.append(temp_revenue)
      titleindex_movies.append(temp_titleindex)
    


  genre_rev_d=pd.DataFrame({'genre_name':genre_name_movies,
                         'revenue':revenue_movies})
  genre_avgrev_d=genre_rev_d.groupby("genre_name").agg({'revenue':'mean'}).rename(columns={'revenue':'genre_revenue_mean_overall'}).reset_index()

  genre_rev_month_monthi_d=pd.DataFrame({'genre_name':genre_name_movies,
                                      'revenue':revenue_movies,
                                      'month':month_movies,
                                      'month_index':month_index_movies})
  
  genre_avgrev_month_monthi_d=genre_rev_month_monthi_d.groupby(["genre_name", "month"]).agg({'revenue':'mean',
                                                                                          'month_index':'first'}).rename(columns={'revenue':'genre_revenue_mean_monthly'}).reset_index()
  
  
  
  genre_avgrev_d=genre_avgrev_month_monthi_d.merge(genre_avgrev_d, on="genre_name", how="left")
  
  return genre_avgrev_d
  
 
def genres_apply(data, data_genres):
  
  date_gen_rev_d=data[["release_date", "original_title","genres", "movie_index"]]

  
  # Creates two attributes; month which is the month, and month_index,
  # which is an index for the month
  date_gen_rev_d["month"]=[x.strftime("%B") for x in date_gen_rev_d["release_date"]]
  date_gen_rev_d["month_index"]=[int(x.strftime("%-m")) for x in date_gen_rev_d["release_date"]]


  genre_name_movies=[]
  month_movies=[]
  month_index_movies=[]
  titleindex_movies=[]
  title_movies=[]
  for index, row in date_gen_rev_d.iterrows():
  
    temp_month=row["month"]
    temp_month_index=row["month_index"]
    temp_titleindex=row["movie_index"]
    temp_title=row["original_title"]
    try:
      temp_genres=ast.literal_eval(row["genres"])
      if len(temp_genres)==0:
        genre_name_movies.append("unknown")
        month_movies.append(temp_month)
        month_index_movies.append(temp_month_index)
        titleindex_movies.append(temp_titlelindex)
        title_movies.append(temp_title)
      for genre in temp_genres:
        genre_name_movies.append(genre["name"])
        month_movies.append(temp_month)
        month_index_movies.append(temp_month_index)
        titleindex_movies.append(temp_titleindex)
        title_movies.append(temp_title)
    except:
      genre_name_movies.append("unknown")
      month_movies.append(temp_month)
      month_index_movies.append(temp_month_index)
      titleindex_movies.append(temp_titleindex)
      title_movies.append(temp_title)
    

    
    
  title_genre_month_d=pd.DataFrame({'movie_index':titleindex_movies,
                                    'original_title':title_movies,
                                   'genre_name':genre_name_movies,
                                   'month_index':month_index_movies})

  

  title_genre_month_revdata_d=title_genre_month_d.merge(data_genres, on=["genre_name", "month_index"], how="left")[["original_title", "movie_index",
                                                                                                        "month_index",
                                                                                                       "genre_revenue_mean_monthly",
                                                                                                       "genre_revenue_mean_overall"]]
  
  
  # Want to fill na
  genre_revenue_mean_overall_mean=np.mean(title_genre_month_revdata_d["genre_revenue_mean_overall"])
  title_genre_month_revdata_d["genre_revenue_mean_monthly"]=title_genre_month_revdata_d.groupby("month_index")["genre_revenue_mean_monthly"].apply(lambda x:x.fillna(x.mean()))
  title_genre_month_revdata_d["genre_revenue_mean_overall"]=title_genre_month_revdata_d["genre_revenue_mean_overall"].fillna(genre_revenue_mean_overall_mean)
  
  
  title_genre_month_avgrevdata_d=title_genre_month_revdata_d.groupby("movie_index").agg({'genre_revenue_mean_monthly':'mean',
                                                                                        'genre_revenue_mean_overall':'mean',
                                                                                        'original_title':'first'}).reset_index()
  
  

  
  
  return title_genre_month_avgrevdata_d

  
  



    ### Two functions, for the original_language attribute

## origlang_create(data): Creates a dataframe of each original_language and 
## its associated average revenue, as determined from dataframe data
## origlang_apply(data, data_origlang): Applies the average revenue of original_language,
## based on dataframe data_origlang, on all movies present in data


def origlang_create(data):
  
  olang_title_rev_d=data[["original_language", "original_title", "revenue"]]
  olang_title_rev_d["original_language"]=olang_title_rev_d["original_language"].fillna(value="unknown")
  
  avgrev_olangs_d=olang_title_rev_d.groupby("original_language").agg({'revenue':'mean'}).rename(columns={'revenue':'origlang_revenue_mean'}).reset_index()

  return avgrev_olangs_d
  

def origlang_apply(data, data_origlang):
  
  olang_title_rev_d=data[["original_language", "original_title", "movie_index"]]
  olang_title_rev_d["original_language"]=olang_title_rev_d["original_language"].fillna(value="unknown")
  
  
  olang_title_rev_avgrev_d=olang_title_rev_d.merge(data_origlang, on="original_language", how="left")
  
  # Fill NaN in case of new original_languages
  olang_title_rev_avgrev_d["origlang_revenue_mean"]=olang_title_rev_avgrev_d["origlang_revenue_mean"].fillna(np.mean(olang_title_rev_avgrev_d["origlang_revenue_mean"]))
  

  return olang_title_rev_avgrev_d




### Two functions, for the production companies attribute

## prod_create(data): Creates a dataframe of each production company and 
## its associated average revenue, as determined from dataframe data
## prod_apply(data, data_prod): Applies the average revenue of production companies,
## based on dataframe data_prod, on all movies present in data

def prod_create(data):
  
  prod_revenue_d=data[["production_companies", "revenue"]]
  
  
  prods_array=[]
  revenue_array=[]
  for index, row in prod_revenue_d.iterrows():
    temp_revenue=row["revenue"]
    try:
      prod_list=ast.literal_eval(row["production_companies"])
      if len(prod_list)==0:
        prods_array.append("unknown")
        revenue_array.append(temp_revenue)
      for prod in prod_list:
        prods_array.append(prod["name"])
        revenue_array.append(temp_revenue)
    except:
      prods_array.append("unknown")
      revenue_array.append(temp_revenue)
      
      
 
  prod_rev_d=pd.DataFrame({'prod_name':prods_array,
                          'revenue':revenue_array})

  prod_avgrev_d=prod_rev_d.groupby("prod_name").agg({'revenue':'mean'}).rename(columns={'revenue':'revenue_mean'}).reset_index()
  
  return prod_avgrev_d
    
  
  
  
  
def prod_apply(data, data_prod):

  prod_revenue_d=data[["original_title", "movie_index","production_companies"]]
  prods_array=[]
  titleindex_array=[]
  title_array=[]
  for index, row in prod_revenue_d.iterrows():
    temp_titleindex=row["movie_index"]
    temp_title=row["original_title"]
    try:
      prod_list=ast.literal_eval(row["production_companies"])
      if len(prod_list)==0:
        prods_array.append("unknown")
        titleindex_array.append(temp_titleindex)
        title_array.append(temp_title)
      for prod in prod_list:
        prods_array.append(prod["name"])
        titleindex_array.append(temp_titleindex)
        title_array.append(temp_title)
    except:
      prods_array.append("unknown")
      titleindex_array.append(temp_titleindex)
      title_array.append(temp_title)
    
    

  title_prod_d=pd.DataFrame({'original_title':title_array,
                             'movie_index':titleindex_array,
                            'prod_name':prods_array})

 
  title_prod_rev_d=title_prod_d.merge(data_prod, on="prod_name", how="left")
  
  # Fill NaN for movies e.g. new production companies
  title_prod_rev_d["revenue_mean"]=title_prod_rev_d["revenue_mean"].fillna(np.mean(data_prod["revenue_mean"]))
  
  
  
 
 
  title_prod_avgrev_d=title_prod_rev_d.groupby("movie_index").agg({'original_title':'first',
                                                                  'revenue_mean':'mean'}).rename(columns={'revenue_mean':'prod_revenue_mean'}).reset_index()
  
  return title_prod_avgrev_d
 


### Two functions, for the keyword attribute

## keyword_create(data): Creates a dataframe of each keyword and 
## its associated average revenue, as determined from dataframe data
## keyword_apply(data, data_keyword): Applies the average revenue of keyword,
## based on dataframe data_prod, on all movies present in data

def keyword_create(data):
  
  
  keyw_revenue_d=data[["Keywords", "revenue", "original_title"]]

  movie_words=[]
  word_revenue=[]

  for index, row in keyw_revenue_d.iterrows():
    temp_revenue=row["revenue"]
    try:
      temp_keywords=ast.literal_eval(row["Keywords"])
      if len(temp_keywords)==0:
        movie_words.append("unknown")
        word_revenue.append(temp_revenue)
      for keyword in temp_keywords:
        movie_words.append(strip_accents(keyword["name"].replace('\xa0', '')))
        word_revenue.append(temp_revenue)
    except:
      movie_words.append("unknown")
      word_revenue.append(temp_revenue)
    
    
  
  keyword_revenue_d=pd.DataFrame({'keyword':movie_words,
                                 'revenue':word_revenue})
  
  keyword_avgrev_d=keyword_revenue_d.groupby("keyword").agg({'revenue':'mean'}).rename(columns={'revenue':'keyword_revenue_mean'}).reset_index()


  
  return keyword_avgrev_d
  
  

def keyword_apply(data, data_keyword):

  keyw_revenue_d=data[["Keywords", "original_title", "movie_index"]]
  
  movie_titles=[]
  movie_words=[]
  movie_index=[]

  for index, row in keyw_revenue_d.iterrows():
    temp_title=row["original_title"]
    temp_titleindex=row["movie_index"]
    try:
      temp_keywords=ast.literal_eval(row["Keywords"])
      if len(temp_keywords)==0:
        movie_titles.append(temp_titl)
        movie_words.append("unknown")
        movie_index.append(temp_titleindex)
      for keyword in temp_keywords:
        movie_words.append(strip_accents(keyword["name"].replace('\xa0', '')))
        movie_titles.append(temp_title)
        movie_index.append(temp_titleindex)
    except:
      movie_titles.append(temp_title)
      movie_words.append("unknown")
      movie_index.append(temp_titleindex)
    

 
  title_keyword_d=pd.DataFrame({'original_title':movie_titles,
                                'movie_index':movie_index,
                                'keyword':movie_words})
  
  rev_keyword_mean=np.mean(data_keyword["keyword_revenue_mean"])
  title_keyword_rev_d=title_keyword_d.merge(data_keyword, on="keyword", how="left")
  
  # Fill NaN with average revenue, i.e. if there are new keywords
  title_keyword_rev_d["keyword_revenue_mean"]=title_keyword_rev_d["keyword_revenue_mean"].fillna(rev_keyword_mean)
  
  title_keyword_avgrev_d=title_keyword_rev_d.groupby("movie_index").agg({'original_title':'first',
                                                                        'keyword_revenue_mean':'mean'}).reset_index()

  return title_keyword_avgrev_d
                             

### Two functions, for the cast attribute

## cast_create(data): Creates a dataframe of each cast member and 
## its associated average revenue, as determined from dataframe data
## cast_apply(data, data_cast): Applies the average revenue of cast members,
## based on dataframe data_cast, on all movies present in data

def cast_create(data):
  
  cast_revenue_d=data[["cast", "revenue"]]


  cast_array=[]
  revenue_array=[]
  for index, row  in cast_revenue_d.iterrows():
    temp_revenue=row["revenue"]
    try:
      list_actors=ast.literal_eval(row["cast"])
      if len(list_actors)==0:
        cast_array.append("unknown")
        revenue_array.append(temp_revenue)
      for act in list_actors:
        cast_array.append(strip_accents(act["name"].replace('\xa0','')))
        revenue_array.append(temp_revenue)
    except:
      cast_array.append("unknown")
      revenue_array.append(temp_revenue)
   

  member_rev_d=pd.DataFrame({'cast_member':cast_array,
                            'revenue':revenue_array})
  member_avgrev_d=member_rev_d.groupby("cast_member").agg({'revenue':'mean'}).rename(columns={'revenue':'cast_revenue_mean'}).reset_index()

  return member_avgrev_d
  

def cast_apply(data, data_cast):
  cast_revenue_d=data[["cast", "original_title", "movie_index"]]


  cast_array=[]
  title_array=[]
  titleindex_array=[]
  for index, row  in cast_revenue_d.iterrows():
    temp_title=row["original_title"]
    temp_titleindex=row["movie_index"]
    try:
      list_actors=ast.literal_eval(row["cast"])
      if len(list_actors)==0:
        cast_array.append("unknown")
        title_array.append(temp_title)
        titleindex_array.append(temp_titleindex)
      for act in list_actors:
        cast_array.append(strip_accents(act["name"].replace('\xa0','')))
        title_array.append(temp_title)
        titleindex_array.append(temp_titleindex)
    except:
      cast_array.append("unknown")
      title_array.append(temp_title)
      titleindex_array.append(temp_titleindex)
    
  title_member_d=pd.DataFrame({'original_title':title_array,
                               'movie_index':titleindex_array,
                              'cast_member':cast_array})
 
  
  
  title_member_rev_d=title_member_d.merge(data_cast, on="cast_member", how="left")
  cast_revenue_mean=np.mean(data_cast["cast_revenue_mean"])
  
  # Fill NaN with mean, e.g. if there are new cast members
  title_member_rev_d["cast_revenue_mean"]=title_member_rev_d["cast_revenue_mean"].fillna(cast_revenue_mean)
  
  
  title_member_avgrev_d=title_member_rev_d.groupby("movie_index").agg({'original_title':'first',
                                                                      'cast_revenue_mean':'mean'}).reset_index()
  
  
  return title_member_avgrev_d
  
  

### Two functions, for the crew attribute

## crew_create(data, jobs): Creates a dataframe of each Crew Job Type in 
## list jobs and its associated average revenue, as determined from dataframe data
## crew_apply(data, data_crew, job): Applies the average revenue of a Crew Job Type
## as determined by string jobs,
## based on dataframe data_cast, on all movies present in data



def crew_create(data, jobs):

  crew_title_revenue_d=data[["crew", "revenue"]]


  jobs_array=[]
  name_array=[]
  revenue_array=[]
  
  for index, row in crew_title_revenue_d.iterrows():
    temp_revenue=row["revenue"]
    try:
      temp_crew=ast.literal_eval(row["crew"])
      if len(temp_crew)==0:
        jobs_array.append("unknown")
        name_array.append("unknown")
        revenue_array.append(temp_revenue)
      for member in temp_crew:
        temp_job=member["job"]
        if len(temp_job)==0:
          jobs_array.append("unknown")
          name_array.append("unknown")
          revenue_array.append(temp_revenue)
        if temp_job in jobs:
          jobs_array.append(temp_job)
          name_array.append(strip_accents(member["name"]).replace('\xa0',''))
          revenue_array.append(temp_revenue)
    except Exception as e:    
      jobs_array.append("unknown")
      name_array.append("unknown")
      revenue_array.append(temp_revenue)
    
    

  job_crewm_rev_d=pd.DataFrame({'job':jobs_array,
                               'crew_member':name_array,
                               'revenue':revenue_array})

  job_crewm_avgrev_d=job_crewm_rev_d.groupby(["job", "crew_member"]).agg({'revenue':'mean'}).rename(columns={'revenue':'crew_revenue_mean'}).reset_index()
  
  
  return job_crewm_avgrev_d

    
def crew_apply(data, data_crew, job):
  
  crew_title_revenue_d=data[["crew", "original_title", "movie_index"]]


  jobs_array=[]
  name_array=[]
  title_array=[]
  titleindex_array=[]
  
  
  for index, row in crew_title_revenue_d.iterrows():
    temp_title=row["original_title"]
    temp_titleindex=row["movie_index"]
    
    temp_jobs_array=[]
    temp_name_array=[]
    temp_title_array=[]
    temp_titleindex_array=[]
    
    try:
      temp_crew=ast.literal_eval(row["crew"])
        
      for member in temp_crew:
        temp_job=member["job"]
        if temp_job==job:
          temp_jobs_array.append(temp_job)
          temp_name_array.append(strip_accents(member["name"]).replace('\xa0',''))
          temp_title_array.append(temp_title)
          temp_titleindex_array.append(temp_titleindex)
    except:
      temp_jobs_array.append(job)
      temp_name_array.append("unknown")
      temp_title_array.append(temp_title)
      temp_titleindex_array.append(temp_titleindex)
      
    if len(temp_jobs_array)==0:
      temp_jobs_array.append(job)
      temp_name_array.append(strip_accents(member["name"]).replace('\xa0',''))
      temp_title_array.append(temp_title)
      temp_titleindex_array.append(temp_titleindex)
    jobs_array.extend(temp_jobs_array)
    name_array.extend(temp_name_array)
    title_array.extend(temp_title_array)
    titleindex_array.extend(temp_titleindex_array)
          
  
    
  

 
  title_job_crewm_d=pd.DataFrame({'original_title':title_array,
                                  'movie_index':titleindex_array,
                                 'job':jobs_array,
                                 'crew_member':name_array})
  

  title_job_crewm_rev_d=title_job_crewm_d.merge(data_crew, on=["job", "crew_member"], how="left")
  
  
  # Fill NaN by a simple mean
  title_job_crewm_rev_d.crew_revenue_mean=title_job_crewm_rev_d["crew_revenue_mean"].fillna(np.mean(title_job_crewm_rev_d["crew_revenue_mean"]))
  
  
  
  title_job_crewm_avgrev_d=title_job_crewm_rev_d.groupby(["movie_index", "job"]).agg({'crew_revenue_mean':'mean',
                                                                                     'original_title':'first'}).rename(columns={'crew_revenue_mean':'%s_revenue_mean'%job.lower()}).reset_index()
  
  
  return title_job_crewm_avgrev_d.drop("job", axis=1)
  
 


### Functions that construct train and test data with the relevant columns

def construct_adjusted_data(train_data, test_data):
  
  train_Xshort_d=train_data[["movie_index","budget", "popularity", "runtime", "revenue"]]
  test_Xshort_d=test_data[["movie_index", "budget", "popularity", "runtime", "revenue"]]
  
  
  
  ## belongs_to_collection

  btc_created=btc_create(train_data)
  btc_applied_train=btc_apply(train_data, btc_created)
  btc_applied_test=btc_apply(test_data, btc_created)

  
  print "---"
  print "btc sizes"
  print btc_applied_train.shape
  print btc_applied_test.shape
  print "---"
  # Merge btc with train and test

  train_Xshort_d=train_Xshort_d.merge(btc_applied_train[["movie_index", "collection_revenue_mean"]], on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(btc_applied_test[["movie_index", "collection_revenue_mean"]], on="movie_index")
  print "---"
  print "train % test sizes after btc merge"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"

  ## genres

  genres_created=genres_create(train_data)
  genres_applied_train=genres_apply(train_data, genres_created)
  genres_applied_test=genres_apply(test_data, genres_created)
  
  print "---"
  print "genres sizes"
  print genres_applied_train.shape
  print genres_applied_test.shape
  print "---"



  train_Xshort_d=train_Xshort_d.merge(genres_applied_train[["movie_index", "genre_revenue_mean_monthly", "genre_revenue_mean_overall"]],
                                     on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(genres_applied_test[["movie_index", "genre_revenue_mean_monthly", "genre_revenue_mean_overall"]],
                                     on="movie_index")
  
  print "---"
  print "train & test sizes after genres"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"


  ## orig_lang

  origlang_created=origlang_create(train_data)
  origlang_applied_train=origlang_apply(train_data, origlang_created)
  origlang_applied_test=origlang_apply(test_data, origlang_created)
  
  print "---"
  print "orig lang sizes"
  print origlang_applied_train.shape
  print origlang_applied_test.shape
  print "---"


  train_Xshort_d=train_Xshort_d.merge(origlang_applied_train[["movie_index", "origlang_revenue_mean"]], on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(origlang_applied_test[["movie_index", "origlang_revenue_mean"]], on="movie_index")

  
  print "---"
  print "train % test sizes after orig lang"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"


  ## Production company

  prod_created=prod_create(train_data)
  prod_applied_train=prod_apply(train_data, prod_created)
  prod_applied_test=prod_apply(test_data, prod_created)


  print "---"
  print "prod sizes"
  print prod_applied_train.shape
  print prod_applied_test.shape
  print "---"

  train_Xshort_d=train_Xshort_d.merge(prod_applied_train[["movie_index", "prod_revenue_mean"]], on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(prod_applied_test[["movie_index", "prod_revenue_mean"]], on="movie_index")
  
  print "---"
  print "train and test sizes after prod"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"

  keyword_created=keyword_create(train_data)
  keyword_applied_train=keyword_apply(train_data, keyword_created)
  keyword_applied_test=keyword_apply(test_data, keyword_created)
  print "---"
  print "keyword sizes"
  print keyword_applied_train.shape
  print keyword_applied_test.shape
  print "---"
  
  train_Xshort_d=train_Xshort_d.merge(keyword_applied_train[["movie_index", "keyword_revenue_mean"]], on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(keyword_applied_test[["movie_index", "keyword_revenue_mean"]], on="movie_index")
  
  print "---"
  print "train and test sizes after keyword"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"

  cast_created=cast_create(train_data)
  cast_applied_train=cast_apply(train_data, cast_created)
  cast_applied_test=cast_apply(test_data, cast_created)
  
  print "---"
  print "cast sizes"
  print cast_applied_train.shape
  print cast_applied_test.shape
  print "---"

  train_Xshort_d=train_Xshort_d.merge(cast_applied_train[["movie_index", "cast_revenue_mean"]], on="movie_index")
  test_Xshort_d=test_Xshort_d.merge(cast_applied_test[["movie_index", "cast_revenue_mean"]], on="movie_index")
  
  print "---"
  print "train and test sizes after cast"
  print train_Xshort_d.shape
  print test_Xshort_d.shape
  print "---"


  jobs=["Producer", "Executive Producer", "Director", "Screenplay", "Director of Photography", "Original Music Composer", "Writer"]
  crew_created=crew_create(train_data, jobs)
  for job in jobs:
    print "---"
    print "Currently Iterating for job: %s"%job
    temp_crew_applied_train=crew_apply(train_data, crew_created, job)
    temp_crew_applied_test=crew_apply(test_data, crew_created, job)
    
    print "---"
    print "job sizes for: %s"%job
    print temp_crew_applied_train.shape
    print temp_crew_applied_test.shape
    print "---"
  
    train_Xshort_d=train_Xshort_d.merge(temp_crew_applied_train[["movie_index",
                                                                 "%s_revenue_mean"%job.lower()]], on="movie_index")
    test_Xshort_d=test_Xshort_d.merge(temp_crew_applied_test[["movie_index",
                                                             "%s_revenue_mean"%job.lower()]], on="movie_index")
    print "---"
    print "train and test sizes after job: %s"%job
    print train_Xshort_d.shape
    print test_Xshort_d.shape
    print "---"
    
    print "Finished Iterating for job: %s"%job
    print "---"
    
  return train_Xshort_d, test_Xshort_d
  
  


  
  
  ### Construct adjusted data for data sets:

## Construction of data for (1)
"""
X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
"""

train_split, test_split=train_test_split(train_d, test_size=0.1, random_state=10)


print "---"
print "Starting to Construct Adjusted Data for Full Data"
train_adj_d, test_adj_d=construct_adjusted_data(train_split, test_split)
print "Finished Constructing Adjusted Data for Full Data"
print "---"


train_adj_d.to_csv('train_adj_r10_t01_d.csv', index=False)
test_adj_d.to_csv('test_adj_r10_t01_d.csv', index=False)


