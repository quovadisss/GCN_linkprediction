import time
import requests
import pandas as pd
import numpy as np

pat_num = pd.read_csv('data/google_patent.csv')
pat_nums = [i.split('-')[1] for i in pat_num['id']]

url_post = 'http://www.patentsview.org/api/patents/query'
post_len = 10000

# Crawled elements
f = '&f=["patent_number","patent_title","patent_date", '\
'"patent_type","patent_abstract","cpc_subgroup_id", '\
'"assignee_city", "assignee_organization", '\
'"assignee_id", "cited_patent_number", "citedby_patent_number"]'

pt_dict_list = []
st = time.time()
for n, i in enumerate(pat_nums):
    q1 = '?q={"patent_number":'
    q2 = '}'
    q = q1 + '"' + i + '"' + q2

    t_data = q + f
    t_post = requests.get(url_post + t_data)

    while '500' in str(t_post):
        t_post = requests.get(url_post + t_data)
    try:
        t_json = t_post.json()
    except:
        continue

    try:
        pt_dict_list.extend(t_json['patents'])
    except TypeError:
        pass

    if n % 10 == 0:
        print(n / 10)

print('Time consuming:', time.time() - st)

# filter util patent (not design etc..)
util_patent = [i for i in pt_dict_list if i['patent_type'] == 'utility']

# cpcs ->list
for d in util_patent:
    cpc_set = ''
    for c in d['cpcs']:
        try:
            cpc = c['cpc_subgroup_id'].split('/')[0]
            cpc_set = cpc_set + ',' + cpc
        except:
            cpc_set = cpc_set + 'None'
    d['cpc_set'] = cpc_set

# dict to dataframes
df = pd.DataFrame.from_dict(util_patent).drop(['cpcs'],axis=1)
df.drop_duplicates(['patent_abstract'], inplace=True)
print('Length of final patents', len(df))

# Save to csv file
df.to_csv("data/patent.csv")