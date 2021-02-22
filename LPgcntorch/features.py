import datetime
import time
import re
import numpy as np
import pandas as pd
import pickle as pkl

from utils import *
from selenium import webdriver
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm.notebook import tqdm

"""
Get feature matrix: X for GCN layer
1. growing rate
2. Frequency in the network
3. Frequency in the whole patent
4. Relation with the Domain
5. CPC'name vector representation
"""


# 1. Growing rate, 2. Frequency in the network
def feature_one_two(df, unique_cpcs):
    table = df.copy()
    cpcs = [i.split(',')[1:] for i in table['cpc_set']]
    table['datetime'] = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in table['patent_date']]
    start = datetime.date(2015, 1, 1)
    end = datetime.date(2019, 12, 31)
    date_range = int((end-start).days / 5)
    range_cpcs = []
    for i in range(5):
        range_cpc = []
        cri = start+datetime.timedelta(days=date_range)
        for d, c in zip(table['datetime'], cpcs):
            if d == start or d > start:
                if d < cri:
                    range_cpc.extend(c)
        start += datetime.timedelta(days=date_range)
        count_cpc = Counter(range_cpc)
        range_cpcs.append(count_cpc)

    all_cpc = []
    for i in cpcs:
        all_cpc.extend(i)
    all_cpc = Counter(all_cpc)

    # Count cpc in the network
    just_count = []
    for i in unique_cpcs:
        just_count.append(all_cpc[i])

    # Calculate linear coefficient
    coeffi = []
    x = np.arange(5).reshape(-1, 1)
    for i in unique_cpcs:
        count_y = []
        for j in range_cpcs:
            if j[i] == 0:
                count_y.append(1)
            else:
                count_y.append(j[i]+1)
        y = np.array(count_y).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        coeffi.append(np.round(reg.coef_[0][0], 3))

    return coeffi, just_count


# 3. Frequency in the whole patent, 4. Relation with the Domain
def feature_three_four(unique_cpcs, driver_loc):
    # Frequency
    freq = []
    driver = webdriver.Chrome(driver_loc)
    for e, i in enumerate(unique_cpcs):
        driver.get('https://patents.google.com/?q={}&country=US&after=priority:20180201'.format(i))
        while len(freq) == e:
            try:
                css = driver.find_elements_by_css_selector('span.style-scope.search-results')
                num = int(re.sub('[^0-9]', '', css[-1].text))
                freq.append(num)
            except ValueError:
                freq.append(0)
            except IndexError:
                pass

    # Domain relation
    relate_domain = []
    domain = ['G06Q10', 'G06Q20', 'G06Q30', 'G06Q40', 'G06Q50', 'G06Q90', 'G06Q99', 'G06Q2220']
    for i in unique_cpcs:
        if i in domain:
            relate_domain.append(1)
        else:
            relate_domain.append(0)
    
    return freq, relate_domain


# Doc2Vec
def tagging_document(txt):
    tagged_data = []
    for i, d in enumerate(txt):
        tagged_data.append(TaggedDocument(words=word_tokenize(d), tags=[str(i)]))
    
    max_epochs = 100
    vec_size = 100
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,  # size > 100 recommended
                    alpha=alpha,
                    min_alpha=0.00025,
                    window=5,
                    min_count=5,  # minimum count of word. 10~100 would be great!
                    dm=1)  # dm=1 is PV-DM, dm=0 is PV-DBOW

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    return model


# 5. CPC'name vector representation
def feature_five(unique_cpcs, driver_loc):
    # Get all description from USPTO site
    only_first_4 = []
    for u in unique_cpcs:
        only_first_4.append(u[:4])
    only_first_4 = np.unique(only_first_4)

    driver = webdriver.Chrome(driver_loc)
    css_desc_all = []
    css_cpc_all = []
    for u in only_first_4:
        css_desc = []
        driver.get('https://www.uspto.gov/web/patents/classification/cpc/html/cpc-{0}.html#{1}'.format(u, u))
        while len(css_desc) == 0:
            css_desc = driver.find_elements_by_css_selector('div.class-title')
            desc = [i.text for i in css_desc]
            css_cpc = driver.find_elements_by_css_selector('span.symbol')
            cpc = [i.text.split('/') for i in css_cpc]
        css_desc_all.append(desc)
        css_cpc_all.append(cpc)

    cl = only_first_4[0]
    ind = 0
    desc_all = []
    for u in unique_cpcs:
        if cl != u[:4]:
            ind += 1
            cl = u[:4]
        txt = ''
        txt += css_desc_all[ind][0]
        for c, d in zip(css_cpc_all[ind], css_desc_all[ind]):
            if len(c) > 1:
                c[0] = ''.join(c[0].split(' '))
                if u == c[0] and c[1] == '00':
                    txt += ' '
                    txt += d
        txt = re.sub('[-=+,#/\?:;“”^$@*\"※~&%ㆍ{!}』\\‘|\(\)\[\]\<\>`\'…》]', '', txt)
        txt = re.sub('[0-9]', '', txt)
        txt = txt.replace('.', '')
        txt = txt.replace('  ', ' ')
        txt = txt.replace('   ', ' ')
        shortword = re.compile(r'\W*\b\w{1,2}\b')
        txt = shortword.sub('', txt)
        lemm_txt = stopwords_lemma(txt).lower()
        desc_all.append(lemm_txt)

    model = tagging_document(desc_all)
    # Extract Doc2Vec vectors
    doc_vectors = []
    for i in range(len(desc_all)):
        doc_vectors.append(model.docvecs[i])

    return pd.DataFrame(doc_vectors)


def features_by_set(table, driver_loc, tr=True):
    if tr == True:
        with open(data_loc + 'orders.pkl', 'rb') as fr:
            unique_cpcs = pkl.load(fr)[0]
    else:
        with open(data_loc + 'orders.pkl', 'rb') as fr:
            unique_cpcs = pkl.load(fr)[1]

    feat_1, feat_2 = feature_one_two(table, unique_cpcs)
    feat_3, feat_4 = feature_three_four(unique_cpcs, driver_loc)
    feat_5 = feature_five(unique_cpcs, driver_loc)

    feature_df = pd.DataFrame([feat_1, feat_2, feat_3, feat_4]).T
    feature_df = pd.concat([feature_df, feat_5], axis=1)

    return feature_df


data_loc = '/Users/mingyupark/spyder/GCN_linkprediction/data/'
driver_loc = '/Users/mingyupark/spyder/chromedriver'
df = pd.read_csv(data_loc + 'patent.csv').iloc[:, 1:]
tr_df, ts_df = split_train_test(df)

feature_df = features_by_set(tr_df, driver_loc, True)

with open(data_loc + 'features.pkl', 'wb') as fw:
    pkl.dump(np.array(feature_df), fw)


