import random
import pandas as pd
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize




# Delete data which has no cpc_set
def delete_null(table):
    print('Initial length of the raw data is', table.shape[0])
    df_ = table.copy()
    for i in range(df_.shape[0]):
        if df_['cpc_set'][i] == 'None':
            df_.drop(i, inplace=True)
    
    print('Final length of the data is', df_.shape[0])
    return df_


# Split data into train and test
def split_train_test(table):
    table_ = delete_null(table)
    train_range = [str(i) for i in range(2015, 2020)]
    tr_df = table_.loc[table['patent_date'].apply(lambda x: x.split('-')[0]).isin(train_range)]
    ts_df = table_.loc[table['patent_date'].apply(lambda x: x.split('-')[0]) == '2020']
    
    return tr_df, ts_df


# cpc_set to symmetric adjacency matrix with train and test each.
def create_adj(table):
    """
    A hat = D tilde to the power of -1/2 * A tilde * D tilde to the power of -1/2
    A hat is a symmetric adjacency matrix for GCN layers.
    """

    # A tilde : Initial adjacency matrix
    cpcs = [' '.join(i.split(',')[1:]) for i in table['cpc_set']]
    vectorizer = CountVectorizer()
    mode_2 = vectorizer.fit_transform(cpcs)
    cpc_order = vectorizer.get_feature_names()
    cpc_order = [i.upper() for i in cpc_order]
    mode_2 = sp.csr_matrix(mode_2.toarray())
    adj = mode_2.T.dot(mode_2).toarray()
    for e1, i in enumerate(adj):
        for e2, j in enumerate(i):
            if j != 0:
                adj[e1][e2] = 1

    # D tilde to the power of -1/2 : Degree matrix from A tilde
    # dg = []
    # for i in adj:
    #     rowsum = np.sum(i)
    #     r_inv = np.power(rowsum, -0.5)
    #     dg.append(r_inv)
    # dg = np.diag(np.array(dg))

    # # A hat
    # adj = np.dot(dg, adj)
    # adj = np.dot(adj, dg)

    # return sparse_to_tuple(sp.csr_matrix(adj))
    return sp.csr_matrix(adj), cpc_order


def normalize_adj(adj):
    # D tilde to the power of -1/2 : Degree matrix from A tilde
    dg = []
    for i in adj:
        rowsum = np.sum(i)
        r_inv = np.power(rowsum, -0.5)
        dg.append(r_inv)
    dg = np.diag(np.array(dg))

    # A hat
    adj = np.dot(dg, adj)
    adj = np.dot(adj, dg)

    return adj


# Convert sparse matrix to tuple representation
def sparse_to_tuple(sparse_mx):

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def split_trian_valid(adj):
    random.seed(100)
    adj = sparse_to_tuple(sp.csr_matrix(adj))
    pos_ind = [e for e, i in enumerate(adj[1]) if i == 1]
    neg_ind = [e for e, i in enumerate(adj[1]) if i == 0]
    pos_val_ind = random.sample(pos_ind, int(len(pos_ind)*0.1))
    pos_tr_ind = list(set(pos_ind) - set(pos_val_ind))
    neg_val_ind = random.sample(neg_ind, len(pos_val_ind))
    neg_without_val = list(set(neg_ind) - set(neg_val_ind))
    neg_tr_ind = random.sample(neg_without_val, len(pos_tr_ind))
    tr_ind = pos_tr_ind.extend(neg_tr_ind)
    val_ind = pos_val_ind.extend(neg_val_ind)

    tr_pairs = [adj[0][i] for i in tr_ind]
    val_pairs = [adj[0][i] for i in val_ind]

    df = pd.DataFrame(index=np.unique(tr_pairs), columns=np.unique(tr_pairs))


def stopwords_lemma(txt):
    user_stopwords = stopwords.words('english')  # from nltk.corpus import stopwords
    lemma_words = []
    lemma = WordNetLemmatizer()
    token_words = word_tokenize(txt)
    for w in token_words:
        lem = lemma.lemmatize(w)
        if lem not in user_stopwords:
            lemma_words.append(lem)
        else:
            continue

    lemma_result = " ".join(lemma_words)
    return lemma_result