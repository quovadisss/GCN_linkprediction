import random
import pandas as pd
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Delete data which has no cpc_set
def delete_null(table):
    print('Initial length of the raw data is', table.shape[0])
    table_ = table.copy()
    for i in range(table_.shape[0]):
        if table_['cpc_set'][i] == 'None':
            table_.drop(i, inplace=True)
    
    print('Final length of the data is', table_.shape[0])
    return table_


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def make_ind_val(links):
    a_indices = []
    a_values = [1] * len(links)
    b_indices = []
    b_values = [1] * len(links)
    for i in links:
        a_indices.append(i[0][0])
        b_indices.append(i[0][1])
    a_indices = [[i for i in range(len(links))], a_indices]
    b_indices = [[i for i in range(len(links))], b_indices]
    return a_indices, a_values, b_indices, b_values


def sparse_tensors(i, v, links_len, adj_len):
    i_ = torch.LongTensor(i)
    v_ = torch.FloatTensor(v)
    result = torch.sparse.FloatTensor(i_, v_, torch.Size([links_len, adj_len])).to_dense()
    return result


def get_labels(links):
    labels = []
    for i in links:
        if i[1] == 1:
            labels.append(1)
        else:
            labels.append(0)
    labels = torch.LongTensor(labels)
    labels = torch.reshape(labels, (len(labels), 1))
    return labels
    

def evaluation(output, labels, name):
    preds = torch.round(output)
    y_pred = preds.detach().numpy()
    y_label = labels.detach().numpy()
    acc = accuracy_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred)
    f1 = f1_score(y_label, y_pred)
    auc = roc_auc_score(y_label, y_pred)

    print('{} accuracy:'.format(name), acc,
          '{} f1:'.format(name), f1,
          '{} auc:'.format(name), auc)

    return [acc, recall, precision, f1, auc]


def plotting(x, tr_y, val_y, name_a, name_b, title, x_name, y_name, file_name):
    plt.plot(x, tr_y, 'bo', label=name_a) 
    plt.plot(x, val_y, 'b', label=name_b) 
    plt.title(title) 
    plt.xlabel(x_name) 
    plt.ylabel(y_name) 
    plt.legend()
    plt.savefig('data/output/{}.png'.format(file_name))
    plt.close()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = features.astype(np.float32)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return features


def confusion_m(output, labels, name):
    preds = torch.round(output)
    y_pred = preds.detach().numpy()
    y_label = labels.detach().numpy()
    print('{} confusion matrix'.format(name))
    print(confusion_matrix(y_label, y_pred))
