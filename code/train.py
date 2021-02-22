import pandas as pd
import pickle as pkl
from utils import *
from sklearn.preprocessing import StandardScaler

data_loc = '/Users/mingyupark/spyder/GCN_linkprediction/data/'

# Load adj, train/valid network information
with open(data_loc + 'tr_val_info.pkl', 'rb') as fr:
    tr_val_info = pkl.load(fr)

changed_adj = tr_val_info[0]
tr_links = tr_val_info[1]
val_neg_links = tr_val_info[2]
val_pos_links = tr_val_info[3]

# Make final A hat adj for GCN layer
adj = normalize_adj(changed_adj.toarray())

# Get feature matrix: X for GCN layer
# 1. growing rate
# 2. Frequency in the network
# 3. Frequency in the whole patent
# 4. Relation with the Domain
# 5. CPC'name vector representation
with open(data_loc + 'features.pkl', 'rb') as fr:
    feature = pkl.load(fr)
feature = StandardScaler().fit_transform(feature)

