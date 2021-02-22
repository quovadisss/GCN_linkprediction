import pandas as pd
import pickle as pkl
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from LPgcntorch.utils import *
from LPgcntorch.models import GCN
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
with open(data_loc + 'features.pkl', 'rb') as fr:
    features = pkl.load(fr)
features = StandardScaler().fit_transform(features)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

hidden = 16
lr = 0.01
weight_decay = 5e-4
dropout = 0.5
epochs = 200

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            out_dim=25,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

model.cuda()
features = features.cuda()
adj = changed_adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))