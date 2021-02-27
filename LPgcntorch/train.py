from __future__ import division
from __future__ import print_function

import time
import pickle
import pandas as pd
import numpy as np
import argparse

import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--new_epochs', type=int, default=False,
                    help='Save final results.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--out_dim', type=int, default=12,
                    help='Number of output dimensions.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load adj, train/valid network information
with open('data/tr_val_info.pkl', 'rb') as fr:
    tr_val_info = pickle.load(fr)

adj = tr_val_info[0]
tr_links = tr_val_info[1]
val_links = tr_val_info[2]

# Get feature matrix: X for GCN layer
with open('data/features.pkl', 'rb') as fr:
    features = pickle.load(fr)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            out_dim=args.out_dim,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# To Tensor for the inputs
features = torch.FloatTensor(features)
adj = sparse_mx_to_torch_sparse_tensor(adj)

# Create sparse tensors for link features
traina_indices, traina_values, trainb_indices, trainb_values = make_ind_val(tr_links)
vala_indices, vala_values, valb_indices, valb_values = make_ind_val(val_links)

tra = sparse_tensors(traina_indices, traina_values, len(tr_links), len(adj))
trb = sparse_tensors(trainb_indices, trainb_values, len(tr_links), len(adj))
vala = sparse_tensors(vala_indices, vala_values, len(val_links), len(adj))
valb = sparse_tensors(valb_indices, valb_values, len(val_links), len(adj))

tr_labels = get_labels(tr_links)
val_labels = get_labels(val_links)
cosine = torch.nn.CosineSimilarity()
m = torch.nn.Sigmoid()
criterion = torch.nn.BCELoss()

# Cuda
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    tra = tra.cuda()
    trb = trb.cuda()
    vala = vala.cuda()
    valb = valb.cuda()
    tr_labels = tr_labels.cuda()
    val_labels = val_labels.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    pre_output = model(features, adj)
    output = cosine(torch.matmul(tra, pre_output), torch.matmul(trb, pre_output)).reshape(len(tr_links), 1)
    # This saving output prodedure is for another analysis
    if epoch == args.epochs:
        with open('data/last_tr_output.pkl', 'wb') as fw:
            pickle.dump(output, fw)
    tr_output = m(output)
    
    loss_train = criterion(tr_output, tr_labels.float())
    acc_train = accuracy(tr_output, tr_labels.float())
    auc_train = auc_score(tr_output, tr_labels.float())
    loss_train.backward()
    optimizer.step()

    # valid
    output = cosine(torch.matmul(vala, pre_output), torch.matmul(valb, pre_output)).reshape(len(val_links), 1)
    # This saving output prodedure is for another analysis
    if epoch == args.epochs:
        with open('data/last_val_output.pkl', 'wb') as fw:
            pickle.dump(output, fw)
    val_output = m(output)
    loss_val = criterion(val_output, val_labels.float())
    acc_val = accuracy(val_output, val_labels.float())
    auc_val = auc_score(val_output, val_labels.float())

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'roc_auc_train: {:.4f}'.format(auc_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'roc_auc_val: {:.4f}'.format(auc_val),
          'time: {:.4f}s'.format(time.time() - t))
    
    return loss_train, loss_val, acc_train, acc_val, auc_train, auc_val


tr_loss = []
val_loss = []
tr_acc = []
val_acc = []
tr_auc = []
val_auc = []
for e in range(args.epochs):
    loss_train, loss_val, acc_train, acc_val, auc_train, auc_val = train(e)
    tr_loss.append(loss_train)
    val_loss.append(loss_val)
    tr_acc.append(acc_train)
    val_acc.append(acc_val)
    tr_auc.append(auc_train)
    val_auc.append(auc_val)
print('Done!')


# Save loss figure
epochs = range(1, args.epochs + 1)

if args.new_epochs == False:
    plotting(epochs, tr_loss, val_loss,
            'Training loss','Validation loss',
            'Training and validation loss',
            'Epochs',
            'Loss',
            'train_val_loss')
    plotting(epochs, tr_acc, val_acc,
            'Training accuracy','Validation accuracy',
            'Training and validation accuracy',
            'Epochs',
            'Accuracy',
            'train_val_acc')
    plotting(epochs, tr_auc, val_auc,
            'Training roc_auc','Validation roc_auc',
            'Training and validation roc_auc',
            'Epochs',
            'ROC AUC',
            'train_val_auc')
else:
    # Save final model
    with open('data/gcn_model.pkl', 'wb') as fw:
        pickle.dump(model, fw)
