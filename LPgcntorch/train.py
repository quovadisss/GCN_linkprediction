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
parser.add_argument('--edge_operator', type=str, default='cosine',
                    help='Edge operator which change node pairs to edge features')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--new_epochs', action='store_true', default=False,
                    help='Save final results.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=48,
                    help='Number of hidden units.')
parser.add_argument('--out_dim', type=int, default=6,
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
with open('data/tr_val_info_all.pkl', 'rb') as fr:
    tr_val_info = pickle.load(fr)

adj = tr_val_info[0]
tr_links = tr_val_info[1]
val_links = tr_val_info[2]

# Get feature matrix: X for GCN layer
with open('data/features.pkl', 'rb') as fr:
    features = pickle.load(fr)
features = features.T[4:].T


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


def node2edge(node_a, node_b, output, length, dataset):
    def save_operator(array):
        with open('data/output_{0}_{1}_all.pkl'.format(args.edge_operator,
                                                   dataset), 'wb') as fw:
            pickle.dump(array, fw)

    mul_a = torch.matmul(node_a, output)
    mul_b = torch.matmul(node_b, output)

    if args.edge_operator == 'cosine':
        sig = m(cosine(mul_a, mul_b).reshape(length, 1))
        save_operator(sig.detach().numpy())
    elif args.edge_operator == 'hadamard':
        save_operator((mul_a * mul_b).detach().numpy())
    elif args.edge_operator == 'average':
        save_operator(((mul_a + mul_b) / 2).detach().numpy())
    elif args.edge_operator == 'weighted-l1':
        save_operator((torch.abs(mul_a - mul_b)).detach().numpy())
    elif args.edge_operator == 'weighted-l2':
        save_operator((torch.square(mul_a - mul_b)).detach().numpy())


def train(epoch):
    t = time.time()
    print('Epoch: {:04d}'.format(epoch+1))
    model.train()
    optimizer.zero_grad()
    pre_output = model(features, adj)
    mul_a = torch.matmul(tra, pre_output)
    mul_b = torch.matmul(trb, pre_output)
    output = cosine(mul_a, mul_b).reshape(len(tr_links), 1)
    tr_output = m(output)
    loss_train = criterion(tr_output, tr_labels.float())
    eval_tr = evaluation(tr_output, tr_labels.float(), 'train')
    loss_train.backward()
    optimizer.step()

    # valid
    mul_a = torch.matmul(vala, pre_output)
    mul_b = torch.matmul(valb, pre_output)
    output = cosine(mul_a, mul_b).reshape(len(val_links), 1)
    val_output = m(output)
    loss_val = criterion(val_output, val_labels.float())
    eval_val = evaluation(val_output, val_labels.float(), 'valid')
    print('loss_train: {:.4f}'.format(loss_train.item()),
    'loss_val: {:.4f}'.format(loss_val.item()))

    if epoch == args.epochs - 1:
        confusion_m(tr_output, tr_labels.float(), 'train')
        confusion_m(val_output, val_labels.float(), 'valid')

    print('time: {:.4f}s'.format(time.time() - t))
    
    return loss_train, loss_val, eval_tr, eval_val


tr_loss = []
val_loss = []
tr_eval = []
val_eval = []
for e in range(args.epochs):
    loss_train, loss_val, eval_tr, eval_val = train(e)
    tr_loss.append(loss_train)
    val_loss.append(loss_val)
    tr_eval.append(eval_tr)
    val_eval.append(eval_val)
print('Training Done!')

# Extract evaluation values
tr_acc = [i[0] for i in tr_eval]
val_acc = [i[0] for i in val_eval]
tr_recall = [i[1] for i in tr_eval]
val_recall = [i[1] for i in val_eval]
tr_preci = [i[2] for i in tr_eval]
val_preci = [i[2] for i in val_eval]
tr_f1 = [i[3] for i in tr_eval]
val_f1 = [i[3] for i in val_eval]
tr_auc = [i[4] for i in tr_eval]
val_auc = [i[4] for i in val_eval]
print(tr_recall[0], tr_preci[0])

# Save loss figure
epochs = range(1, args.epochs + 1)

if args.new_epochs == False:
    plotting(epochs, tr_loss, val_loss,
            'Training loss','Validation loss',
            'Training and validation loss',
            'Epochs',
            'Loss',
            'train_val_loss_all')
    plotting(epochs, tr_acc, val_acc,
            'Training accuracy','Validation accuracy',
            'Training and validation accuracy',
            'Epochs',
            'Accuracy',
            'train_val_acc_all')
    plotting(epochs, tr_auc, val_auc,
            'Training roc_auc','Validation roc_auc',
            'Training and validation roc_auc',
            'Epochs',
            'ROC AUC',
            'train_val_auc_all')
else:
    # Save final model
    with open('data/gcn_model.pkl', 'wb') as fw:
        pickle.dump(model, fw)

output = model(features, adj)
# If edge operator is not cosine, just save 
node2edge(tra, trb, output, len(tr_links), 'tr')
node2edge(vala, valb, output, len(val_links), 'val')

