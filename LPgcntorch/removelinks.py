from scipy.sparse import csr_matrix
import numpy as np
import random
import copy
import pickle
from utils import *
#import time


#generate negative links for training   
def negative_links(adj, pos_num):
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    neg_links = []
    for i in range(len(adj_array)):
        for j in range(i+1):
            if adj_array[i][j] == 0:
                neg_links.append([[i,j], -1])
    random.shuffle(neg_links)
    return neg_links[:pos_num]


#store postive links in a list for training
def positive_links(changed_array):
    changed_array = csr_matrix(changed_array, dtype=np.int8).toarray()
    pos_links = []
    for i in range(len(changed_array)):
        for j in range(i+1):
            if changed_array[i][j] == 1:
                pos_links.append([[i,j],1])
    return pos_links


#generate nagative val samples
def negative_valid(pos_val_num, adj, neg_links):
    neg_val = []
    
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    rows = adj_array.shape[0]
    cols = adj_array.shape[1]
    
    for i in range(pos_val_num):
        while(True):
            row = random.randint(0,rows-1)
            col = random.randint(0,cols-1)
            
            if adj_array[row][col] == 0:
                if [[row,col],-1] not in neg_links and [[col,row],-1] not in neg_links and [row,col] not in neg_val and [col,row] not in neg_val:
                    neg_val.append([row,col])
                    break
                
    neg_val = sorted(neg_val, key=lambda a:a[0])
    print('The number of negative val links:', int(len(neg_val)))
    return neg_val


#randomly remove existed links for LPG
def removed_links(adj, percent):
    total_num = adj.nnz/2
    remove_links_num = int(total_num*percent)
    removes = []
    
    adj_array = csr_matrix(adj, dtype=np.int8).toarray()
    rows = adj_array.shape[0]
    changed_adj = copy.deepcopy(adj_array)
    
    # select the number of removed links
    print('The number of links will be removed is ',remove_links_num)
    for i in range(remove_links_num):
        while(True):
            row = random.randint(0,rows-1) #select a row randomly
            
            ones = []
            for col in range(len(changed_adj[row])):
                if changed_adj[row][col] == 1:
                    ones.append(col)
            
            #make sure there are no isolated nodes in graph
            nums = len(ones)
            if nums >1:
                col = ones[random.randint(0,nums-1)]
                if changed_adj[col].sum() > 1 and row!=col:
                    if changed_adj[row][col] == 0 or changed_adj[col][row] == 0:
                        print(row,'--',col)
                        print('remove error!!!')
                    changed_adj[row][col] = 0
                    changed_adj[col][row] = 0
                    removes.append([row,col])
                    break
    
    changed_adj = csr_matrix(changed_adj)
    removes = sorted(removes, key=lambda a:a[0])
    print('The number of removd links:', int(len(removes)) )
    print('The number links in changed adj', changed_adj.nnz/2)

    tr_pos_links = positive_links(changed_adj)
    tr_neg_links = negative_links(adj, len(tr_pos_links))
    tr_pos_links.extend(tr_neg_links)
    random.shuffle(tr_pos_links)
    val_neg_links = negative_valid(len(removes), adj, tr_neg_links)

    # Return changed_adj, train links(pos + neg), val negative links, val positive links
    return changed_adj, tr_pos_links, val_neg_links, removes


data_loc = '/Users/mingyupark/spyder/GCN_linkprediction/data/'
# Load data
df = pd.read_csv(data_loc + 'patent.csv').iloc[:, 1:]

# Split data into train and test
tr_df, ts_df = split_train_test(df)

# Create symmetric adjacency matrix
tr_adj, tr_cpc_order = create_adj(tr_df)
ts_adj, ts_cpc_order = create_adj(ts_df)

tr_adj.toarray()
tr_adj.shape
#remove setting
remove_percentage = 0.1
#the number of non-zero
print('The number of links in original graph:', tr_adj.nnz/2)
#Remove links for training
changed_adj, tr_links, val_neg_links, val_pos_links = removed_links(tr_adj, remove_percentage)
tr_val_info = [changed_adj, tr_links, val_neg_links, val_pos_links]
orders = [tr_cpc_order, ts_cpc_order]

# Save all links
with open(data_loc + 'tr_val_info.pkl', 'wb') as fw:
    pickle.dump(tr_val_info, fw)
# Save test adj
with open(data_loc + 'ts_adj.pkl', 'wb') as fw:
    pickle.dump(ts_adj, fw)
# Save orders
with open(data_loc + 'orders.pkl', 'wb') as fw:
    pickle.dump(orders, fw)