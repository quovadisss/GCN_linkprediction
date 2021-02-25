# Link Prediction using GCN on pytorch

## Project explanation
This project is to predict whether patent's cpc nodes are linked or not. To accomplish this project, general GCN model from Kipf are used on pytorch. The patents are crawled in the Mobile Payment Industry.

## Framework
1) Search 'Mobile Payment' in google patent advanced search and get patent numbers.
2) Crawl all patents by using the patent numbers from 1).
3) Create adjacency matrix and feature matrix
4) Remove links and split data into train set and validation set.
5) Get new node features from GCN layers.
6) Calculate similarities of node pairs.
7) Minimize loss with the labels and update weights.

## Code execution procedures
1) crawling.py
2) removelinks.py
3) features.py
4) train.py

## Reference
- kenyonke/LinkPredictionGCN
- tkipf/pygcn


