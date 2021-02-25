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

## Usage
1) ```python crawling.py```
2) ```python removelinks.py```
3) ```python features.py```
4) ```python train.py```
* Best epochs was 44~46

## Reference
- kenyonke/LinkPredictionGCN
- tkipf/pygcn

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
