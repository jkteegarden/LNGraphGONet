import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from math import *
import networkx as nx
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import gc


#Clear GPU memory
def torch_clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()

#DatasetLoader based on the tutorial "Creating Your Own Datasets" - pytorch-geometric 
class GeneExpressionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name,mask_features=None,transform=None,n_classes=1,class_weights=False,n_samples=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        loader =  np.load(file_name)
        self.X = loader['x'].astype('float32')
        if n_classes <2 :
            self.y = loader['y'].astype('float32')
        else:
            self.y = loader['y']
        self.transform = transform
        
        if self.y.dtype =='<U4':
            le = preprocessing.LabelEncoder()
            le.fit(self.y)
            self.y = le.transform(self.y)
            self.target_names = list(le.classes_)

        if mask_features is not None:
            self.X = self.X[:,mask_features]
        if n_samples:
            sss = StratifiedShuffleSplit(n_splits=1,train_size=n_samples,test_size=self.X.shape[0]-n_samples,random_state=42) #keeping the proportion of the original classes
            for train_index, test_index in sss.split(self.X , self.y):
                self.X, self.y = self.X[train_index,:], self.y[train_index]
        
        if class_weights:
            self.class_weight = torch.tensor(class_weight.compute_class_weight('balanced',
                                                 np.unique(self.y),
                                                 self.y).astype('float32'))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': self.X[idx,], 'labels': self.y[idx]}
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
            sample['labels'] = torch.tensor(sample['labels'])

        return sample["data"],sample["labels"]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data)
    
#inpired from the function with the same name from pytorch-geometric
def from_networkx(G,label,dim_inital_node_embedding=1,random=False):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        label (list): Label of patient's outcome. (binary(scalar): 0/1)
        dim_inital_node_embedding (int): Here 1, initialized to 0 and will be updated latter after applying the first module.
    """

    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges(data=False))).t().contiguous()

    data = {}
    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    data.x = torch.zeros(data.num_nodes, dim_inital_node_embedding)

    return data

#Functions for the training
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_accuracy(y_true, y_pred,n_classes):
    if n_classes<2:
        y_pred = y_pred > 0.5
    return int(np.sum(np.equal(y_true,y_pred))) / y_true.shape[0]

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def model_forward(batchGE, batchGraph, model, device=torch.device("cuda")):
    batchGraph = batchGraph.to(device)
    out = model(transcriptomic_data=batchGE, graph_data=batchGraph).to(device)
    return out
