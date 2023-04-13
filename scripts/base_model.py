import numpy as np
from math import *
import torch
from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter,scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import pdb

#inpired from the modules Graph Convolutional Layers from pytorch-geometric
class DAGProp(torch.nn.Module):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, root_weight: bool = True,
                 bias: bool = True, nonlinearity: Callable = torch.tanh, aggr: str = "mean",**kwargs):
        super(DAGProp, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.nonlinearity = nonlinearity
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, batch: OptTensor = None , 
                size: Size = None) -> Tensor:
        num_nodes = scatter_add(batch.new_ones(x.size(0),dtype=torch.int16), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        num_edges_ref = edge_index.shape[1] // batch_size #in our case, all the sample graphs share the same DAG structure

        out = x.new_zeros(x.size()) #new embedding: h_v^{(1)}
        visited = x.new_zeros(size=(x.size(0),),dtype=torch.int16) #to propagate sequentially: trace the visits

        #1. Start propagation from the leaves
        #Propagation rule: h_v=\sigma(h_{G(v)}) | |\mathcal{N}(v)| =0  
        leaves = torch.where(scatter(src=edge_index.new_ones(edge_index[1,:].size(),dtype=torch.int8),
                            index=edge_index[1,:], reduce="sum")==0)[0] #determine the leaf nodes
        #if you don't know the leaves otherwise take the object leaves computed beforehand in the analysis part
        out[leaves] = self.nonlinearity(x[leaves])
        visited[leaves]=1 #update the trace of the visits
        
        #2. Compute self-nodes processing (case: |\mathcal{N}(v)|>0)
        #Propagation rule: h_v = w_G h_{G(v)}
        if self.root_weight:
            temp = x[leaves] # remove leaves first [op isin() doesn't exist]
            x[leaves] = 0
            mask = x.nonzero(as_tuple=True)[0]
            x[leaves] = temp
            out[mask] = self.lin_r(x[mask]) 
        
        #3. Update the embedding of the other nodes 
        #Propagation rule: $h_v = \sigma(w_{\mathcal{N}} h_{\mathcal{N}(v)})$ with $h_v^{(1) + h_v}$ computed from the previous step

        #First trial: we use a reference (a given graph) as the graph will be the same whatever the sample is 
        #Uncomment the following line if the graph is different for each sample
        previous_visits = leaves[leaves < max_num_nodes]
        adj_mat_cropped = edge_index[:,:num_edges_ref]
        while torch.sum(visited)!=visited.shape[0]: 
            #determine what are the parents of the GO terms previously visited (denoted as previous_visits) according to the relation $u_{\mathit{visited}}->v$
            mask=(adj_mat_cropped[0,:][..., None] == previous_visits).any(-1)
            fathers = torch.unique(adj_mat_cropped[1,mask])
            #determine the entire neighboring (child GO terms) of the parent GO termss under consideration (including previous_visits) 
            del mask
            mask=(adj_mat_cropped[1,:][..., None]== fathers).any(-1)
            #look if the entire neighboring has been visited already
            mask1=scatter(src=visited[adj_mat_cropped[0,mask]], index=adj_mat_cropped[1,mask], reduce="sum")
            mask1=mask1[fathers]
            mask2=scatter(src=visited.new_ones(size=visited[adj_mat_cropped[0,mask]].size(),dtype=torch.int16), index=adj_mat_cropped[1,mask], reduce="sum")
            mask2=mask2[fathers]
            ref_next_visits = fathers[mask1==mask2]
            #extract the adjacency matrix restricted to the parents selected and their neighbors
            del mask
            mask = (adj_mat_cropped[1,:][..., None] == ref_next_visits).any(-1) 
            adj_mat = [adj_mat_cropped[:,mask] + i*max_num_nodes for i in range(batch_size)]
            adj_mat = torch.cat(adj_mat, dim=1)
            #propagate the index to the all batch
            next_visits = [ref_next_visits + i*max_num_nodes for i in range(batch_size)]
            next_visits = torch.cat(next_visits, dim=0)
            #prepare the inputs for aggregation: extract the neighbood embedding of fathers 
            mask = adj_mat[0,:]
            #take the most updated embedding of the neighboring
            children = out[mask].view(-1)
            out[next_visits] += self.lin_l(scatter(src=children,index=adj_mat[1,:],reduce=self.aggr)[next_visits][:,None])
            out[next_visits] = self.nonlinearity(out[next_visits])
            previous_visits = ref_next_visits
            visited[next_visits]=1

        return out

    def __repr__(self):
        return '{}({}, {}, aggr={}, nonlinearity={})'.format(self.__class__.__name__, self.in_channels,self.out_channels,self.aggr,self.nonlinearity.__name__)
    
#inpired from the modules Pooling Layers from pytorch-geometric
class TopSelection(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5, **kwargs):
        super(TopSelection, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        perm = topk(torch.abs(x).view(-1), self.ratio, batch)
        num_nodes = x.size(0)
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=num_nodes)
        return x, edge_index, edge_attr, batch, perm, x

    def __repr__(self):
        return '{}({}, ratio={})'.format(
            self.__class__.__name__, 
            self.in_channels,
            self.ratio)
    
class RandomSelection(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5, **kwargs):
        super(RandomSelection, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        max_num_nodes = scatter_add(batch.new_ones(x.size(0),dtype=torch.int16), batch, dim=0).max().item()  
        perm = [torch.sort(torch.randperm(max_num_nodes,dtype=torch.long, device=x.device)[:ceil(max_num_nodes*self.ratio)]+k*max_num_nodes,descending=False)[0] for k in torch.unique(batch)]
        perm = torch.cat(perm, dim=0)
        num_nodes = x.size(0)
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes = num_nodes)

        return x, edge_index, edge_attr, batch, perm, x

    def __repr__(self):
        return '{}({}, ratio={})'.format(
            self.__class__.__name__, 
            self.in_channels,
            self.ratio)

def concatenate(x: Tensor, batch: Tensor, size: Optional[int] = None) -> Tensor:
    batch_size = int(batch.max().item() + 1) if size is None else size
    return x.view(batch_size,-1)

def concatenate_and_mask(x: Tensor, batch: Tensor, idx_nodes_kept : Tensor, num_nodes : Tensor) -> Tensor:
    num_nodes_kept = scatter_add(batch.new_ones(x.size(0),dtype=torch.int16), batch, dim=0)
    #it is the same number of nodes across the graph samples
    batch_size, max_num_nodes = num_nodes_kept.size(0), num_nodes.max().item()
    output = x.new_zeros((batch_size,max_num_nodes)) #padding if not the same number of nodes
    for i in torch.arange(batch_size):
        #get by batch the original indices of the nodes kept
        mask = idx_nodes_kept[i*num_nodes_kept[i]:(i+1)*num_nodes_kept[i]] - i*num_nodes[i]   #substract to get the number in the correct range
        output[i,mask]=x[i*num_nodes_kept[i]:(i+1)*num_nodes_kept[i]].view(-1) #shape : (num_nodes_by_graph,1) -> (_,max_num_nodes)
    return output

class Mask(torch.nn.Module):
    def __init__(self, in_channels, method, n_nodes, **kwargs):
        super(Mask, self).__init__()
        self.method = method
        self.in_channels = in_channels
        if self.method.__name__ == "global_mean_pool": 
            self.out_neurons=1
        else:
            self.out_neurons=n_nodes

    def forward(self, *args):
        return self.method(*args)
        
    def __repr__(self):
        return '{}({}, {}, method={})'.format(
            self.__class__.__name__,
            self.out_neurons,
            self.in_channels,
            self.method.__name__)

class Net(torch.nn.Module):
    def __init__(self,n_genes,n_nodes,n_nodes_annot,n_nodes_emb,n_prop1,n_classes,adj_mat_fc1,
                 propagation="DAGProp",selection=None,ratio=1.0,
                 mask="concatenate_and_mask"):
        super(Net, self).__init__()
        self.n_genes = n_genes
        self.n_nodes = n_nodes
        self.n_nodes_annot = n_nodes_annot
        self.n_nodes_emb = n_nodes_emb
        self.n_prop1 = n_prop1
        self.n_classes = n_classes
        adj_mat_fc1 = torch.tensor(adj_mat_fc1, dtype=torch.float).t()
        self.adj_mat_fc1 = Parameter(adj_mat_fc1, requires_grad=False)
        self.fc1 = Linear(in_features=n_genes,out_features=n_nodes_annot)
        with torch.no_grad():
            self.fc1.weight.mul_(self.adj_mat_fc1) #mask all the connections btw genes and neurons that do not represent GO annotations
        self.propagation = eval(propagation)(in_channels=n_nodes_emb, out_channels=n_prop1,aggr = "mean") # expected dim: [nSamples, nNodes, nChannels]
        if selection:
            self.ratio = ratio
            if selection=="random":
                self.selection = RandomSelection(in_channels=n_prop1,ratio=ratio)
            elif selection=="top":
                self.selection = TopSelection(in_channels=n_prop1,ratio=ratio)
        else:
            mask="concatenate"
        self.mask = Mask(method=globals()[mask],in_channels=n_prop1,n_nodes=n_nodes) #option no selection => concatenate  
        self.fc2 = Linear(in_features=n_nodes,out_features=n_classes) 

    def forward(self,transcriptomic_data,graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        initial_embedding = self.fc1(transcriptomic_data)
        for k in np.arange(graph_data.num_graphs): 
            x[self.n_nodes*k:self.n_nodes*k+self.n_nodes_annot]=initial_embedding[k].unsqueeze_(1) #initialize the signal coming from the genes based on GO annotations
        
        x = self.propagation(x, edge_index,batch)
        
        num_nodes = scatter_add(batch.new_ones(x.size(0),dtype=torch.int16), batch, dim=0)
        
        if self.selection: 
            x, edge_index, _, batch,idx_nodes_kept,_ = self.selection(x, edge_index, None, batch)
        if self.mask.method.__name__ == "concatenate_and_mask":
            x = self.mask(x,batch,idx_nodes_kept,num_nodes)
        else:
            x = self.mask(x,batch)
            
        x = self.fc2(x)
        
        if self.n_classes>=2:
            return x
        else:
            return x.view(-1)