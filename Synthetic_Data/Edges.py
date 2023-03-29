#erdos-renyi graph generation
import random
from collections import defaultdict
from itertools import product
from typing import Callable, Optional
#import pygsp
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected, get_laplacian
import torch_geometric.transforms as T
import torch_geometric.nn.models.autoencoder as auto
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv, ChebConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
from torch.nn.functional import dropout
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import multivariate_normal

class GraphNetwork(torch.nn.Module):
    def __init__(self, hidden_channels,input_channels,num_classes):
        super().__init__()


        # Initialize EdgeConv layers
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels,hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)


        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels//2)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)

        # self.d1 = dropout(p = 0.2)
        # self.d2 = dropout(p = 0.2)
        # self.d3 = dropout(p = 0.2)


        self.linear = torch.nn.Linear(hidden_channels//2, num_classes)

    def forward(self, data):
        """ Performs a forward pass on our simplified cGCN.

        Parameters:
        data (Data): Graph being passed into network.

        Returns:
        torch.Tensor (N x 2): Probability distribution over class labels.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
      
        x = self.conv1(x, edge_index)
        # x = F.dropout(x,p= 0.1)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        # x = F.dropout(x,p= 0.1)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        #x = F.dropout(x,p= 0.1)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.dropout(x,p= 0.1)
        x = self.linear(x)        
        x = F.softmax(x, dim=1)

        return x

class SyntheticDataset(InMemoryDataset):
    """A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 1,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.kwargs = kwargs

        data_listA = [self.generate_data() for _ in range(max(num_graphs, 1))]
        data_listB = [self.generate_data2() for _ in range(max(num_graphs, 1))]
        data_list = data_listA + data_listB
        #data_list = data_listA
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = self.avg_num_nodes

        data = Data()

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([0])
        fc_edge_index = nx.complete_graph(self.avg_num_nodes)
        p =  0.4
        edges_to_remove = []
        for edge in fc_edge_index.edges:
            if random.uniform(0, 1) < p:
                edges_to_remove.append(edge)
        fc_edge_index.remove_edges_from(edges_to_remove)
        while not nx.is_connected(fc_edge_index):
          for edge in fc_edge_index.edges:
              if random.uniform(0, 1) < p:
                  edges_to_remove.append(edge)
          fc_edge_index.remove_edges_from(edges_to_remove)


        adj_list = nx.to_numpy_array(fc_edge_index).astype(dtype="float32")
        data.edge_index = from_networkx(fc_edge_index).edge_index
        degree_matrix = torch_geometric.utils.degree(data.edge_index[0], num_nodes=self.avg_num_nodes)
        if self.num_channels > 0 and self.task == 'graph':
          #*(data.y+4)
            # data.x = torch.from_numpy(adj_list).to(torch.float32)
            data.x = torch.from_numpy(adj_list).to(torch.float32)*torch.randn(self.avg_num_nodes,self.avg_num_nodes)
            #data.x = torch.cat((data.x, degree_matrix.unsqueeze(-1)), dim=1)

        elif self.num_channels > 0 and self.task == 'node':
            data.x = torch.randn(self.avg_num_nodes,
                                 self.num_channels) + (data.y.unsqueeze(1)//2)+1
        else:
            data.num_nodes = num_nodes
        # if self.edge_dim > 1:
        #     data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        # elif self.edge_dim == 1:
        #     data.edge_weight = torch.rand(data.num_edges)
        #     data.edge_weight = (data.edge_weight - torch.min(data.edge_weight))/(torch.max(data.edge_weight) - torch.min(data.edge_weight))+(data.y+1)
        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))
        return data

    def generate_data2(self) -> Data:
        num_nodes = self.avg_num_nodes

        data = Data()

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, 15))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([1])
        # fc_edge_index = nx.barabasi_albert_graph(self.avg_num_nodes,34)
        # data.edge_index = from_networkx(fc_edge_index).edge_index
        fc_edge_index = nx.complete_graph(self.avg_num_nodes)
        p =  0.6
        edges_to_remove = []
        for edge in fc_edge_index.edges:
            if random.uniform(0, 1) < p:
                edges_to_remove.append(edge)
        fc_edge_index.remove_edges_from(edges_to_remove)
        while not nx.is_connected(fc_edge_index):
          for edge in fc_edge_index.edges:
              if random.uniform(0, 1) < p:
                  edges_to_remove.append(edge)
          fc_edge_index.remove_edges_from(edges_to_remove)

        adj_list = nx.to_numpy_array(fc_edge_index).astype(dtype="float32")
        data.edge_index = from_networkx(fc_edge_index).edge_index
        degree_matrix = torch_geometric.utils.degree(data.edge_index[0], num_nodes=self.avg_num_nodes)
        if self.num_channels > 0 and self.task == 'graph':
          #*((data.y)//2) 
            data.x = torch.from_numpy(adj_list).to(torch.float32)*torch.randn(self.avg_num_nodes,self.avg_num_nodes)
            #data.x = torch.cat((data.x, degree_matrix.unsqueeze(-1)), dim=1)
        elif self.num_channels > 0 and self.task == 'node':
            data.x = torch.randn(self.avg_num_nodes,
                                 self.num_channels) + data.y.unsqueeze(1)//2
        else:
            data.num_nodes = num_nodes
        # if self.edge_dim > 1:
        #     data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        # elif self.edge_dim == 1:
        #     data.edge_weight = torch.rand(data.num_edges) + (data.y)//2
        #     data.edge_weight = (data.edge_weight - torch.min(data.edge_weight))/(torch.max(data.edge_weight) - torch.min(data.edge_weight))
        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))
        return data

class GraphNetwork2(torch.nn.Module):
    def __init__(self, hidden_channels,input_channels,num_classes):
        super().__init__()

        # Initialize MLPs used by EdgeConv layers
        self.mlp1 = Sequential(Linear(2 * input_channels, hidden_channels), ReLU())
        self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())

        # Initialize EdgeConv layers
        self.conv1 = EdgeConv(self.mlp1, aggr='max')
        self.conv2 = EdgeConv(self.mlp2, aggr='max')
        self.conv3 = EdgeConv(self.mlp3, aggr='max')

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)



        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        """ Performs a forward pass on our simplified cGCN.

        Parameters:
        data (Data): Graph being passed into network.

        Returns:
        torch.Tensor (N x 2): Probability distribution over class labels.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)        
        x = F.softmax(x, dim=1)

        return x

def train(model, loss_fn, device, data_loader, optimizer):
    """ Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)

        loss = loss_fn(out, batch.y)

        loss.backward()
        optimizer.step()

    return loss.item()

def eval(model, device, loader):
    """ Calculate accuracy for all examples in a DataLoader.

    Parameters:
    model (nn.Module): Model to be evaluated.
    device (torch.Device): Device used for training.
    loader (torch.utils.data.DataLoader): DataLoader containing examples to test.
    """
    model.eval()
    cor = 0
    tot = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = torch.argmax(model(batch), 1)

        y = batch.y
        cor += (pred == y).sum()
        tot += pred.shape[0]

    return cor / tot


def main():
    avg_num_nodes = 35
    num_channels = 25
    dataset = SyntheticDataset(num_graphs = 5000,avg_num_nodes = avg_num_nodes,avg_degreee=2, num_channels = num_channels, num_classes = 2,edge_dim = 1,  is_undirected=True)
    dataset = dataset.shuffle()

    # Train/test split (80-20)
    train_share = int(len(dataset) * 0.8)

    train_dataset = dataset[:train_share]
    test_dataset = dataset[train_share:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphNetwork(32,35,2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(0, 50):
        loss = train(model, loss_fn, device, train_loader, optimizer)
        train_result = eval(model, device, train_loader)
        test_result = eval(model, device, test_loader)
        # for param in model.parameters():
        #     loss += 0.5 * param.norm(2) ** 2
        losses.append(loss)
        scheduler.step()
        
        print(f'Epoch: {epoch + 1:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_result:.2f}%, '
            f'Test: {100 * test_result:.2f}%')

    plt.plot(losses)
    plt.show()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("/home/ac1906/Research_2022-2023/Synthetic_Data/figures/Edge_Loss.png")


    print("========= Non Graph Based ==========")
    #Run Non-Deep Learning Methods
    #Using Gaussian Naive Bayes
    print("========== Non-Graph Based Methods ==========")
    train_list = []
    train_labels = []
    for i in train_dataset:
        train_list.append(i.x.flatten())
        train_labels.append(i.y)

    test_list = []
    test_labels = []

    for i in test_dataset:
        test_list.append(i.x.flatten())
        test_labels.append(i.y)

    x_train = torch.stack(train_list).numpy()
    y_train = torch.stack(train_labels).numpy().flatten()

    x_test = torch.stack(test_list).numpy()
    y_test = torch.stack(test_labels).numpy().flatten()


    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for QDA is {}".format(correct/total))

    clf2 = GaussianNB()
    clf2.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for GNB is {}".format(correct/total))
    


if __name__ == "__main__":
    main()


