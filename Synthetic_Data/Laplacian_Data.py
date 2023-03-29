#erdos-renyi graph generation
import random
from collections import defaultdict
from itertools import product
from typing import Callable, Optional
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected, get_laplacian
import torch_geometric.transforms as T
import torch_geometric.nn.models.autoencoder as auto
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import networkx as nx    
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

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
        p = 0.5
        G = nx.erdos_renyi_graph(self.avg_num_nodes, p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(self.avg_num_nodes, p)
        self.G = G

        data_listA = [self.generate_data() for _ in range(max(num_graphs, 1))]
        data_list = data_listA
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = self.avg_num_nodes

        data = Data()

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

     
        data.edge_index = from_networkx(self.G).edge_index
        L = nx.laplacian_matrix(self.G).toarray()
        if self.num_channels > 0 and self.task == 'graph':
            L_inv = np.linalg.pinv(L)
            mean = np.ones((self.avg_num_nodes))*data.y.item()/1000;
            data.x = torch.tensor(multivariate_normal.rvs(mean, L_inv, size = self.num_channels).astype(np.float32).T)

        elif self.num_channels > 0 and self.task == 'node':
            data.x = torch.randn(self.avg_num_nodes,
                                 self.num_channels) + (data.y.unsqueeze(1)//2)+1
        else:
            data.num_nodes = num_nodes
        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.ones(data.num_edges)
        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))
        return data

class GraphNetwork(torch.nn.Module):
    def __init__(self, hidden_channels,input_channels,num_classes):
        super().__init__()

        # Initialize MLPs used by EdgeConv layers
        # self.mlp1 = Sequential(Linear(2 * dataset.num_node_features, hidden_channels), ReLU())
        # self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        # self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())

        # Initialize EdgeConv layers
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels,hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, hidden_channels//4)
        self.conv4 = GCNConv(hidden_channels//4, hidden_channels//8)


        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels//2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels//4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels//8)


        self.linear = torch.nn.Linear(hidden_channels//8, num_classes)

    def forward(self, data):
        """ Performs a forward pass

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
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)

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
    #make dataset
    avg_num_nodes = 35
    num_channels = 100
    dataset = SyntheticDataset(num_graphs = 1000 ,avg_num_nodes = avg_num_nodes ,avg_degreee=2, num_channels= num_channels, num_classes = 2,edge_dim = 1,  is_undirected=True)
    dataset = dataset.shuffle()

    # Train/test split (80-20)
    train_share = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_share]
    test_dataset = dataset[train_share:]
    
    #device agnostic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initalize model
    model = GraphNetwork(1024,num_channels,2).to(device)
    #model = GCNClassification(32,35,2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=1e-6)

    #make train and test data
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(0, 30):
        loss = train(model, loss_fn, device, train_loader, optimizer)
        train_result = eval(model, device, train_loader)
        test_result = eval(model, device, test_loader)
        
        losses.append(loss)
        
        print(f'Epoch: {epoch + 1:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_result:.2f}%, '
            f'Test: {100 * test_result:.2f}%')

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("/home/ac1906/Research_2022-2023/Synthetic_Data/figures/Lap_Loss_2.png")
    
    
    
    #Run Non-Deep Learning Methods
    print("\n========== Non-Graph Based Methods with PCA ==========")
    train_list = []
    train_labels = []
    pca = PCA(n_components=0.95)
    pca_2 = PCA(n_components=0.95)

    for i in train_dataset:
        data = i.x.numpy()
        train_list.append(data.flatten())
        train_labels.append(i.y.item())

    test_list = []
    test_labels = []

    for i in test_dataset:
        data = i.x.numpy()
        test_list.append(data.flatten())
        test_labels.append(i.y.item())

    x_train = np.array(train_list)
    y_train = np.array(train_labels).flatten()
    
    x_test = np.array(test_list)
    y_test = np.array(test_labels).flatten()

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for LDA is {}".format(correct/total))

    clf2 = GaussianNB()
    clf2.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for GNB is {} \n".format(correct/total))



    print("========== Non-Graph Based Methods without PCA ==========")
    train_list = []
    train_labels = []
    for i in train_dataset:
        train_list.append(i.x.flatten())
        train_labels.append(i.y.item())

    test_list = []
    test_labels = []

    for i in test_dataset:

        test_list.append(i.x.flatten())
        test_labels.append(i.y.item())

    x_train = torch.stack(train_list).numpy()
    y_train = np.array(y_train)

    x_test = torch.stack(test_list).numpy()
    y_test = np.array(y_test)
    
    print(y_train.shape)
    print(y_test.shape)

    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for LDA is {}".format(correct/total))

    clf2 = GaussianNB()
    clf2.fit(x_train, y_train)
    preds = clf.predict(x_test)
    correct =  (preds == y_test).sum()
    total = preds.shape[0]
    print("Accuracy for GNB is {} \n".format(correct/total))


if __name__ == "__main__":
    main()