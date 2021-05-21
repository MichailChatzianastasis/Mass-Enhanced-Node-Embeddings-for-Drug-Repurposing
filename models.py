import networkx as nx
import os
import pickle
import os.path as osp
import numpy as np
import sys
from msi.msi import MSI
import pickle
from tests.msi import test_msi
from operator import itemgetter
import pickle
import networkx as nx
import torch
import torch.nn as nn

#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec

class ModelNode2Vec():

    def __init__(self):
        super(ModelNode2Vec, self).__init__()
        self.edges = []
        self.construct_msi_dataset_to_pg()
        self.train_model()

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()
        
        path_to_graph = "./results/graph.pkl"
        path_node2idx = "./results/node2idx.pkl"

        with open(path_to_graph, 'rb') as handle:
            graph = pickle.load(handle)

        with open(path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
    
    
        start_node_list = []
        end_node_list = []
        for edge in graph.edges:
                start_node_list.append(node2idx[edge[0]])
                end_node_list.append(node2idx[edge[1]])
        self.edges = torch.tensor([start_node_list,end_node_list])        
    
    def train_model(self):
        EPOCHS = 2
        EMB_DIM = 256 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Node2Vec(self.edges, embedding_dim=EMB_DIM, walk_length=20,
                        context_size=10, walks_per_node=10,
                        num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    
        loader = model.loader(batch_size=128, shuffle=True, num_workers=12)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        model.train()
        for epoch in range(1, EPOCHS):
            
            total_loss = 0

            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            loss = total_loss / len(loader)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            #acc = test()
            #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        
        node_embeddings = np.array(model.forward().data.cpu())
        with open(f'/data/multiscale-interactome/data/node2vec_embeddings_{EMB_DIM}_{EPOCHS}.pickle', 'wb') as handle:
            pickle.dump(node_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
class MLP(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLP, self).__init__()      
        self.fc1 = nn.Linear(2*EMB_DIM, EMB_DIM)
        self.fc2=  nn.Linear(EMB_DIM, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x

def construct_dataset_from_node_embeddings(node_embeddings_path="./data/node2vec_embeddings_256_2.pickle"):
    '''
    Construct a dataset X = (emb_drug,emb_disease) , y=0 or 1 (if drug treat the disease). 
    It takes as argument the path of the node embeddings
    '''
    with open(node_embeddings_path, 'rb') as handle:
        node_embeddings = pickle.load(handle)
    print(node_embeddings.shape)



def train_MLP():
    epochs = 50
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #forward loop
    losses = []
    accur = []
    for i in range(epochs):
        for j,(x_train,y_train) in enumerate(trainloader):
        
            #calculate output
            output = model(x_train)
        
            #calculate loss
            loss = loss_fn(output,y_train.reshape(-1,1))
        
            #accuracy
            predicted = model(torch.tensor(x,dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i%10 == 0:
            losses.append(loss)
            accur.append(acc)
            print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

class ModelMetaPath2Vec():
    "NEED FIX"
    def __init__(self):
        super(ModelMetaPath2Vec, self).__init__()
        construct_msi_dataset_to_pg()

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()
        
        
        path_to_graph = "./results/graph.pkl"
        path_node2idx = "./results/node2idx.pkl"

        with open(path_to_graph, 'rb') as handle:
            graph = pickle.load(handle)

        with open(path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
        
        
        metapath = [
            ('drug', 'to', 'protein'),
            ('protein', 'to', 'protein'),
            ('protein', 'from', 'indication'),
            ('indication', 'to', 'protein'), 
            ('protein', 'from', 'protein'),
            ('protein', 'to', 'biological_function'),
            ('biological_function', 'to', 'biological_function'),
            ('biological_function', 'from', 'biological_function'),
            ('biological_function', 'from', 'protein'),
            ('protein','from','drug')
        ]

        edge_index_dict = {}
        #construction of edge_index_dict 
        def construct_two_edge_list(edge_list,reversed=False):
            start_node_list = []
            end_node_list = []
            for edge in edge_list:
                if(reversed==True):
                    start_node_list.append(node2idx[edge[1]])
                    end_node_list.append(node2idx[edge[0]])
                else:
                    start_node_list.append(node2idx[edge[0]])
                    end_node_list.append(node2idx[edge[1]])
            return torch.tensor([start_node_list,end_node_list])


        edge_index_dict[('drug','to','protein')] = construct_two_edge_list(list(msi.components["drug_to_protein"].edge_list))
        edge_index_dict[('indication','to','protein')] = construct_two_edge_list(list(msi.components["indication_to_protein"].edge_list))
        edge_index_dict[('protein','to','protein')] = construct_two_edge_list(list(msi.components["protein_to_protein"].edge_list))
        edge_index_dict[('protein', 'to', 'biological_function')] = construct_two_edge_list(list(msi.components["protein_to_biological_function"].edge_list))
        edge_index_dict[('biological_function', 'to', 'biological_function')] = construct_two_edge_list(list(msi.components["biological_function_to_biological_function"].edge_list))

        edge_index_dict[('protein','from','drug')] = construct_two_edge_list(list(msi.components["drug_to_protein"].edge_list),reversed=True)
        edge_index_dict[('protein','from','indication')] = construct_two_edge_list(list(msi.components["indication_to_protein"].edge_list),reversed=True)
        edge_index_dict[('protein','from','protein')] = construct_two_edge_list(list(msi.components["protein_to_protein"].edge_list),reversed=True)
        edge_index_dict[('biological_function', 'from', 'protein')] = construct_two_edge_list(list(msi.components["protein_to_biological_function"].edge_list),reversed=True)
        edge_index_dict[('biological_function', 'from', 'biological_function')] = construct_two_edge_list(list(msi.components["biological_function_to_biological_function"].edge_list),reversed=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MetaPath2Vec(edge_index_dict, embedding_dim=128,
                            metapath=metapath, walk_length=50, context_size=7,
                            walks_per_node=5, num_negative_samples=5,
                            sparse=True).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=12)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train(self,epoch, log_steps=100, eval_steps=2000):
            model.train()

            total_loss = 0
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (i + 1) % log_steps == 0:
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Loss: {total_loss / log_steps:.4f}'))
                    total_loss = 0

                if (i + 1) % eval_steps == 0:
                    acc = test()
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Acc: {acc:.4f}'))

        @torch.no_grad()
        def test(train_ratio=0.1):
            model.eval()

            z = model('author', batch=data.y_index_dict['author'])
            y = data.y_dict['author']

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * train_ratio)]
            test_perm = perm[int(z.size(0) * train_ratio):]

            return model.test(z[train_perm], y[train_perm], z[test_perm],
                            y[test_perm], max_iter=150)

        for epoch in range(1, 6):
            train(epoch)
            #acc = test()
            #print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

if __name__ == "__main__":
    ModelNode2Vec()
    construct_dataset_from_node_embeddings()