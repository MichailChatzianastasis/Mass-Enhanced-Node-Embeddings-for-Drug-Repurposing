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
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset
from diff_prof.diffusion_profiles import DiffusionProfiles
from tests.msi import test_msi
import csv
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import svm 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier

#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GENConv, DeepGCNLayer, GATConv


from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
import torch_geometric.transforms as T

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import init
from torch.nn.parameter import Parameter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import DataReader, Word2vecDataset
import stellargraph as sg
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec


class DeepwalkModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
            super(DeepwalkModel, self).__init__()
            self.emb_size = emb_size
            self.lamb = 0.1
            self.emb_dimension = emb_dimension
            self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
            self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

            self.mass = nn.Embedding(emb_size, 1, sparse=True)

            initrange = 1.0 / self.emb_dimension
            init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
            init.constant_(self.v_embeddings.weight.data, 0)
            # init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
            # init.uniform_(self.mass.weight.data, -initrange, initrange)
            #init.constant_(self.mass.weight.data, 1)
            
    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        #mass = self.mass(pos_u)
        #mass_v = self.mass(pos_v)
        #mass_neg_v = self.mass(neg_v)
        
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        #dist = torch.sum(torch.pow(emb_u-emb_v, 2), dim=1)
        dist = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        #score = mass*mass_v - self.lamb*torch.log(dist)
        # score = mass*mass_v / (self.lamb*torch.pow(dist,2))
        # score = mass + self.lamb*dist
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        #dist = torch.sum(torch.pow(emb_u.unsqueeze(1).repeat(1, 5, 1)-emb_neg_v, 2), dim=2)
        dist = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        #neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) - self.lamb*torch.log(dist)
        # neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) / (self.lamb*torch.pow(dist,2))
        # neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) + self.lamb*dist
        # neg_score = mass.repeat(1, 5)  + self.lamb*dist
        #neg_score = torch.clamp(neg_score, max=10, min=-10)
        #neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        
        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def get_embedding(self, id2word):
        u_embeddings = self.u_embeddings.weight.cpu().data.numpy()
        embeddings = dict()
        for wid, w in id2word.items():
            embeddings[w] = u_embeddings[wid]
        return embeddings


class GravityModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(GravityModel, self).__init__()
        self.emb_size = emb_size
        self.lamb = 0.1
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        self.mass = nn.Embedding(emb_size, 1, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        # init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
        # init.uniform_(self.mass.weight.data, -initrange, initrange)
        init.constant_(self.mass.weight.data, 1)
        
    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        mass = self.mass(pos_u)
        mass_v = self.mass(pos_v)
        mass_neg_v = self.mass(neg_v)
        
        # score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        # score = torch.clamp(score, max=10, min=-10)
        # score = -F.logsigmoid(score)

        # neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        # neg_score = torch.clamp(neg_score, max=10, min=-10)
        # neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        dist = torch.sum(torch.pow(emb_u-emb_v, 2), dim=1)
        # dist = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = mass*mass_v - self.lamb*torch.log(dist)
        # score = mass*mass_v / (self.lamb*torch.pow(dist,2))
        # score = mass + self.lamb*dist
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        dist = torch.sum(torch.pow(emb_u.unsqueeze(1).repeat(1, 5, 1)-emb_neg_v, 2), dim=2)
        # dist = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) - self.lamb*torch.log(dist)
        # neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) / (self.lamb*torch.pow(dist,2))
        # neg_score = mass.repeat(1, 5) * torch.squeeze(mass_neg_v) + self.lamb*dist
        # neg_score = mass.repeat(1, 5)  + self.lamb*dist
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        
        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def get_embedding(self, id2word):
        u_embeddings = self.u_embeddings.weight.cpu().data.numpy()
        embeddings = dict()
        for wid, w in id2word.items():
            embeddings[w] = u_embeddings[wid]
        return embeddings

    # def get_mass(self):
    #     return self.mass.weight.cpu().data.numpy()
    
    def get_mass(self,id2word):
        mass = self.mass.weight.cpu().data.numpy()
        masses = dict()
        for wid, w in id2word.items():
            masses[w] = mass[wid]
        return masses
        # return self.mass.weight.cpu().data.numpy()    


class TrainDeepwalkModel():
   
    # from model_one_mass_vector import SkipGramModel

    def __init__(self,EMB_DIM,EPOCHS):
        super(TrainDeepwalkModel, self).__init__()
        self.edges = []
        self.EMB_DIM = EMB_DIM
        self.EPOCHS = EPOCHS

        self.path_to_graph = "./results/graph.pkl"
        self.path_node2idx = "./results/node2idx.pkl"
        self.path_edges = "./results/edges.pkl"

        #self.construct_msi_dataset_to_pg()
        
        self.main()
  

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()

        with open(self.path_to_graph, 'rb') as handle:
            G = pickle.load(handle)

        with open(self.path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
    
    
        start_node_list = []
        end_node_list = []
        for edge in graph.edges:
                start_node_list.append(node2idx[edge[0]])
                end_node_list.append(node2idx[edge[1]])
        self.edges = torch.tensor([start_node_list,end_node_list])        
        
        output_file = open(self.path_edges,"wb")
        pickle.dump(self.edges,output_file)
        output_file.close()


    def random_walk(self,G, node, walk_length):
        walk = [node]
        for i in range(walk_length):
            neighbors =  list(G.neighbors(walk[i]))
            walk.append(np.random.choice(neighbors))

        walk = [str(node) for node in walk]
        return walk

    def generate_walks(self,G, num_walks, walk_length):
        walks = []
        for i in range(num_walks):
            nodes = G.nodes()
            nodes = np.random.permutation(nodes)
            for j in range(nodes.shape[0]):
                walk = self.random_walk(G, nodes[j], walk_length)
                walks.append(walk)

        return walks

    def main(self):
        # Loads the graph
        msi = MSI()
        msi.load()

        with open(self.path_to_graph, 'rb') as handle:
            G = pickle.load(handle)

        #G = nx.read_weighted_edgelist('web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
        #G = nx.read_edgelist('karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
        
        print("Number of nodes:", G.number_of_nodes())
        print("Number of edges:", G.number_of_edges())
        n = G.number_of_nodes()

        num_walks = 1
        walk_length = 2
        walks = self.generate_walks(G, num_walks=num_walks, walk_length=walk_length)
        print('Generated walks')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = DataReader(walks, min_count=0)
        dataset = Word2vecDataset(walks, data, window_size=4)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=dataset.collate)

        emb_size = len(data.word2id)
        emb_dimension = self.EMB_DIM
        epochs = self.EPOCHS
        initial_lr=0.025

        skipgram_model = DeepwalkModel(emb_size, emb_dimension).to(device)
        optimizer = optim.SparseAdam(list(skipgram_model.parameters()), lr=initial_lr)
        #optimizer = optim.Adam(skipgram_model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        for epoch in range(epochs):

            print("\n\n\nIteration: " + str(epoch + 1))

            loss_all = 0.0
            count = 0
            for i, sample_batched in enumerate(tqdm(dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(device)
                    pos_v = sample_batched[1].to(device)
                    neg_v = sample_batched[2].to(device)
                    
                    optimizer.zero_grad()
                    loss = skipgram_model(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    scheduler.step()

                    #running_loss = running_loss * 0.9 + loss.item() * 0.1
                    #print(" Loss: " + str(running_loss))
                    #print(" Loss: " +loss.item())
                    loss_all += loss.item()
                    count += len(sample_batched[0])
            print("Loss:", loss/count)

        #self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
        embeddings = skipgram_model.get_embedding(data.id2word)

        E = np.zeros((n,emb_dimension))
        for i,node in enumerate(G.nodes()):
            E[i,:] = embeddings[str(node)]
            
         
        node_embeddings = np.array(embeddings)
       

        #fix embeddings
        a = node_embeddings.item()
        
        new_embeddings = np.empty((len(a),emb_dimension))
         
        path_node2idx = "./results/node2idx.pkl"
        with open(path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
        
        for idx,(key,value) in enumerate(a.items()):
            new_embeddings[node2idx[key]] = value
 
 
            
        new_embeddings = new_embeddings.astype(np.float32)

        with open(f'./data/deepwalk_embeddings_{emb_dimension}_{epochs}_{walk_length}_{num_walks}.pickle', 'wb') as handle:
            pickle.dump(new_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

     


class TrainGravityModel():
   
    # from model_one_mass_vector import SkipGramModel

    def __init__(self,EMB_DIM,EPOCHS):
        super(TrainGravityModel, self).__init__()
        self.edges = []
        self.EMB_DIM = EMB_DIM
        self.EPOCHS = EPOCHS

        self.path_to_graph = "./results/graph.pkl"
        self.path_node2idx = "./results/node2idx.pkl"
        self.path_edges = "./results/edges.pkl"

        #self.construct_msi_dataset_to_pg()
        
        self.main()
  

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()

        with open(self.path_to_graph, 'rb') as handle:
            G = pickle.load(handle)

        with open(self.path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
    
    
        start_node_list = []
        end_node_list = []
        for edge in graph.edges:
                start_node_list.append(node2idx[edge[0]])
                end_node_list.append(node2idx[edge[1]])
        self.edges = torch.tensor([start_node_list,end_node_list])        
        
        output_file = open(self.path_edges,"wb")
        pickle.dump(self.edges,output_file)
        output_file.close()


    def random_walk(self,G, node, walk_length):
        walk = [node]
        for i in range(walk_length):
            neighbors =  list(G.neighbors(walk[i]))
            walk.append(np.random.choice(neighbors))

        walk = [str(node) for node in walk]
        return walk

    def generate_walks(self,G, num_walks, walk_length):
        walks = []
        for i in range(num_walks):
            nodes = G.nodes()
            nodes = np.random.permutation(nodes)
            for j in range(nodes.shape[0]):
                walk = self.random_walk(G, nodes[j], walk_length)
                walks.append(walk)

        return walks

    def main(self):
        # Loads the graph
        msi = MSI()
        msi.load()

        with open(self.path_to_graph, 'rb') as handle:
            G = pickle.load(handle)

        #G = nx.read_weighted_edgelist('web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
        #G = nx.read_edgelist('karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
        
        print("Number of nodes:", G.number_of_nodes())
        print("Number of edges:", G.number_of_edges())
        n = G.number_of_nodes()

        num_walks = 1
        walk_length = 2
        walks = self.generate_walks(G, num_walks=num_walks, walk_length=walk_length)
        print('Generated walks')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = DataReader(walks, min_count=0)
        dataset = Word2vecDataset(walks, data, window_size=4)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=dataset.collate)

        emb_size = len(data.word2id)
        emb_dimension = self.EMB_DIM
        epochs = self.EPOCHS
        initial_lr=0.025

        skipgram_model = GravityModel(emb_size, emb_dimension).to(device)
        optimizer = optim.SparseAdam(list(skipgram_model.parameters()), lr=initial_lr)
        #optimizer = optim.Adam(skipgram_model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        for epoch in range(epochs):

            print("\n\n\nIteration: " + str(epoch + 1))

            loss_all = 0.0
            count = 0
            for i, sample_batched in enumerate(tqdm(dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(device)
                    pos_v = sample_batched[1].to(device)
                    neg_v = sample_batched[2].to(device)
                    
                    optimizer.zero_grad()
                    loss = skipgram_model(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    scheduler.step()

                    #running_loss = running_loss * 0.9 + loss.item() * 0.1
                    #print(" Loss: " + str(running_loss))
                    #print(" Loss: " +loss.item())
                    loss_all += loss.item()
                    count += len(sample_batched[0])
            print("Loss:", loss/count)

        #self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
        embeddings = skipgram_model.get_embedding(data.id2word)

        E = np.zeros((n,emb_dimension))
        for i,node in enumerate(G.nodes()):
            E[i,:] = embeddings[str(node)]
            
        #READOUT masses:
        mass = skipgram_model.get_mass(data.id2word)
        masses = np.zeros((n,1))
        for i,node in enumerate(G.nodes()):
            masses[i] = mass[str(node)]
        
        node_embeddings = np.array(embeddings)
       

        #fix embeddings
        a = node_embeddings.item()
        
        new_embeddings = np.empty((len(a),emb_dimension))
        new_masses = np.empty((len(masses),len(masses[0])))
        
        path_node2idx = "./results/node2idx.pkl"
        with open(path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
        
        for idx,(key,value) in enumerate(a.items()):
            new_embeddings[node2idx[key]] = value
            new_masses[node2idx[key]] = masses[idx]

 
            
        new_embeddings = new_embeddings.astype(np.float32)

        with open(f'./data/gravity_embeddings_{emb_dimension}_{epochs}_{walk_length}_{num_walks}.pickle', 'wb') as handle:
            pickle.dump(new_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'./data/gravity_masses_{emb_dimension}_{epochs}_{walk_length}_{num_walks}.pickle', 'wb') as handle:
            pickle.dump(new_masses, handle, protocol=pickle.HIGHEST_PROTOCOL)

class ModelNode2Vec():

    def __init__(self,EMB_DIM,EPOCHS):
        super(ModelNode2Vec, self).__init__()
        self.edges = []
        self.EMB_DIM = EMB_DIM
        self.EPOCHS = EPOCHS

        self.path_to_graph = "./results/graph.pkl"
        self.path_node2idx = "./results/node2idx.pkl"
        self.path_edges = "./results/edges.pkl"

        #self.construct_msi_dataset_to_pg()
        
        self.train_model()
  

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()

        with open(self.path_to_graph, 'rb') as handle:
            graph = pickle.load(handle)

        with open(self.path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)
    
    
        start_node_list = []
        end_node_list = []
        for edge in graph.edges:
                start_node_list.append(node2idx[edge[0]])
                end_node_list.append(node2idx[edge[1]])
        self.edges = torch.tensor([start_node_list,end_node_list])        
        
        output_file = open(self.path_edges,"wb")
        pickle.dump(self.edges,output_file)
        output_file.close()

    def train_model(self):

        with open(self.path_edges,"rb") as handle:
            self.edges = pickle.load(handle)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #default walk_length=20,context_size=10,walks_per_node=10,p=q=1,,
        #sequenece: walk_length,context_size,walks_per_node,num_negative_samples,p,q


        combinations = [[20,10,10,1,1,1],[20,10,10,5,1,1],[20,10,10,10,1,1],[30,10,10,1,1,1]]      
        for combination in combinations:
            print(combination)
            model = Node2Vec(self.edges, embedding_dim=self.EMB_DIM, walk_length=combination[0],
                            context_size=combination[1], walks_per_node=combination[2],
                            num_negative_samples=combination[3], p=combination[4], q=combination[5], sparse=True).to(device)
        
            loader = model.loader(batch_size=128, shuffle=True, num_workers=12)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            model.train()
            for epoch in range(1, self.EPOCHS):
                
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
            with open(f'./data/node2vec_embeddings_{self.EMB_DIM}_{self.EPOCHS}_{str(combination[0])}_{str(combination[1])}_{str(combination[2])}_{str(combination[3])}_{str(combination[4])}_{str(combination[5])}.pickle', 'wb') as handle:
                pickle.dump(node_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
class MLP(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLP, self).__init__()      
        self.fc1 = nn.Linear(2*EMB_DIM, EMB_DIM)
        self.fc2=  nn.Linear(EMB_DIM, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x) 

class CNN(nn.Module):
    def __init__(self,EMB_DIM):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(1,1,3,dilation=128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(1,1,3)
        self.fc1 = nn.Linear(126,50)
        self.fc2 = nn.Linear(50,1)

    def forward(self,x):
        x = self.relu(self.conv1(torch.unsqueeze(x,1)))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class MLP2(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLP2, self).__init__()      
        self.fc1 = nn.Linear(2*EMB_DIM, 30)
        self.fc2 = nn.Linear(30,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchNorm = nn.BatchNorm1d(EMB_DIM)

        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)

class MLPSet(nn.Module):
    def __init__(self,EMB_DIM):
        super(MLPSet,self).__init__()
        self.fc1 = nn.Linear(EMB_DIM,100)
        self.fc2 = nn.Linear(100,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        x = torch.square(x[:,0] - x[:,1]) 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)        
        return torch.sigmoid(x)

class MLPSetBilnear(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLPSetBilnear, self).__init__()
        self.OUT_DIM = 60     
        self.EMB_DIM = EMB_DIM
        
        self.fc1 = nn.Bilinear(self.EMB_DIM, self.EMB_DIM, self.EMB_DIM)
        self.fc2=  nn.Linear(self.EMB_DIM, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x[:,0,:],x[:,1,:])))
        x = self.fc2(x)
        return torch.sigmoid(x)

class Dataset_Mlp(Dataset):
    def __init__(self,node_embeddings_model="node2vec",node_embeddings_path="./data/node2vec_embeddings_256_200_10.pickle",EMB_DIM=256,downstream_model="mlp",mlp_model=None,mlp_epochs=1):
        #self.x = torch.tensor(x,dtype=torch.float32)
        #self.y = torch.tensor(y,dtype=torch.float32)
        #self.length = self.x.shape[0]
        self.node_embeddings_path = node_embeddings_path
        self.EMB_DIM = EMB_DIM
        self.downstream_model = downstream_model
        self.mlp_model = mlp_model
        self.mlp_epochs = mlp_epochs
        self.node_embeddings_model = node_embeddings_model
        
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length

    def kfold_split(self,pairs:dict, perc:float, shuffle:bool) -> list:
        keys = list(pairs.keys())
        sets = len(keys)
        cv_perc = int(sets*perc)
        folds = int(sets/cv_perc)

        indices = []         
        for fold in range(folds):
        
            # If you want to generate random keys
            if shuffle:
                # Choose random keys 
                random_keys = list(np.random.choice(keys, cv_perc))
                other_keys = list(set(keys) - set(random_keys)) 
                indices.append((other_keys, random_keys))
            else: 
                if fold == 0: 
                    fold_keys = keys[-cv_perc*(fold+1):]
                else:
                    fold_keys = keys[-cv_perc*(fold+1):-cv_perc*(fold)]
                other_keys = list(set(keys) - set(fold_keys)) 
                indices.append((other_keys, fold_keys))             
        return indices

    def construct_dataset_from_node_embeddings(self):
 

        '''
        Construct a dataset X = (emb_drug,emb_disease) , y=0 or 1 (if drug treat the disease). 
        It takes as argument the path of the node embeddings
        '''
        
        if(self.node_embeddings_model=="diffusion_profiles"):
            msi = MSI()
            msi.load()
            dp_saved = DiffusionProfiles(alpha = None, max_iter = None, tol = None, weights = None, num_cores = None, save_load_file_path = "./data/10_top_msi/")
            msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
            dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)
            node_embeddings = dp_saved.drug_or_indication2diffusion_profile
        
        else:
            with open(self.node_embeddings_path, 'rb') as handle:
                node_embeddings = pickle.load(handle)

        path_to_graph = "./results/graph.pkl"
        path_node2idx = "./results/node2idx.pkl"

        with open(path_to_graph, 'rb') as handle:
            graph = pickle.load(handle)

        with open(path_node2idx, 'rb') as handle:
            node2idx = pickle.load(handle)

        with open("./data/disease_codes_dict.pickle",'rb') as handle:
            disease_codes_dict = pickle.load(handle)

        with open("./data/drug_codes_dict.pickle",'rb') as handle:
            drug_codes_dict = pickle.load(handle)

        with open('./data/disease_to_drug_dict.pickle', 'rb') as handle:
            disease_to_drug_dict = pickle.load(handle)

        with open('./data/drug_to_disease_dict.pickle', 'rb') as handle:
            drug_to_disease_dict = pickle.load(handle)

        
 

        #five cross validation
        #result_indices = self.kfold_split(pairs=drug_to_disease_dict, perc=0.2, shuffle=True)  # [ ([key1train,key2train,..],[ktest,ktest2]) fold1 , 
        with open('./data/results_indices.pickle','rb') as handle:
            result_indices = pickle.load(handle)
                                                                                          #   ([],[])  fold2 ...]
        mean_average_precision = []
        average_recall50 = []
        median_rocauc = []
        for indices in result_indices: #for every fold
            train_indices, test_indices = indices
            X_positive_codes_train = [] # [[drug,disease],[drug,disease]...etc] codes of positive examples
            X_positive_codes_test = []
            for key,value in drug_to_disease_dict.items(): #key = drug ,  value is a list of [disease]
                if(key in train_indices):
                    for disease in value:
                        X_positive_codes_train.append([key,disease[0]])
                else:
                    for disease in value:
                        X_positive_codes_test.append([key,disease[0]])
            
            #generate negative examples.(we assume that if a pair (drug,disease) is absense from the initial dataset, then the drug do not cure the disease. So its negative example)
            # number_drugs = 1661
            # number_diseases = 840
            # positive examples = 5926
            print("Positive examples generated")
            #for every drug we peak randomly 7=5926/840  diseases until that are not positive examples.
            X_negative_codes_train = []
            X_negative_codes_test = []
            
            diseases_list = list(disease_codes_dict)
            diseases_mask = np.zeros((len(diseases_list)))


            for disease in disease_to_drug_dict:
                counter = 0
                #40
                while(counter<40):
                    positive = False
                    random_key = random.choice(list(drug_codes_dict.keys()))
                    approved_drugs = disease_to_drug_dict[disease]
                    for approved_drug in approved_drugs:
                        if(approved_drug[1] == random_key):
                            positive = True
                    if(positive == False):
                        if(random_key in train_indices):
                            X_negative_codes_train.append([random_key,disease])
                        else:
                            X_negative_codes_test.append([random_key,disease])
                        counter+=1 
            for drug in drug_to_disease_dict:
                counter = 0
                #3
                while(counter<3):
                    positive = False
                    #generate random disease code
                    random_key = random.choice(list(disease_codes_dict.keys()))
                    #check if drug,random_key belong to positive example
                    diseases = drug_to_disease_dict[drug]
                    for disease in diseases:
                        if(random_key == disease[0]):
                            positive = True
                    if(positive == False):
                        if(drug in train_indices):
                            X_negative_codes_train.append([drug,random_key])
                            disease_codes_dict[random_key] += 1
                        else:
                            X_negative_codes_test.append([drug,random_key])
                        counter+= 1
            
            
            print("Negative examples generated")
            #print(len(X_negative_codes)) # this should close to len(X_positive_codes)
            # construct X_input. 
            #for every code take the embedding[code]

            X_positive_train = []
            X_negative_train = []
            X_positive_test = []
            X_negative_test = []

            #node_embeddings = node_embeddings.item()
            
            for sample in X_positive_codes_train:
                if(self.node_embeddings_model=="diffusion_profiles"):
                    a = torch.tensor(node_embeddings[sample[0]],dtype=torch.float)
                    b = torch.tensor(node_embeddings[sample[1]],dtype=torch.float)
                else:
                    
                    a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                    b = torch.tensor(node_embeddings[node2idx[sample[1]]])
                X_positive_train.append(torch.stack([a,b]))
            
            for sample in X_negative_codes_train:
                if(self.node_embeddings_model=="diffusion_profiles"):
                    a = torch.tensor(node_embeddings[sample[0]],dtype=torch.float)
                    b = torch.tensor(node_embeddings[sample[1]],dtype=torch.float)
                else:
                    a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                    b = torch.tensor(node_embeddings[node2idx[sample[1]]])
                X_negative_train.append(torch.stack([a,b]))
            
            for sample in X_positive_codes_test:
                if(self.node_embeddings_model=="diffusion_profiles"):
                    a = torch.tensor(node_embeddings[sample[0]],dtype=torch.float)
                    b = torch.tensor(node_embeddings[sample[1]],dtype=torch.float)
                else:
                    a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                    b = torch.tensor(node_embeddings[node2idx[sample[1]]])          
                X_positive_test.append(torch.stack([a,b]))

            for sample in X_negative_codes_test:
                if(self.node_embeddings_model=="diffusion_profiles"):
                    a = torch.tensor(node_embeddings[sample[0]],dtype=torch.float)
                    b = torch.tensor(node_embeddings[sample[1]],dtype=torch.float)
                else:
                    a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                    b = torch.tensor(node_embeddings[node2idx[sample[1]]])
                X_negative_test.append(torch.stack([a,b]))
            
            #with open('/data/multiscale-interactome/data/X_positive.pickle', 'wb') as handle:
            #    pickle.dump(X_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #with open('/data/multiscale-interactome/data/X_negative.pickle', 'wb') as handle:
            #    pickle.dump(X_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
            X_train = X_positive_train + X_negative_train
            X_test = X_positive_test + X_negative_test
        
            y_train = torch.tensor(([1 for i in range(len(X_positive_train))] + [0 for i in range(len(X_negative_train))]))
            y_test = torch.tensor(([1 for i in range(len(X_positive_test))] + [0 for i in range(len(X_negative_test))]))

            print("Train",len(X_positive_train),len(X_negative_train))
            print("Test",len(X_positive_test),len(X_negative_test))
            
           
            if(self.downstream_model == "gnn"):
                res = train_GNN(X_positive_codes_train,X_negative_codes_train,X_positive_codes_test,X_negative_codes_test,
                                                                                                    node_embeddings,
                                                                                                    indices,
                                                                                                    self.node_embeddings_path, 
                                                                                                    self.EMB_DIM, 
                                                                                                    self.node_embeddings_model,
                                                                                                    self.downstream_model,
                                                                                                    self.mlp_model,
                                                                                                    self.mlp_epochs)
            elif(self.downstream_model == "mlp" or self.downstream_model == "no"): 
                res = train_MLP(torch.stack((X_train)), torch.stack((X_test)), y_train, y_test, indices,
                                                                                                    self.node_embeddings_path,
                                                                                                    self.EMB_DIM,
                                                                                                    self.node_embeddings_model,
                                                                                                    self.downstream_model,
                                                                                                    self.mlp_model,
                                                                                                    self.mlp_epochs
                                                                                                    )

            elif(self.downstream_model == "svm" or self.downstream_model == "rf" or self.downstream_model == "ada"):
                res = train_sklearn_classifiers(torch.stack((X_train)), torch.stack((X_test)), y_train, y_test, indices,
                                                                                                    self.node_embeddings_path,
                                                                                                    self.EMB_DIM,
                                                                                                    self.node_embeddings_model,
                                                                                                    self.downstream_model,
                                                                                                    self.mlp_model,
                                                                                                    self.mlp_epochs
                                                                                                    )                                                                                                   
                                                                                                
            mean_average_precision.append(res[0])
            average_recall50.append(res[1])
            median_rocauc.append(res[2])

        final_ap = sum(mean_average_precision) / len(mean_average_precision)
        final_ar = sum(average_recall50) / len(average_recall50)
        final_roc = sum(median_rocauc) / len(median_rocauc)
        
        print("Final average results accross 5 cross validation : ")
        print("Mean average precision:", final_ap)
        print("Average recall50:", final_ar)
        print("Median rocauc", final_roc)
        with open('./data/results.txt','a') as handle:
            handle.write(f'{self.node_embeddings_path} {self.node_embeddings_model} {self.downstream_model} {self.mlp_model}_{self.mlp_epochs} {final_ap} {final_ar} {final_roc}\n')


def train_sklearn_classifiers(X_train,X_test,y_train,y_test,indices,node_embeddings_path,EMB_DIM,node_embeddings_model,downstream_model,mlp_model,mlp_epochs):
    from evaluate import evaluate_model
    from evaluate import construct_disease_drug_tsv


    dataset_train = np.reshape(X_train,(-1,2*X_train.shape[-1]))
    dataset_test =  np.reshape(X_test,(-1,2*X_test.shape[-1]))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #forward loop
    losses = []
    accur = []
    losses_test = []
    accur_test = []


    #model = svm.SVC(probability=True)
    #model = RandomForestClassifier(random_state=0)
    #model = AdaBoostClassifier()
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(dataset_train,y_train)

    #torch.save(model.state_dict(), "./data/rf_model_"+str(EMB_DIM))
    construct_disease_drug_tsv()
    return evaluate_model(  model=node_embeddings_model, 
                            downstream_model=downstream_model, 
                            mlp_model_name=model,
                            node_embeddings_path=node_embeddings_path,
                            EMB_DIM=EMB_DIM, 
                            indices=indices
                            )


def train_MLP(X_train,X_test,y_train,y_test,indices,node_embeddings_path,EMB_DIM,node_embeddings_model,downstream_model,mlp_model,mlp_epochs):
    from evaluate import evaluate_model
    from evaluate import construct_disease_drug_tsv
    
    if(downstream_model == "mlp"):
        if(mlp_model.startswith("MLPSet") == False):
            dataset_train = TensorDataset(torch.reshape(X_train,(-1,2*X_train.shape[-1])),y_train.type(torch.FloatTensor))
            dataset_test =  TensorDataset(torch.reshape(X_test,(-1,2*X_test.shape[-1])),y_test.type(torch.FloatTensor))
        else:
            dataset_train = TensorDataset(X_train,y_train.type(torch.FloatTensor))
            dataset_test =  TensorDataset(X_test,y_test.type(torch.FloatTensor))
            
        #DataLoader
        trainloader = DataLoader(dataset_train,batch_size=16,shuffle=True)
        testloader = DataLoader(dataset_test,batch_size=16,shuffle=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        epochs = mlp_epochs
        model = eval(mlp_model)(EMB_DIM).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
       
        #forward loop
        model.train()
        losses = []
        accur = []
        losses_test = []
        accur_test = []
        n_batches = len(trainloader)
        n_batches_test = len(testloader)
        for i in range(epochs):
            for j,(x_train,y_train) in enumerate(trainloader):
                x_train,y_train = x_train.to(device),y_train.to(device)
                #zero gradients
                optimizer.zero_grad()
                #calculate output
                output = model(x_train).squeeze()
                #calculate loss
                loss = criterion(output,y_train.squeeze())
                #accuracy
                correct = (torch.round(output) == y_train).float().sum().detach().item()
                acc = correct / y_train.shape[0]
                #backprop
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
                accur.append(acc)
            print("Train: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(losses[n_batches*i:n_batches*(i+1)]),np.mean(accur[n_batches*i:n_batches*(i+1)])))
            model.eval()
            with torch.no_grad():
                for j,(x_test,y_test) in enumerate(testloader):
                    x_test,y_test = x_test.to(device),y_test.to(device)
                    output = model(x_test).squeeze()
                    #calculate loss
                    loss = criterion(output,y_test.squeeze())
                    #accuracy
                    correct = (torch.round(output) == y_test).float().sum().detach().item()
                    acc = correct / y_test.shape[0]
                    losses_test.append(loss.detach().item())
                    accur_test.append(acc)
                print("Test: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(losses_test[n_batches_test*i:n_batches_test*(i+1)]),np.mean(accur_test[n_batches_test*i:n_batches_test*(i+1)])))         
        torch.save(model.state_dict(), "./data/MLP_model_"+str(EMB_DIM))
    construct_disease_drug_tsv()
    return evaluate_model(  model=node_embeddings_model, 
                            downstream_model=downstream_model, 
                            mlp_model_name=mlp_model,
                            node_embeddings_path=node_embeddings_path,
                            EMB_DIM=EMB_DIM, 
                            indices=indices
                            )

class GNN(torch.nn.Module):
    '''
    GNN for node pair classification. It propagates the graph, and predicts if a certain disease treat a certain drug. (like the MLP above)
    Experiments with and without node features
    '''
    def __init__(self,node_features,edge_index,num_features,final_node_dimensions,gnn_embeddings_path, evaluation=False):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features ,32 , heads=8)
        self.conv2 = GATConv(32*8, 32, heads=8, concat=False)
        self.conv3 = GATConv(32, 32)

        self.linear = nn.Linear(64,1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.node_features = node_features
        self.edge_index = edge_index
        self.gnn_embeddings_path = gnn_embeddings_path
        if(evaluation == True) :
            self.gnn_embeddings = self.load_gnn_embeddings()

    def forward(self, pair_index, save_gnn_embeddings_bool=False, evaluation=False):
        if(evaluation == False):
            x = self.dropout(self.relu(self.conv1(self.node_features, self.edge_index)))
            x = self.dropout(self.relu(self.conv2(x,self.edge_index)))
            x = self.dropout(self.relu(self.conv3(x,self.edge_index)))

            if(save_gnn_embeddings_bool == True):
                self.save_gnn_embeddings(x)
        else:
            x = self.gnn_embeddings
        #compute the pair representation
        #aggregators:
        #1) multiply
        #pair_repr = torch.mul(x[pair_index[:,0]],x[pair_index[:,1]])
        #2) concat
        pair_repr = torch.cat((x[pair_index[:,0]],x[pair_index[:,1]]),dim=-1)
        #3) difference
        #pair_repr = torch.sub(x[pair_index[:,0]], x[pair_index[:,1]])
        x = self.linear(pair_repr)
        return torch.sigmoid(x)

    def save_gnn_embeddings(self,x):
        with open(self.gnn_embeddings_path, "wb") as handle:
            pickle.dump(x,handle,protocol=pickle.HIGHEST_PROTOCOL)

    def load_gnn_embeddings(self):
        with open(self.gnn_embeddings_path, "rb") as handle:
            return pickle.load(handle)

class Dataset_GNN():
    def __init__(self,node_embeddings={}):
        super(Dataset_GNN, self).__init__()
        self.edges = []
        self.x = [] # node attributes : one-hot vectors indicating the node type ( drug,disease,protein,biological function)
        self.msi = []
        
        self.construct_msi_dataset_to_pg(node_embeddings)

    def idx2onehot(self,idx):
        result = []
        if(type(idx) == list):
            for i in idx:
                result.append(self.msi.type2onehot[self.msi.node2type[self.msi.idx2node[i]]])
        else:
            result = self.msi.type2onehot[self.msi.node2type[self.msi.idx2node[idx]]]
        
        return result

    def construct_msi_dataset_to_pg(self,node_embeddings):
        #self.msi = MSI()
        #self.msi.load()
        with open('./data/msi.pickle', 'rb') as handle:
            self.msi = pickle.load(handle)
        
        self.msi.type2onehot = { "drug" : [1,0,0,0],
                                "indication" : [0,1,0,0],
                                "protein" : [0,0,1,0],
                                "biological_function" : [0,0,0,1]
                                }

        #construct edges
        start_node_list = []
        end_node_list = []
        for edge in self.msi.graph.edges:
                start_node_list.append(self.msi.node2idx[edge[0]])
                end_node_list.append(self.msi.node2idx[edge[1]])
        self.edge_index = torch.tensor([start_node_list,end_node_list],dtype=torch.long)    
        
        #construct node features (one hot indicating node type or node embeddings from node2vec etc)
        features = []
        if(len(node_embeddings)==0):
            for idx in self.msi.idx2node:
                features.append(self.idx2onehot(idx))
            self.num_features = 4
        else:
            for idx in self.msi.idx2node:
                features.append(node_embeddings[idx])
            self.num_features = len(node_embeddings[0])
        self.x = torch.tensor(features, dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index)      

        #self.loader = DataLoader([self.data], batch_size=32)



def train_GNN(X_positive_codes_train,X_negative_codes_train,X_positive_codes_test,X_negative_codes_test,
                                                                                    node_embeddings,
                                                                                    indices,
                                                                                    node_embeddings_path, 
                                                                                    EMB_DIM, 
                                                                                    node_embeddings_model,
                                                                                    downstream_model,
                                                                                    mlp_model,
                                                                                    mlp_epochs
                                                                                    ):        
    from evaluate import evaluate_model
    from evaluate import construct_disease_drug_tsv
    
    dataset_gnn = Dataset_GNN(node_embeddings)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for sample in X_positive_codes_train:
        start = torch.tensor(dataset_gnn.msi.node2idx[sample[0]])
        end = torch.tensor(dataset_gnn.msi.node2idx[sample[1]])
        X_train.append(torch.stack([start,end]))
        y_train.append(1)

    for sample in X_negative_codes_train:
        start = torch.tensor(dataset_gnn.msi.node2idx[sample[0]])
        end = torch.tensor(dataset_gnn.msi.node2idx[sample[1]])
        X_train.append(torch.stack([start,end]))
        y_train.append(0)

    for sample in X_positive_codes_test:
        start = torch.tensor(dataset_gnn.msi.node2idx[sample[0]])
        end = torch.tensor(dataset_gnn.msi.node2idx[sample[1]])
        X_test.append(torch.stack([start,end]))
        y_test.append(1)

    for sample in X_negative_codes_test:
        start = torch.tensor(dataset_gnn.msi.node2idx[sample[0]])
        end = torch.tensor(dataset_gnn.msi.node2idx[sample[1]])
        X_test.append(torch.stack([start,end]))
        y_test.append(0)

    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)


    dataset_train = TensorDataset(torch.stack(X_train),y_train.type(torch.FloatTensor))
    dataset_test = TensorDataset(torch.stack(X_test),y_test.type(torch.FloatTensor))

    #DataLoader
    trainloader = DataLoader(dataset_train,batch_size=128,shuffle=True)
    testloader = DataLoader(dataset_test,batch_size=128,shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_embeddings_path = "./data/gnn_embeddings.pickle"
    model = GNN(node_features = dataset_gnn.data.x.to(device),
                edge_index = dataset_gnn.data.edge_index.to(device),
                num_features = dataset_gnn.num_features,
                final_node_dimensions = 64,
                gnn_embeddings_path = gnn_embeddings_path
                ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #forward loop
    epochs = mlp_epochs
    model.train()
    losses = []
    accur = []
    losses_test = []
    accur_test = []
    n_batches = len(trainloader)
    n_batches_test = len(testloader)


    for i in range(epochs):
        for j,(x_train,y_train) in enumerate(trainloader): #x_train contains the pairs of nodes to evaluate
            x_train, y_train = x_train.to(device),y_train.to(device)
            #zero gradients
            optimizer.zero_grad()
            #calculate output
            output = model(x_train).squeeze()
            #calculate loss
            loss = criterion(output,y_train.squeeze())
            #accuracy
            correct = (torch.round(output) == y_train).float().sum().detach().item()
            acc = correct / y_train.shape[0]
            #backprop
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            accur.append(acc)
        print("Train: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(losses[n_batches*i:n_batches*(i+1)]),np.mean(accur[n_batches*i:n_batches*(i+1)])))
        model.eval()
        with torch.no_grad():
            for j,(x_test,y_test) in enumerate(testloader):
                x_test, y_test = x_test.to(device),y_test.to(device)
                
                output = model(x_test).squeeze()
                #calculate loss
                loss = criterion(output,y_test.squeeze())
                #accuracy
                correct = (torch.round(output) == y_test).float().sum().detach().item()
                acc = correct / y_test.shape[0]
                losses_test.append(loss.detach().item())
                accur_test.append(acc)
            print("Test: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(losses_test[n_batches_test*i:n_batches_test*(i+1)]),np.mean(accur_test[n_batches_test*i:n_batches_test*(i+1)])))         
        torch.save(model.state_dict(), "./data/GNN_model_"+str(EMB_DIM))  
    construct_disease_drug_tsv()
    model.eval()
    sample_x, _ = iter(trainloader).next()
    model(sample_x.to(device),save_gnn_embeddings_bool=True) #save gnn_embeddings for fast evaluation
    return evaluate_model( model=node_embeddings_model, 
                            downstream_model=downstream_model, 
                            mlp_model_name=mlp_model,
                            node_embeddings_path=node_embeddings_path,
                            EMB_DIM=EMB_DIM, 
                            indices=indices,
                            gnn_params = { "node_features" : dataset_gnn.data.x,
                                        "edge_index" : dataset_gnn.data.edge_index,
                                        "num_features" : dataset_gnn.num_features,
                                        "final_node_dimensions" : 64,
                                        "gnn_embeddings_path" : gnn_embeddings_path
                                        } 
                            )
    
    
class ModelMetaPath2Vec():
    "NEED FIX"
    def __init__(self):
        super(ModelMetaPath2Vec, self).__init__()
        self.node2idx= []
        self.graph = []
        self.model = []
        self.loader = []
        self.optimizer = []

        self.construct_msi_dataset_to_pg()

    def construct_msi_dataset_to_pg(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()
        
        
        path_to_graph = "./results/graph.pkl"
        path_node2idx = "./results/node2idx.pkl"

        with open(path_to_graph, 'rb') as handle:
            self.graph = pickle.load(handle)

        with open(path_node2idx, 'rb') as handle:
            self.node2idx = pickle.load(handle)
        
        
        '''
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
        '''
        metapaths = [ 
            ['drug','protein','drug'],
            ['drug','protein','indication','drug'],
            ['drug','protein','protein','indication','drug'],
            ['drug','protein','biological_function','indication','drug'],
            ['drug','protein','protein','biological_function','indication','drug'],
            ['indication','protein','indication'],
            ['indication','protein','drug','indication'],
            ['indication','protein','protein','drug','indication'],
            ['indication','protein','biological_function','drug','indication'],
            ['indication','protein','protein','biological_function','drug','indication'],
        ]

        stellar = True

        g = sg.StellarGraph.from_networkx(self.graph)

        # Create the random walker
        rw = UniformRandomMetaPathWalk(g)
        print(g.info())
        walk_length = 50
        walks = rw.run(
            nodes=list(g.nodes()),  # root nodes
            length=walk_length,  # maximum length of a random walk
            n=5,  # number of random walks per root node
            metapaths=metapaths,  # the metapaths
        )



        if(stellar):
            print(walks)
            model = Word2Vec(walks, window=5, min_count=0, sg=1, workers=2, epochs=1)
        else:
            edge_index_dict = {}


            edge_index_dict[('protein','to','protein')] = self.construct_two_edge_list(list(msi.components["protein_to_protein"].edge_list))
            edge_index_dict[('protein','from','protein')] = self.construct_two_edge_list(list(msi.components["protein_to_protein"].edge_list),
                                                                                        reversed=True)

            edge_index_dict[('drug','to','protein')] = self.construct_two_edge_list(list(msi.components["drug_to_protein"].edge_list))
            edge_index_dict[('protein','from','drug')] = self.construct_two_edge_list(list(msi.components["drug_to_protein"].edge_list),
                                                                                    reversed=True)

            edge_index_dict[('indication','to','protein')] = self.construct_two_edge_list(list(msi.components["indication_to_protein"].edge_list))
            edge_index_dict[('protein','from','indication')] = self.construct_two_edge_list(list(msi.components["indication_to_protein"].edge_list),
                                                                                            reversed=True)


            edge_index_dict[('protein', 'to', 'biological_function')] = self.construct_two_edge_list(list(msi.components["protein_to_biological_function"].edge_list))
            edge_index_dict[('biological_function', 'from', 'protein')] = self.construct_two_edge_list(list(msi.components["protein_to_biological_function"].edge_list),
                                                                                                    reversed=True)
                                                                                                    
            edge_index_dict[('biological_function', 'to', 'biological_function')] = self.construct_two_edge_list(list(msi.components["biological_function_to_biological_function"].edge_list))
            edge_index_dict[('biological_function', 'from', 'biological_function')] = self.construct_two_edge_list(list(msi.components["biological_function_to_biological_function"].edge_list),
                                                                                                                reversed=True)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'


            self.loader = self.model.loader(batch_size=2, shuffle=True, num_workers=4)
            self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
            print(len(self.loader.dataset))
            for k,v in edge_index_dict.items():
                print(v.shape)

            for epoch in range(1, 6):
                self.train(epoch)
                #acc = test()
                #print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

    #construction of edge_index_dict 
    def construct_two_edge_list(self,edge_list,reversed=False):
        start_node_list = []
        end_node_list = []
        for edge in edge_list:
            if(reversed==True):
                start_node_list.append(self.node2idx[edge[1]])
                end_node_list.append(self.node2idx[edge[0]])
            else:
                start_node_list.append(self.node2idx[edge[0]])
                end_node_list.append(self.node2idx[edge[1]])
        return torch.tensor([start_node_list,end_node_list])


    def train(self,epoch,log_steps=100, eval_steps=2000):
        self.model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(self.loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

            if (i + 1) % eval_steps == 0:
                acc = test()
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(self.loader)}, '
                    f'Acc: {acc:.4f}'))

    @torch.no_grad()
    def test(self,train_ratio=0.1):
        self.model.eval()

        z = self.model('author', batch=data.y_index_dict['author'])
        y = data.y_dict['author']

        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * train_ratio)]
        test_perm = perm[int(z.size(0) * train_ratio):]

        return self.model.test(z[train_perm], y[train_perm], z[test_perm],
                        y[test_perm], max_iter=150)

class Hin2Vec():
    def __init__(self):
        super(Hin2Vec, self).__init__()
        self.construct_csv()

    def construct_csv(self):
        '''
        Construct pytorch geometric dataset Data from msi.
        '''
        msi = MSI()
        msi.load()
        
        
        path_to_graph = "./results/graph.pkl"
        path_node2idx = "./results/node2idx.pkl"

        with open(path_to_graph, 'rb') as handle:
            self.graph = pickle.load(handle)

        with open(path_node2idx, 'rb') as handle:
            self.node2idx = pickle.load(handle)
        
        
        metapath = [
            ('drug','protein'),
            ('indication','protein'),
            ('protein','protein'),
            ('protein','biological_function'),
            ('biological_function', 'biological_function'),
        ]

        dest_node = []
        source_node = []
        weight = []
        source_class = []
        dest_class = []
        edge_class = []

        for edge in list(msi.components["drug_to_protein"].edge_list):
            source_node.append(self.node2idx[edge[0]])
            dest_node.append(self.node2idx[edge[1]])
            weight.append(1)
            source_class.append("drug")
            dest_class.append("protein")
            edge_class.append("drug_protein")
        
        for edge in list(msi.components["indication_to_protein"].edge_list):
            source_node.append(self.node2idx[edge[0]])
            dest_node.append(self.node2idx[edge[1]])
            weight.append(1)
            source_class.append("indication")
            dest_class.append("protein")
            edge_class.append("indication_protein")

        for edge in list(msi.components["protein_to_protein"].edge_list):
            source_node.append(self.node2idx[edge[0]])
            dest_node.append(self.node2idx[edge[1]])
            weight.append(1)
            source_class.append("protein")
            dest_class.append("protein")
            edge_class.append("protein_protein")

        for edge in list(msi.components["protein_to_biological_function"].edge_list):
            source_node.append(self.node2idx[edge[0]])
            dest_node.append(self.node2idx[edge[1]])
            weight.append(1)
            source_class.append("protein")
            dest_class.append("biological_function")
            edge_class.append("protein_to_biological_function")

        for edge in list(msi.components["biological_function_to_biological_function"].edge_list):
            source_node.append(self.node2idx[edge[0]])
            dest_node.append(self.node2idx[edge[1]])
            weight.append(1)
            source_class.append("biological_function")
            dest_class.append("biological_function")
            edge_class.append("biological_function_to_biological_function")

        counter = 0 
        rows = zip(dest_node,source_node,weight,source_class,dest_class,edge_class)
        with open("./data/hin2vec.csv", 'w') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(["","dest_node","source_node","weight","source_class","dest_class","edge_class"]) #header
            for row in rows:
                writer.writerow((counter,*row))
                counter+=1


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * in_channels, cached=True)
        self.conv2 = GCNConv(2 * in_channels, 4 * in_channels, cached=True)
        self.conv3 = GCNConv(4 * in_channels, 8 * in_channels, cached=True)

        self.conv_mu = GCNConv(8 * in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(8 * in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x,edge_index).relu()
        x = self.conv3(x,edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class DatasetAutoencoder():
    def __init__(self,EMB_DIM,EPOCHS):
        super(DatasetAutoencoder, self).__init__()
        self.edges = []
        self.x = [] # node attributes : one-hot vectors indicating the node type ( drug,disease,protein,biological function)
        self.msi = []
        self.EMB_DIM = EMB_DIM
        self.EPOCHS = EPOCHS
        
        self.construct_msi_dataset_to_pg()
        self.configure_train()

    def idx2onehot(self,idx):
        result = []
        if(type(idx) == list):
            for i in idx:
                result.append(self.msi.type2onehot[self.msi.node2type[self.msi.idx2node[i]]])
        else:
            result = self.msi.type2onehot[self.msi.node2type[self.msi.idx2node[idx]]]
        
        return result

    def construct_msi_dataset_to_pg(self):
        #self.msi = MSI()
        #self.msi.load()
        with open('./data/msi.pickle', 'rb') as handle:
            self.msi = pickle.load(handle)
        
        self.msi.type2onehot = { "drug" : [1,0,0,0],
                                "indication" : [0,1,0,0],
                                "protein" : [0,0,1,0],
                                "biological_function" : [0,0,0,1]
                                }

        #construct edges
        start_node_list = []
        end_node_list = []
        for edge in self.msi.graph.edges:
                start_node_list.append(self.msi.node2idx[edge[0]])
                end_node_list.append(self.msi.node2idx[edge[1]])
        self.edge_index = torch.tensor([start_node_list,end_node_list],dtype=torch.long)    
        
        #construct node features (one hot indicating node type)
        features = []
        for idx in self.msi.idx2node:
            features.append(self.idx2onehot(idx))
        self.x = torch.tensor(features,dtype=torch.float)
        self.data = Data(x=self.x,edge_index=self.edge_index)      
        self.data.train_mask = self.data.val_mask = self.data.test_mask = None
        self.data = train_test_split_edges(self.data)

        #self.loader = DataLoader([self.data], batch_size=32)
     
    def configure_train(self):
        num_features = 4

        self.model = VGAE(VariationalGCNEncoder(num_features, self.EMB_DIM))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        x = self.data.x.to(device)
        self.train_pos_edge_index = self.data.train_pos_edge_index.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(1, self.EPOCHS + 1):
            loss = self.train(x)
            auc, ap = self.test(x)
            print('Epoch: {:03d},Train Loss: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(epoch, loss, auc, ap))

        node_embeddings = np.array(self.model.encode(x,self.edge_index).data.cpu())
        print(node_embeddings)
        with open(f'./data/VariationalGCNEncoder_embeddings_{self.EMB_DIM}_{self.EPOCHS}.pickle', 'wb') as handle:
            pickle.dump(node_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def train(self,x):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(x, self.train_pos_edge_index)
        loss = self.model.recon_loss(z, self.train_pos_edge_index)
        loss = loss + (1 / self.data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

        
    def test(self,x):
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x, self.train_pos_edge_index)
        return self.model.test(z, self.data.test_pos_edge_index, self.data.test_neg_edge_index)

  

def test_geometric():
    import os.path as osp
    import torch
    from torch_geometric.datasets import AMiner
    from torch_geometric.nn import MetaPath2Vec
    dataset = AMiner("./datasets_for_other_tasks")
    data = dataset[0]

    metapath = [
        ('author', 'wrote', 'paper'),
        ('paper', 'published in', 'venue'),
        ('venue', 'published', 'paper'),
        ('paper', 'written by', 'author'),
    ]

    print(data.edge_index_dict)

    for k,v in data.edge_index_dict.items():
        print(v.shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                        metapath=metapath, walk_length=50, context_size=7,
                        walks_per_node=5, num_negative_samples=5,
                        sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=12)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    def train(epoch, log_steps=100, eval_steps=2000):
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

        return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                        max_iter=150)


    for epoch in range(1, 6):
        train(epoch)
        acc = test()
        print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

def visualize_graph():
    import matplotlib.pyplot as plt

    with open("./results/graph.pkl", 'rb') as handle:
        G = pickle.load(handle)

    with open("./data/gravity_masses_128_50_2_1.pickle","rb") as handle:
        masses = pickle.load(handle)

    with open( "./results/node2idx.pkl", 'rb') as handle:
        node2idx = pickle.load(handle)

    res =  list(G.nodes())[:100]
    #pos = nx.spring_layout(G)  #setting the positions with respect to G, not k.
    k = G.subgraph(res)  
    nx.draw_networkx(k, node_color = 'b')
    plt.savefig("./graph.pdf")

if __name__ == "__main__":
    #visualize_graph()

    #node2vec hyperparams
    EMB_DIM = 128
    NODE2VEC_EPOCHS = 50
    combinations = [[20,10,10,1,1,1],[20,10,10,5,1,1],[20,10,10,10,1,1],[30,10,10,1,1,1]]  
    #Train node2vec    
    #ModelNode2Vec(EMB_DIM,NODE2VEC_EPOCHS)
    #TrainGravityModel(EMB_DIM,NODE2VEC_EPOCHS)
    #TrainDeepwalkModel(EMB_DIM,NODE2VEC_EPOCHS)

    #exit()
    for combination in combinations:
        #datasetAutoencoder = DatasetAutoencoder(EMB_DIM,NODE2VEC_EPOCHS) #train graph autoencoder
        #configuration
        #EMB_DIM = 29959 # Diffusion profile
        MLP_MODEL = "MLP" 
        downstream_model= "ada"  #use mlp or no
        MLP_EPOCHS = 25
        NODE_EMBEDDINGS_MODEL = "node2vec" 
        #NODE_EMBEDDINGS_PATH = f'./data/node2vec_embeddings_{EMB_DIM}_{NODE2VEC_EPOCHS}_{str(combination[0])}_{str(combination[1])}_{str(combination[2])}_{str(combination[3])}_{str(combination[4])}_{str(combination[5])}.pickle'
        NODE_EMBEDDINGS_PATH = f'./data/gravity_embeddings_{EMB_DIM}_{NODE2VEC_EPOCHS}_2_1.pickle'
        #NODE_EMBEDDINGS_PATH = f'./data/deepwalk_embeddings_{EMB_DIM}_{NODE2VEC_EPOCHS}_2_1.pickle'

        
        #mlp_model = "MLPSetBilnear"

        print("Node_embeddings_model:", NODE_EMBEDDINGS_MODEL,"Mlp_model:",MLP_MODEL,"EMB_DIM:",EMB_DIM)

        #dataset = Dataset_Mlp(NODE_EMBEDDINGS_MODEL,"./data/node2vec_embeddings_"+str(EMB_DIM)+"_"+str(NODE2VEC_EPOCHS)+".pickle",EMB_DIM,MLP_MODEL)
        dataset = Dataset_Mlp(node_embeddings_model = NODE_EMBEDDINGS_MODEL,
                            node_embeddings_path = NODE_EMBEDDINGS_PATH,
                            EMB_DIM = EMB_DIM,
                            downstream_model = downstream_model,
                            mlp_model = MLP_MODEL,
                            mlp_epochs = MLP_EPOCHS)

        dataset.construct_dataset_from_node_embeddings()
        exit()
