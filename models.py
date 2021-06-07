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


#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec

class ModelNode2Vec():

    def __init__(self,EMB_DIM,EPOCHS):
        super(ModelNode2Vec, self).__init__()
        self.edges = []
        self.EMB_DIM = EMB_DIM
        self.EPOCHS = EPOCHS

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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Node2Vec(self.edges, embedding_dim=self.EMB_DIM, walk_length=20,
                        context_size=10, walks_per_node=10,
                        num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    
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
        with open(f'/data/multiscale-interactome/data/node2vec_embeddings_{self.EMB_DIM}_{self.EPOCHS}.pickle', 'wb') as handle:
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
        self.dropout = nn.Dropout(p=0.1)

        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return torch.sigmoid(x) 


class MLP2(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLP2, self).__init__()      
        self.fc1 = nn.Linear(2*EMB_DIM, EMB_DIM)
        self.fc2=  nn.Linear(EMB_DIM, int(EMB_DIM/2))
        self.fc3=  nn.Linear(int(EMB_DIM/2), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return torch.sigmoid(x)

class MLPSet(nn.Module):
    def __init__(self,EMB_DIM):
        super(MLPSet,self).__init__()
        self.fc1 = nn.Linear(EMB_DIM,50)
        self.fc2 = nn.Linear(50,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.mul(x[:,0],x[:,1])
        x = self.dropout(self.relu(self.fc2(x)))        
        return torch.sigmoid(x)

class MLPSetBilnear(nn.Module):
    '''
    Takes as input the embedding of a drug and the embeddings of disease and predicts if the drug treats the disease.
    '''
    def __init__(self,EMB_DIM):
        super(MLPSetBilnear, self).__init__()
        self.OUT_DIM = 60     
        self.fc1 = nn.Linear(EMB_DIM, self.OUT_DIM) 
        self.fc2 = nn.Bilinear(self.OUT_DIM, self.OUT_DIM, 40)
        self.fc3=  nn.Linear(40, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.EMB_DIM = EMB_DIM

        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x[:,0,:],x[:,1,:])))
        x = self.dropout(self.relu(self.fc3(x)))
        return torch.sigmoid(x)

class Dataset_Mlp(Dataset):
    def __init__(self,node_embeddings_path="./data/node2vec_embeddings_256_200.pickle",EMB_DIM=256,mlp_model=None):
        #self.x = torch.tensor(x,dtype=torch.float32)
        #self.y = torch.tensor(y,dtype=torch.float32)
        #self.length = self.x.shape[0]
        self.node_embeddings_path = node_embeddings_path
        self.EMB_DIM = EMB_DIM
        self.mlp_model = mlp_model
        
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

        #five cross validation
        result_indices = self.kfold_split(pairs=disease_to_drug_dict, perc=0.2, shuffle=True)  # [ ([key1train,key2train,..],[ktest,ktest2]) fold1 , 
                                                                                        #   ([],[])  fold2 ...]
        mean_average_precision = []
        average_recall50 = []
        for indices in result_indices: #for every fold
            train_indices, test_indices = indices

            X_positive_codes_train = [] # [[disease,drug],[disease,drug]...etc] codes of positive examples
            X_positive_codes_test = []
            for key,value in disease_to_drug_dict.items(): #key = disease,  value is a list of ["disease_name","drug_code","drug_name"] cure the disease. 
                if(key in train_indices):
                    for drug in value:
                        X_positive_codes_train.append([key,drug[1]])
                else:
                    for drug in value:
                        X_positive_codes_test.append([key,drug[1]])
            
            #generate negative examples.(we assume that if a pair (disease,drug) is absense from the initial dataset, then the drug do not cure the disease. So its negative example)
            # number_drugs = 1661
            # number_diseases = 840
            # positive examples = 5926
            print("Positive examples generated")
            #for every disease we peak randomly 7=5926/840  drugs until that are not positive examples.
            X_negative_codes_train = []
            X_negative_codes_test = []
            
            for disease in disease_codes_dict:
                counter = 0
                while(counter<7):
                    positive = False
                    #generate random drug code
                    random_key = random.choice(list(drug_codes_dict.keys()))
                    #check if disease,random_key belong to positive example
                    drugs = disease_to_drug_dict[disease]
                    for drug in drugs:
                        if(random_key == drug[1]):
                            positive = True
                    if(positive == False):
                        if(disease in train_indices):
                            X_negative_codes_train.append([disease,random_key])
                        else:
                            X_negative_codes_test.append([disease,random_key])
                        counter+= 1

            print("Negative examples generated")
            #print(len(X_negative_codes)) # this should close to len(X_positive_codes)
            # construct X_input. 
            #for every code take the embedding[code]

            X_positive_train = []
            X_negative_train = []
            X_positive_test = []
            X_negative_test = []
    
            for sample in X_positive_codes_train:
                a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                b = torch.tensor(node_embeddings[node2idx[sample[1]]])
                X_positive_train.append(torch.stack([a,b]))
            
            for sample in X_negative_codes_train:
                a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                b = torch.tensor(node_embeddings[node2idx[sample[1]]])
                X_negative_train.append(torch.stack([a,b]))
            
            for sample in X_positive_codes_test:
                a = torch.tensor(node_embeddings[node2idx[sample[0]]]) 
                b = torch.tensor(node_embeddings[node2idx[sample[1]]])          
                X_positive_test.append(torch.stack([a,b]))

            for sample in X_negative_codes_test:
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

            res = train_MLP(torch.stack((X_train)), torch.stack((X_test)), y_train, y_test, indices, self.node_embeddings_path, self.EMB_DIM, self.mlp_model)
            mean_average_precision.append(res[0])
            average_recall50.append(res[1])
        print("Final average results accross 5 cross validation : ")
        print("Mean average precision:", sum(mean_average_precision) / len(mean_average_precision))
        print("Average recall50:", sum(average_recall50) / len(average_recall50))

def train_MLP(X_train,X_test,y_train,y_test,indices,node_embeddings_path,EMB_DIM,mlp_model):
    from evaluate import evaluate_model
    from evaluate import construct_disease_drug_tsv
    
    if(mlp_model.startswith("MLPSet") == False):
        dataset_train = TensorDataset(torch.reshape(X_train,(-1,2*X_train.shape[-1])),y_train.type(torch.FloatTensor))
        dataset_test =  TensorDataset(torch.reshape(X_test,(-1,2*X_test.shape[-1])),y_test.type(torch.FloatTensor))
    else:
        dataset_train = TensorDataset(X_train,y_train.type(torch.FloatTensor))
        dataset_test =  TensorDataset(X_test,y_test.type(torch.FloatTensor))
        

    #DataLoader
    trainloader = DataLoader(dataset_train,batch_size=8,shuffle=True)
    testloader = DataLoader(dataset_test,batch_size=8,shuffle=True)

    epochs = 30
    model = eval(mlp_model)(EMB_DIM)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    #forward loop
    losses = []
    accur = []
    losses_test = []
    accur_test = []
    n_batches = len(trainloader)
    n_batches_test = len(testloader)
    for i in range(epochs):
        for j,(x_train,y_train) in enumerate(trainloader):
            
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

                output = model(x_test).squeeze()
                #calculate loss
                loss = criterion(output,y_test.squeeze())
                #accuracy
                correct = (torch.round(output) == y_test).float().sum().detach().item()
                acc = correct / y_test.shape[0]
                losses_test.append(loss.detach().item())
                accur_test.append(acc)
            print("Test: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(losses_test[n_batches_test*i:n_batches_test*(i+1)]),np.mean(accur_test[n_batches_test*i:n_batches_test*(i+1)])))        
        
    
    construct_disease_drug_tsv()
    torch.save(model.state_dict(), "./data/MLP_model_"+str(EMB_DIM))
    return evaluate_model(model="node2vec", mlp=True, mlp_model_name=mlp_model, node_embeddings_path=node_embeddings_path, EMB_DIM=EMB_DIM, indices=indices)
    


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
    EMB_DIM = 128
    NODE2VEC_EPOCHS = 100
    #Train node2vec
    #ModelNode2Vec(EMB_DIM,EPOCHS)
    mlp_model= "MLPSetBilnear"
    print("mlp_model",mlp_model,"EMB_DIM",EMB_DIM)

    dataset = Dataset_Mlp("./data/node2vec_embeddings_"+str(EMB_DIM)+"_"+str(NODE2VEC_EPOCHS)+".pickle",EMB_DIM,mlp_model)
    dataset.construct_dataset_from_node_embeddings()
