import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle 
import numpy as np
from walker import load_a_HIN_from_pandas
from model import NSTrainSet, HIN2vec, train


def save_dict_node_embeddings_from_txt(input_txt,output_pkl):
    with open(input_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]     
    node_embeddings = {}
    
    for line in content[1:]:
        n_list = line.split()
        list_of_floats = [float(item) for item in n_list]
        node_embeddings[int(list_of_floats[0])] = torch.FloatTensor(list_of_floats[1:])

    output = open(output_pkl, 'wb')
    pickle.dump(node_embeddings, output)
    output.close() 

 

# set method parameters
window = 4
walk = 10   
walk_length = 100
embed_size = 100
neg = 5
sigmoid_reg = True 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

# set dataset [PLEASE USE YOUR OWN DATASET TO REPLACE THIS]
demo_edge = pd.read_csv('../data/hin2vec.csv', index_col=0)

edges = [demo_edge]

print('finish loading edges')

# init HIN
hin = load_a_HIN_from_pandas(edges)
hin.window = window

print("before train set")
dataset = NSTrainSet(hin.sample(walk_length, walk), hin.node_size, neg=neg)
print("aster dataset")
hin2vec = HIN2vec(hin.node_size, hin.path_size, embed_size, sigmoid_reg)

# load model
# hin2vec.load_state_dict(torch.load('hin2vec.pt'))

# set training parameters
n_epoch = 10
batch_size = 20
log_interval = 200

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.AdamW(hin2vec.parameters())  # 原作者使用的是SGD？ 这里使用AdamW
loss_function = nn.BCELoss()

for epoch in range(n_epoch):
    train(log_interval, hin2vec, device, data_loader, optimizer, loss_function, epoch)

torch.save(hin2vec, 'hin2vec.pt')

# set output parameters [the output file is a bit different from the original code.]
node_vec_fname = 'hin2vec.txt'
# path_vec_fname = 'meta_path_vec.txt'
path_vec_fname = 'hin2vec_metapath.txt'

print(f'saving node embedding vectors to {node_vec_fname}...')
node_embeds = pd.DataFrame(hin2vec.start_embeds.weight.cpu().data.numpy())
node_embeds.rename(hin.id2node).to_csv(node_vec_fname, sep=' ')
save_dict_node_embeddings_from_txt("hin2vec.txt","../data/hin2vec_embeddings_"+str(embed_size)+"_"+str(n_epoch)+".pickle")

if path_vec_fname:
    print(f'saving meta path embedding vectors to {path_vec_fname}...')
    path_embeds = pd.DataFrame(hin2vec.path_embeds.weight.cpu().data.numpy())
    path_embeds.rename(hin.id2path).to_csv(path_vec_fname, sep=' ')
    
# save model
# torch.save(hin2vec.state_dict(), 'hin2vec.pt')