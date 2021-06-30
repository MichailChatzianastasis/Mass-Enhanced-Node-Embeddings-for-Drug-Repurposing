import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HIN2vec(nn.Module):

    def __init__(self, node_size, path_size, embed_dim, sigmoid_reg=False, r=True):
        super().__init__()

        # self.args = args

        def binary_reg(x: torch.Tensor):
            raise NotImplementedError()
            # return (x >= 0).float()  # do not have gradients

        self.reg = torch.sigmoid if sigmoid_reg else binary_reg

        self.__initialize_model(node_size, path_size, embed_dim, r)

    def __initialize_model(self, node_size, path_size, embed_dim, r):
        self.start_embeds = nn.Embedding(node_size, embed_dim)
        self.end_embeds = self.start_embeds if r else nn.Embedding(node_size, embed_dim)

        self.path_embeds = nn.Embedding(path_size, embed_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(embed_dim, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, start_node: torch.LongTensor, end_node: torch.LongTensor, path: torch.LongTensor):
        # assert start_node.dim() == 1  # shape = (batch_size,)

        s = self.start_embeds(start_node)  # (batch_size, embed_size)
        e = self.end_embeds(end_node)
        p = self.path_embeds(path)
        p = self.reg(p)

        agg = torch.mul(s, e)
        agg = torch.mul(agg, p)
        # agg = F.sigmoid(agg)
        # output = self.classifier(agg)

        output = torch.sigmoid(torch.sum(agg, axis=1))

        return output



def train(log_interval, model, device, train_loader: DataLoader, optimizer, loss_function, epoch):
    model = model.to(device)
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[:, 0], data[:, 1], data[:, 2])
        loss = loss_function(output.view(-1), target)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print(f'\rTrain Epoch: {epoch} '
                  f'[{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.3f}%)]\t'
                  f'Loss: {loss.item():.3f}\t\t',
                  # f'data = {data}\t target = {target}',
                  end='')
    print()


class NSTrainSet(Dataset):
    """
    完全随机的负采样 todo 改进一下？
    """

    def __init__(self, sample, node_size, neg=5):
        """
        :param node_size: 节点数目
        :param neg: 负采样数目
        :param sample: HIN.sample()返回值，(start_node, end_node, path_id)
        """

        print('init training dataset...')

        l = len(sample)

        x = np.tile(sample, (neg + 1, 1))
        y = np.zeros(l * (1 + neg))
        y[:l] = 1

        # x[l:, 2] = np.random.randint(0, path_size - 1, (l * neg,))
        x[l:, 1] = np.random.randint(0, node_size - 1, (l * neg,))

        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
        self.length = len(x)

        print('finished')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length

