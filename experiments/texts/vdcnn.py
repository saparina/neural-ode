import torch
import torch.nn as nn
import torch.nn.functional as F
import neuralode.adjoint as adj
from experiments.texts.data_helper import DataHelper

import numpy as np

def norm(dim, type='group_norm'):
    return nn.BatchNorm1d(dim)

# maximum length of an input sequence
SEQ_LEN = 1014
BATCH_SIZE = 64
MAX_EPOCH = 18

embd_size = 16
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

save_every = 10

PAD = "<PAD>" # padding
PAD_IDX = 0


def conv1d(in_feats, out_feats, stride=1):
    return nn.Conv1d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    bs, d, s = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs,  1,s)), dim=1)


class ConvBlockOde(adj.OdeWithGrad):
    def __init__(self, dim):
        super(ConvBlockOde, self).__init__()
        # 1 additional dim for time
        self.conv1 = conv1d(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv1d(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride):
        super(ConvBlock1D, self).__init__()
        # architecture
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, KERNEL_SIZE, first_stride, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequential(x)


class ContinuousResNet(nn.Module):
    def __init__(self, feature, vocab_size, embd_size=16, num_labels=4):
        super(ContinuousResNet, self).__init__()
        self.k = 4
        self.embed = nn.Embedding(vocab_size, embd_size)
        self.conv = nn.Conv1d(embd_size, 64, KERNEL_SIZE, STRIDE, PADDING)
        self.feature = feature
        self.norm = norm(64)
        self.fc = nn.Sequential(
            nn.Linear(64 * self.k, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )
        self.softmax = nn.Softmax(1)


    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.feature(x)
        x = self.norm(x)
        x = F.relu(x)
        h = x.topk(self.k)[0].view(-1, 64 * self.k)
        x = self.fc(h)
        out = self.softmax(x)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    func = ConvBlockOde(64)
    feat = adj.NeuralODE(func)
    sequence_max_length = SEQ_LEN
    batch_size = BATCH_SIZE
    num_epochs = MAX_EPOCH
    database_path = '.data/ag_news/'
    data_helper = DataHelper(sequence_max_length=sequence_max_length)
    vocab = len(data_helper.char_dict.keys()) + 2
    ode_res = ContinuousResNet(feat,vocab_size=vocab).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_data, train_label, test_data, test_label = data_helper.load_dataset(database_path)
    train_batches = data_helper.batch_iter(np.column_stack((train_data, train_label)), batch_size, num_epochs)
    optimizer = torch.optim.SGD(ode_res.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    train_losses = []
    print("Test training phase")
    for batch in train_batches:
        train_data_b,label = batch
        train_data_b = torch.from_numpy(train_data_b).to(device)
        label = torch.from_numpy(label).squeeze().to(device)
        output = ode_res(train_data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(loss.item())
        train_losses += [loss.item()]

