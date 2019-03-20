import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import neuralode.adjoint as adj
from experiments.data_helper import DataHelper


def norm(dim, type='group_norm'):
    return nn.BatchNorm1d(dim)

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ag_news' )
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embd_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1014)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', action='store_true')


# ode
parser.add_argument('--use_ode', action='store_true')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--solver', type=str, default='euler')

# logger and saver
parser.add_argument('--save', type=str, default='./result_text')
parser.add_argument('--save_every', type=int, default=20)
parser.add_argument('--log_every', type=int, default=1)

args = parser.parse_args()


def conv1d(in_feats, out_feats, stride=1):
    return nn.Conv1d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        return self.conv_block(x) + residual


class ContinuousResNet(nn.Module):
    def __init__(self, feature, vocab_size, embd_size=16, num_labels=4):
        super(ContinuousResNet, self).__init__()
        self.k = 4
        self.embed = nn.Embedding(vocab_size, embd_size)
        self.conv = nn.Conv1d(embd_size, 64, KERNEL_SIZE, STRIDE, PADDING)
        self.feature = feature
        self.norm = norm(64)
        hidden = 256
        self.fc = nn.Sequential(
            nn.Linear(64 * self.k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.feature(x)
        x = self.norm(x)
        x = F.relu(x)
        h = x.topk(self.k)[0].view(-1, 64 * self.k)
        x = self.fc(h)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    use_ode = args.use_ode
    if use_ode:
        func = ConvBlockOde(64)
        feat = adj.NeuralODE(func, tol=args.tol, solver=args.solver)
    else:
        feat = nn.Sequential(*[ConvBlock1D(64, 64) for _ in range(args.num_blocks)])

    sequence_max_length = args.seq_len
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    database_path = '.data/ag_news/'
    data_helper = DataHelper(sequence_max_length=sequence_max_length)
    vocab = len(data_helper.char_dict.keys()) + 2
    model = ContinuousResNet(feat, vocab_size=vocab).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss_all = []

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    epoch_time_all = []
    train_loss_all = []
    accuracy_all = []
    print('Trained model has {} parameters'.format(get_param_numbers(model)))

    for epoch in range(max_epochs):
        train_losses = []

        t_start = time.time()

        train_data, train_label, test_data, test_label = data_helper.load_dataset(database_path)
        train_batches = data_helper.batch_iter(np.column_stack((train_data, train_label)), batch_size, max_epochs)

        train_size = train_data.shape[0] // batch_size
        test_size = test_data.shape[0] // batch_size
        for j, batch in tqdm(enumerate(train_batches), total=train_size):
            train_data_b,label = batch
            train_data_b = torch.from_numpy(train_data_b).to(device)
            label = torch.from_numpy(label).squeeze().to(device)
            model.zero_grad()

            output = model(train_data_b)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses += [loss.item()]

        epoch_time_all.append(time.time() - t_start)
        train_loss_all.append(train_losses)
        print('Train loss: {:.4f}'.format(np.mean(train_losses)))

        test_loader = data_helper.batch_iter(np.column_stack((test_data, test_label)), batch_size, max_epochs)

        model.eval()
        print("Testing...")
        num_items = 0
        accuracy = 0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=test_size):
                data = torch.from_numpy(data).to(device)
                target = torch.from_numpy(target).squeeze().to(device)
                output = model(data)
                accuracy += torch.sum((torch.argmax(output, dim=1) == target).long()).item()
        accuracy = accuracy * 100 / test_data.shape[0]
        print("Accuracy: {}%".format(np.round(accuracy, 3)))
        accuracy_all.append(np.round(accuracy, 3))
        model.train()
        torch.save({'state_dict': model.state_dict()}, os.path.join(args.save, 'model_' + str(epoch) + '.pth'))

        torch.save({'accuracy': accuracy_all,
                    'train_loss': train_loss_all,
                    'epoch_time': epoch_time_all}, os.path.join(args.save, 'log.pkl'))