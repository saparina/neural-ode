import argparse
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import neuralode.adjoint as adj
from neuralode.utils import get_mnist_loaders, get_cifar_loaders


parser = argparse.ArgumentParser()
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--use_ode', action='store_true')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--data', type=str, default='mnist' )
parser.add_argument('--save', type=str, default='./mnist_result')
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--log_every', type=int, default=10)

args = parser.parse_args()


def norm(dim, type='group_norm'):
    if type == 'group_norm':
        return nn.GroupNorm(min(32, dim), dim)
    return nn.BatchNorm2d(dim)


def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class ConvBlockOde(adj.OdeWithGrad):
    def __init__(self, dim):
        super(ConvBlockOde, self).__init__()
        # 1 additional dim for time
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class ContinuousResNet(nn.Module):
    def __init__(self, feature, channels=1):
        super(ContinuousResNet, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = feature
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        # batch x 64 x 6 x 6
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out


def train(epoch, train_loader, model, optimizer, device):
    num_items = 0
    train_losses = []
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    print(f"Training Epoch {epoch} ")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        train_losses += [loss.item()]
        num_items += data.shape[0]
    print('Train loss: {:.4f}'.format(np.mean(train_losses)))
    return train_losses

def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    use_continuous_ode = args.use_ode
    print("Use ode training ", use_continuous_ode)
    test_size = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data == 'cifar':
        number_channel = 3
    else:
        number_channel = 1

    if use_continuous_ode:
        func = ConvBlockOde(64)
        feat = adj.NeuralODE(func)
    else:
        feat = nn.Sequential(*[ResBlock(64, 64) for _ in range(6)])
    model = ContinuousResNet(feat, channels=number_channel).to(device)
    if args.data == 'cifar':
        train_loader, test_loader, _ = get_cifar_loaders(batch_size=args.batch_size, test_batch_size=test_size, perc=1.0)
    else:
        train_loader, test_loader, train_eval_loader\
            = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=test_size, perc=1.0)
    optimizer = torch.optim.Adam(model.parameters())
    print('Trained model has {} parameters'.format(get_param_numbers(model)))

    # save all losses, epoch times, accuracy
    train_loss_all = []
    epoch_time_all = []
    accuracy_all = []
    for epoch in range(1, args.max_epochs + 1):
        t_start = time.time()
        train_loss_all.append(train(epoch, train_loader, model, optimizer, device))
        epoch_time_all.append(time.time() - t_start)

        accuracy = 0.0
        num_items = 0

        model.eval()
        print(f"Testing...")
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
                num_items += data.shape[0]
        accuracy = accuracy * 100 / num_items
        print("Accuracy: {}%".format(np.round(accuracy, 3)))
        accuracy_all.append(np.round(accuracy, 3))
        model.train()

        if epoch % args.save_every == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.save, 'model.pth'))
        if epoch % args.log_every == 0:
            torch.save({'accuracy': accuracy_all, 
                        'train_loss': train_loss_all,
                        'epoch_time': epoch_time_all}, os.path.join(args.save, 'log.pkl'))
