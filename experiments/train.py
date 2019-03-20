import argparse
import os
import time

import numpy as np
from tqdm import tqdm

import neuralode.adjoint as adj
from experiments.data_helper import get_mnist_loaders, get_cifar_loaders
from experiments.resnet import *


parser = argparse.ArgumentParser()
# train
parser.add_argument('--data', type=str, default='mnist' )
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=6)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', action='store_true')


# ode
parser.add_argument('--use_ode', action='store_true')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--solver', type=str, default='euler')

# logger and saver
parser.add_argument('--save', type=str, default='./result')
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--log_every', type=int, default=10)

args = parser.parse_args()


def train(epoch, train_loader, model, optimizer, device):
    num_items = 0
    train_losses = []
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    print("Training Epoch {}".format(epoch))
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
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

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
        feat = adj.NeuralODE(func, tol=args.tol, solver=args.solver)
    else:
        feat = nn.Sequential(*[ResBlock(64, 64) for _ in range(args.num_res_blocks)])
    model = ContinuousResNet(feat, channels=number_channel).to(device)
    if args.data == 'cifar':
        train_loader, test_loader, _ = get_cifar_loaders(batch_size=args.batch_size, test_batch_size=test_size)
    else:
        train_loader, test_loader, train_eval_loader\
            = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=test_size, perc=1.0)

    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4 if args.weight_decay else 0)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 if args.weight_decay else 0)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=5e-4 if args.weight_decay else 0)
    else:
        raise Exception('Unknown optimizer')

    sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    print('Trained model has {} parameters'.format(get_param_numbers(model)))

    # save all losses, epoch times, accuracy
    train_loss_all = []
    epoch_time_all = []
    accuracy_all = []
    for epoch in range(1, args.max_epochs + 1):
        t_start = time.time()
        train_loss_all.append(train(epoch, train_loader, model, optimizer, device))
        epoch_time_all.append(time.time() - t_start)

        sheduler.step()
        
        accuracy = 0.0
        num_items = 0

        model.eval()
        print("Testing...")
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
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.save, 'model_' + str(epoch) + '.pth'))
        if epoch % args.log_every == 0:
            torch.save({'accuracy': accuracy_all, 
                        'train_loss': train_loss_all,
                        'epoch_time': epoch_time_all}, os.path.join(args.save, 'log.pkl'))
