import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def _check_none_zero(tensor, shape, device):
    return tensor.to(device) if tensor is not None else torch.zeros(*shape).to(device)


def euler(z0, t0, t1, f, eps=0.05):
    n_steps = np.round((torch.abs(t1 - t0)/eps).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    for i_step in range(int(n_steps)):
        z = z + h * f(z, t)
        t = t + h
    return z


def get_mnist_loaders(batch_size=128, test_batch_size=1000, perc=1.0):
    transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader
