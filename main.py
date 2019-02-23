import argparse

import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import bayes_layers as bnn
from bayes_models import MLP, PointMLP, AdaptedMLP

def base_model(x):
    return - (x + 0.5) * np.sin(3 * np.pi * x)


def noise_model(x):
    return 0.45 * (x + 0.5) ** 2


def sample_data(x):
    return base_model(x) + np.random.normal(0, noise_model(x))


def create_data():
    data_size = {'train': 500, 'valid': 100, 'test': 100}
    toy_data = []
    for section in data_size.keys():
        x = (np.random.rand(data_size[section], 1) - 0.5)
        toy_data.append([x, sample_data(x).reshape(-1)])
    x = np.arange(-1, 1, 1 / 100)
    toy_data.append([[[_] for _ in x], base_model(x)])
    return toy_data


def make_model(args):
    if args.method.lower().strip() == 'bayes':
        MLP_factory = MLP
        prediction = lambda y: y.mean[:, 0].view(-1)
        loss = bnn.regression_loss
    else:
        MLP_factory = PointMLP
        prediction = lambda y: y.mean[:, 0].view(-1)
        loss = bnn.point_regression_loss

    mlp = MLP_factory(args.x_dim, args.y_dim, args)
    mlp = AdaptedMLP(mlp)
    return mlp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deterministic Variational Inference")
    parser.add_argument('--method', type=str, default='bayes',
                        help="Method: bayes|point")
    parser.add_argument('--x-dim', type=int, default=1,
                        help="input dimension")
    parser.add_argument('--y-dim', type=int, default=1,
                        help="output dimension")
    parser.add_argument('--nonlinear', type=str, default='relu',
                        help="Non-Linearity")

    parser.add_argument('--seed', type=int, default=3,
                        help="Random Seed")

    args = parser.parse_args()
    args.prior_type = ["empirical", "wider_he", "wider_he"]
    args.hidden_dims = [128, 128]
    args.adapter = {
        'in': {"scale": [[1.0]], "shift": [[0.0]]},
        'out': {"scale": [[1.0, 0.83]], "shift": [[0.0, -3.5]]}
    }

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)
    print('hidden_dims' in args)

    model = MLP(args.x_dim, args.y_dim, args)
    model = AdaptedMLP(model)

    pseudo_x = torch.randn(32, 1)
    print(model)
    print(model(pseudo_x))

    # main()

