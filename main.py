import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


import dvi.bayes_utils as bu
from dvi.dataset import ToyDataset
from dvi.bayes_models import MLP, AdaptedMLP
from dvi.loss import GLLLoss


def train(epoch, model, criterion, dataloader, optimizer):
    model.mlp.train()

    for xs, ys in dataloader:
        xs, ys = xs.to(args.device), ys.to(args.device)

        optimizer.zero_grad()

        pred = model(xs)

        kl = model.surprise()

        log_likelihood = criterion(pred, ys)
        batch_log_likelihood = torch.mean(log_likelihood)

        lmbd = bu.anneal(epoch)

        loss = lmbd * kl / args.train_size - batch_log_likelihood

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        accuracy = torch.mean(torch.abs(pred.mean[:, 0].reshape(-1) - ys))
    if epoch % 20 == 0:
        print("Epoch {}: GLL={:.4f}, KL={:.4f}(anneal:{}) | MAE={:.4f}".format(epoch, batch_log_likelihood.item(), kl.item()/500, lmbd, accuracy))


def main():
    trainset = ToyDataset(data_size=args.train_size, sampling=True)
    trainloader = DataLoader(trainset, batch_size=args.train_size, shuffle=True)


    mlp = MLP(args.x_dim, args.y_dim, args.hidden_dims)
    model = AdaptedMLP(mlp, args.adapter, device=args.device)

    criterion = GLLLoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)

    for epoch in range(1, args.epochs+1):
        train(epoch, model, criterion, trainloader, optimizer)
    torch.save(model, 'temp.pth.tar')


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

    parser.add_argument('--epochs', type=int, default=3000,
                        help="Epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--train-size', type=int, default=500,
                        help='Train size (Also Training batch data size)')
    parser.add_argument('--test-size', type=int, default=100,
                        help='Test size (Also Testing batch data size)')

    parser.add_argument('--seed', type=int, default=3,
                        help="Random Seed")

    args = parser.parse_args()
    args.prior_type = ["empirical", "wider_he", "wider_he"]
    args.hidden_dims = [128, 128]
    args.adapter = {
        'in': {"scale": [[1.0]], "shift": [[0.0]]},
        'out': {"scale": [[1.0, 0.83]], "shift": [[0.0, -3.5]]}
    }

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)

    main()

