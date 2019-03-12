import numpy as np

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    # data_size = {'train': 500, 'valid': 100, 'test': 100}
    def __init__(self, model=None, noise=None,
                 data_size=500, sampling=True, transform=None):
        if model is None:
            model = self.base_model
        self.model = model

        if noise is None:
            noise = self.noise_model
        self.noise = noise

        self.data_size = data_size

        self.sampling = sampling

        self.transform = transform

        self.create_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        x, y = self.data[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __repr__(self):
        return "Toy Dataset"

    def base_model(self, x):
        return - (x + 0.5) * np.sin(3 * np.pi * x)

    def noise_model(self, x):
        return 0.45 * (x + 0.5) ** 2

    def sample_data(self, x):
        return self.model(x) + np.random.normal(0, self.noise(x))

    def create_data(self):
        if self.sampling:
            xs = np.random.rand(self.data_size, 1) - 0.5
        else:
            xs = np.arange(-1, 1, 1 / 100)
        ys = self.sample_data(xs)

        self.data = [(torch.from_numpy(x).float(), torch.tensor(torch.from_numpy(y).item()).float())
                     for x, y in zip(xs, ys)]

