import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DirectionDataset(Dataset):
    def __init__(self, path="dataset_j/raw/hairstyle_d.npy"):
        data = np.load(path,mmap_mode='r')
        self.data = torch.from_numpy(data).float().reshape(len(data), -1)
        self.num_samples = self.data.shape[0]
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0).clamp(min=1e-6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.data[idx] - self.mean) / self.std


def get_dataloader(batch_size=128):
    train_dataset = DirectionDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, train_dataset