from pathlib import Path

import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def load_data(file_name, url):
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    file_path =  data_dir / Path(file_name)
    if not file_path.is_file():
        gdown.download(url=url, output=str(file_path), fuzzy=True)
    else:
        print(f'File {file_name} already exists in /data dir! (please remove it to re-download)')
    return np.load(file_path).T


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if self.transform:
            x = self.transform(x)
        return x


def create_datasets(z, ratio=[0.8, 0.1, 0.1], generator=torch.manual_seed(42)):
    dataset = CustomDataset(z)
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=ratio, generator=generator)
    return train_dataset, val_dataset, test_dataset


def create_loaders(z, batch_size=512, generator=torch.manual_seed(42)):
    train_dataset, val_dataset, test_dataset = create_datasets(z)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader
