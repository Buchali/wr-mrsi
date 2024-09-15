from pathlib import Path

import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def download_data(file_name, url):
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    file_path =  data_dir / Path(file_name)
    if not file_path.is_file():
        gdown.download(url=url, output=str(file_path), fuzzy=True)
    else:
        print(f'File {file_name} already exists in /data dir! (please remove it to re-download)')
    return np.load(file_path).T


def load_data(file_names, urls):
    test_data_list = []
    for i, file_name in enumerate(file_names):
        test_data = download_data(file_name, urls[i])
        test_data_list.append(test_data.T)

    test_data = np.vstack(test_data_list)
    return test_data


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


def create_datasets(z, ratio=[0.9, 0.1], generator=torch.manual_seed(42)):
    dataset = CustomDataset(z)
    return random_split(dataset=dataset, lengths=ratio, generator=generator)
