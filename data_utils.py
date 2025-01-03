from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def load_data(sub_list: list, data_dir='data'):
    """
    Loads data from a specified directory for a list of subjects.

    Args:
        sub_list (list): A list of subject numbers.
        data_dir (str): The directory where the data is stored.

    Returns:
        np.ndarray: A 2D numpy array containing the loaded data.
    """
    data_dir = Path(data_dir)
    data_list = []
    for sub_num in sub_list:
        file_name = Path(f"HC{sub_num:02d}_M01.npy")
        data_list.append(np.load(data_dir/file_name))
    data = np.vstack(data_list)
    return data


def select_random_subjects(num_picks=2, num_subjects=12):
    """
    Selects a specified number of random subjects from a range of numbers.

    Args:
        num_picks (int): The number of subjects to select.
        num_subjects (int): The total number of subjects in the range.

    Returns:
        tuple: A tuple containing two lists:
            - picked_subjects (list): The list of selected subjects.
            - remaining_subjects (list): The list of remaining subjects.
    """
    numbers = np.arange(1, num_subjects + 1)
    # Ensure num_picks is valid
    if num_picks > len(numbers):
        raise ValueError("num_picks cannot be greater than the number of available subjects.")
    # Select random subjects
    picked_subjects = np.random.choice(numbers, size=num_picks, replace=False)
    # Compute the remaining numbers
    remaining_subjects = np.setdiff1d(numbers, picked_subjects)
    return picked_subjects.tolist(), remaining_subjects.tolist()


def remaining_subjects(sub_list=[1, 2], num_subjects=12):
    """
    Computes the remaining subjects after a list of subjects has been picked.

    Args:
        sub_list (list): A list of subject numbers.
        num_subjects (int): The total number of subjects in the range.

    Returns:
        list: A list of remaining subjects.
    """
    numbers = np.arange(1, num_subjects + 1)
    # Ensure num_picks is valid
    for num in sub_list:
        if num > len(numbers):
            raise ValueError("num_picks cannot be greater than the number of available subjects.")

    # Select random subjects
    picked_subjects = np.array(sub_list)
    # Compute the remaining numbers
    remaining_subjects = np.setdiff1d(numbers, picked_subjects)
    return remaining_subjects.tolist()


class CustomDataset(Dataset):
    """
    Custom dataset class for PyTorch.
    """
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
    """
    Creates datasets for training and validation.
    """
    dataset = CustomDataset(z)
    return random_split(dataset=dataset, lengths=ratio, generator=generator)
